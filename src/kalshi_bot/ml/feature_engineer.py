"""Feature engineering for ML models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np

from kalshi_bot.utils.logging import get_logger

if TYPE_CHECKING:
    from kalshi_bot.persistence.database import Database
    from kalshi_bot.agents.weather.agent import WeatherResearchAgent

logger = get_logger(__name__)


@dataclass
class MarketFeatures:
    """Computed features for a market at a point in time."""

    ticker: str
    event_ticker: str
    computed_at: datetime

    # Price features
    current_price: float  # Normalized 0-1
    price_momentum_1h: float  # Price change over 1 hour
    price_momentum_6h: float  # Price change over 6 hours
    price_momentum_24h: float  # Price change over 24 hours
    price_volatility: float  # Std dev of recent prices

    # Liquidity features
    spread: float  # Normalized spread
    volume: float  # Log-scaled volume
    open_interest: float  # Log-scaled open interest
    volume_momentum: float  # Change in volume

    # Time features
    hours_to_expiry: float  # Hours until market closes
    is_same_day: bool  # Expires within 24 hours
    day_of_week: int  # 0=Monday, 6=Sunday
    hour_of_day: int  # 0-23

    # Category features (inferred from event_ticker)
    category: str  # politics, sports, economics, weather, other

    # Confidence in features (based on data availability)
    feature_confidence: float = 1.0

    # Weather-specific features (only populated for weather markets)
    nws_probability: float | None = None       # NWS-based probability estimate
    nws_confidence: float | None = None        # Confidence in NWS estimate
    nws_temp_forecast: float | None = None     # Forecasted temp (normalized)
    forecast_hours_out: float | None = None    # Hours until event
    forecast_recency: float | None = None      # How fresh the forecast is

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        result = {
            "ticker": self.ticker,
            "event_ticker": self.event_ticker,
            "computed_at": self.computed_at.isoformat(),
            "current_price": self.current_price,
            "price_momentum_1h": self.price_momentum_1h,
            "price_momentum_6h": self.price_momentum_6h,
            "price_momentum_24h": self.price_momentum_24h,
            "price_volatility": self.price_volatility,
            "spread": self.spread,
            "volume": self.volume,
            "open_interest": self.open_interest,
            "volume_momentum": self.volume_momentum,
            "hours_to_expiry": self.hours_to_expiry,
            "is_same_day": self.is_same_day,
            "day_of_week": self.day_of_week,
            "hour_of_day": self.hour_of_day,
            "category": self.category,
            "feature_confidence": self.feature_confidence,
        }
        # Add weather features if present
        if self.nws_probability is not None:
            result["nws_probability"] = self.nws_probability
            result["nws_confidence"] = self.nws_confidence
            result["nws_temp_forecast"] = self.nws_temp_forecast
            result["forecast_hours_out"] = self.forecast_hours_out
            result["forecast_recency"] = self.forecast_recency
        return result

    def to_array(self, include_weather: bool = True) -> np.ndarray:
        """Convert to numpy array for model input."""
        base_features = [
            self.current_price,
            self.price_momentum_1h,
            self.price_momentum_6h,
            self.price_momentum_24h,
            self.price_volatility,
            self.spread,
            self.volume,
            self.open_interest,
            self.volume_momentum,
            self.hours_to_expiry,
            float(self.is_same_day),
            self.day_of_week / 6.0,  # Normalize to 0-1
            self.hour_of_day / 23.0,  # Normalize to 0-1
            # Category encoding (one-hot would be better, but simple for now)
            float(self.category == "politics"),
            float(self.category == "sports"),
            float(self.category == "economics"),
            float(self.category == "weather"),
        ]

        if include_weather:
            # Add weather features (use 0 for non-weather markets)
            weather_features = [
                self.nws_probability if self.nws_probability is not None else 0.0,
                self.nws_confidence if self.nws_confidence is not None else 0.0,
                self.nws_temp_forecast if self.nws_temp_forecast is not None else 0.0,
                self.forecast_hours_out if self.forecast_hours_out is not None else 0.0,
                self.forecast_recency if self.forecast_recency is not None else 0.0,
            ]
            base_features.extend(weather_features)

        return np.array(base_features)


# Feature names for model interpretation
FEATURE_NAMES = [
    "current_price",
    "price_momentum_1h",
    "price_momentum_6h",
    "price_momentum_24h",
    "price_volatility",
    "spread",
    "volume",
    "open_interest",
    "volume_momentum",
    "hours_to_expiry",
    "is_same_day",
    "day_of_week",
    "hour_of_day",
    "is_politics",
    "is_sports",
    "is_economics",
    "is_weather",
]

# Weather-specific feature names (appended for weather markets)
WEATHER_FEATURE_NAMES = [
    "nws_probability",      # NWS-based probability estimate
    "nws_confidence",       # Confidence in NWS estimate
    "nws_temp_forecast",    # Forecasted temp (normalized)
    "forecast_hours_out",   # Hours until weather event
    "forecast_recency",     # How fresh the forecast is (1.0 = just fetched)
]

# All feature names including weather
ALL_FEATURE_NAMES = FEATURE_NAMES + WEATHER_FEATURE_NAMES


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""

    # Momentum calculation windows (in hours)
    momentum_windows: list[int] = field(default_factory=lambda: [1, 6, 24])

    # Volatility window (number of snapshots)
    volatility_window: int = 12  # ~1 hour at 5-min intervals

    # Minimum snapshots required for reliable features
    min_snapshots: int = 3


class FeatureEngineer:
    """
    Computes features for ML models from market snapshot data.

    Features are designed to capture:
    - Price dynamics (momentum, volatility)
    - Market liquidity (spread, volume, open interest)
    - Temporal patterns (time to expiry, day/hour effects)
    - Market category (politics, sports, etc.)
    - Weather forecasts (for weather markets)
    """

    # Category detection keywords
    CATEGORY_KEYWORDS = {
        "politics": ["president", "election", "congress", "senate", "vote", "trump", "biden", "democrat", "republican"],
        "sports": ["nfl", "nba", "mlb", "nhl", "game", "match", "win", "score", "team", "player"],
        "economics": ["gdp", "cpi", "inflation", "fed", "rate", "jobs", "employment", "economic"],
        "weather": ["weather", "temperature", "rain", "snow", "hurricane", "storm", "high", "low"],
    }

    def __init__(
        self,
        db: Database,
        config: FeatureConfig | None = None,
        weather_agent: "WeatherResearchAgent | None" = None,
    ) -> None:
        """
        Initialize the feature engineer.

        Args:
            db: Database connection
            config: Feature configuration
            weather_agent: Optional weather agent for weather market features
        """
        self._db = db
        self._config = config or FeatureConfig()
        self._weather_agent = weather_agent

    def set_weather_agent(self, agent: "WeatherResearchAgent") -> None:
        """Set the weather agent for weather feature computation."""
        self._weather_agent = agent

    async def compute_features(
        self,
        ticker: str,
        as_of: datetime | None = None,
    ) -> MarketFeatures | None:
        """
        Compute features for a single market.

        Args:
            ticker: Market ticker
            as_of: Time to compute features as of (default: now)

        Returns:
            MarketFeatures or None if insufficient data
        """
        if as_of is None:
            as_of = datetime.utcnow()

        # Get historical snapshots
        snapshots = await self._get_snapshots_before(ticker, as_of, hours=48)

        if len(snapshots) < self._config.min_snapshots:
            logger.debug(f"Insufficient snapshots for {ticker}: {len(snapshots)}")
            return None

        # Get the most recent snapshot
        current = snapshots[0]
        event_ticker = current["event_ticker"]

        # Compute price features
        current_price = self._get_price(current) / 100.0  # Normalize to 0-1
        price_momentum_1h = await self._compute_momentum(snapshots, 1)
        price_momentum_6h = await self._compute_momentum(snapshots, 6)
        price_momentum_24h = await self._compute_momentum(snapshots, 24)
        price_volatility = self._compute_volatility(snapshots)

        # Compute liquidity features
        spread = self._compute_spread(current)
        volume = np.log1p(current.get("volume", 0))  # Log scale
        open_interest = np.log1p(current.get("open_interest", 0))
        volume_momentum = self._compute_volume_momentum(snapshots)

        # Compute time features
        expiration_time = self._parse_datetime(current.get("expiration_time"))
        close_time = self._parse_datetime(current.get("close_time"))
        market_end = expiration_time or close_time

        if market_end:
            hours_to_expiry = max(0, (market_end - as_of).total_seconds() / 3600)
            is_same_day = hours_to_expiry <= 24
        else:
            hours_to_expiry = 168  # Default 1 week if unknown
            is_same_day = False

        # Normalize hours to expiry (cap at 720 hours = 30 days)
        hours_to_expiry = min(hours_to_expiry, 720) / 720.0

        day_of_week = as_of.weekday()
        hour_of_day = as_of.hour

        # Infer category
        category = self._infer_category(event_ticker, current.get("title", ""))

        # Compute feature confidence based on data availability
        confidence = min(1.0, len(snapshots) / 20.0)

        # Initialize weather features
        nws_probability = None
        nws_confidence = None
        nws_temp_forecast = None
        forecast_hours_out = None
        forecast_recency = None

        # Get weather features if this is a weather market and we have a weather agent
        if category == "weather" and self._weather_agent:
            weather_features = await self._weather_agent.get_weather_features(ticker)
            if weather_features:
                nws_probability = weather_features.get("nws_probability")
                nws_confidence = weather_features.get("nws_confidence")
                nws_temp_forecast = weather_features.get("nws_temp_forecast")
                forecast_hours_out = weather_features.get("forecast_hours_out")
                forecast_recency = weather_features.get("forecast_recency")

        return MarketFeatures(
            ticker=ticker,
            event_ticker=event_ticker,
            computed_at=as_of,
            current_price=current_price,
            price_momentum_1h=price_momentum_1h,
            price_momentum_6h=price_momentum_6h,
            price_momentum_24h=price_momentum_24h,
            price_volatility=price_volatility,
            spread=spread,
            volume=volume,
            open_interest=open_interest,
            volume_momentum=volume_momentum,
            hours_to_expiry=hours_to_expiry,
            is_same_day=is_same_day,
            day_of_week=day_of_week,
            hour_of_day=hour_of_day,
            category=category,
            feature_confidence=confidence,
            nws_probability=nws_probability,
            nws_confidence=nws_confidence,
            nws_temp_forecast=nws_temp_forecast,
            forecast_hours_out=forecast_hours_out,
            forecast_recency=forecast_recency,
        )

    async def compute_batch_features(
        self,
        tickers: list[str],
        as_of: datetime | None = None,
    ) -> list[MarketFeatures]:
        """
        Compute features for multiple markets.

        Args:
            tickers: List of market tickers
            as_of: Time to compute features as of

        Returns:
            List of MarketFeatures (excludes markets with insufficient data)
        """
        features = []
        for ticker in tickers:
            market_features = await self.compute_features(ticker, as_of)
            if market_features:
                features.append(market_features)
        return features

    async def compute_training_features(
        self,
        ticker: str,
        settlement_time: datetime,
        hours_before: list[int] | None = None,
    ) -> list[tuple[MarketFeatures, int]]:
        """
        Compute features at multiple time points before settlement.

        Used for training data preparation - creates multiple samples
        from a single settled market.

        Args:
            ticker: Market ticker
            settlement_time: When the market settled
            hours_before: List of hours before settlement to sample

        Returns:
            List of (features, label) tuples where label is 1 for YES, 0 for NO
        """
        if hours_before is None:
            hours_before = [1, 3, 6, 12, 24]

        # Get the outcome
        settlement = await self._db.fetch_one(
            "SELECT outcome FROM market_settlements WHERE ticker = ?",
            (ticker,),
        )

        if not settlement:
            return []

        label = 1 if settlement["outcome"] == "yes" else 0
        results = []

        for hours in hours_before:
            as_of = settlement_time - timedelta(hours=hours)
            features = await self.compute_features(ticker, as_of)
            if features:
                results.append((features, label))

        return results

    async def _get_snapshots_before(
        self,
        ticker: str,
        before: datetime,
        hours: int = 48,
    ) -> list[dict]:
        """Get snapshots before a specific time."""
        after = before - timedelta(hours=hours)
        return await self._db.fetch_all(
            """
            SELECT * FROM market_snapshots
            WHERE ticker = ?
              AND snapshot_at <= ?
              AND snapshot_at >= ?
            ORDER BY snapshot_at DESC
            """,
            (ticker, before.isoformat(), after.isoformat()),
        )

    def _get_price(self, snapshot: dict) -> float:
        """Extract price from snapshot, preferring last_price."""
        last_price = snapshot.get("last_price")
        if last_price is not None:
            return float(last_price)

        # Fall back to mid of bid/ask
        yes_bid = snapshot.get("yes_bid")
        yes_ask = snapshot.get("yes_ask")
        if yes_bid is not None and yes_ask is not None:
            return (yes_bid + yes_ask) / 2.0
        if yes_bid is not None:
            return float(yes_bid)
        if yes_ask is not None:
            return float(yes_ask)

        return 50.0  # Default to 50 if no price data

    async def _compute_momentum(
        self,
        snapshots: list[dict],
        hours: int,
    ) -> float:
        """Compute price momentum over specified hours."""
        if len(snapshots) < 2:
            return 0.0

        current_time = self._parse_datetime(snapshots[0]["snapshot_at"])
        target_time = current_time - timedelta(hours=hours)

        # Find snapshot closest to target time
        best_snapshot = None
        best_diff = float("inf")

        for s in snapshots[1:]:
            snap_time = self._parse_datetime(s["snapshot_at"])
            diff = abs((snap_time - target_time).total_seconds())
            if diff < best_diff:
                best_diff = diff
                best_snapshot = s

        if best_snapshot is None:
            return 0.0

        current_price = self._get_price(snapshots[0])
        past_price = self._get_price(best_snapshot)

        if past_price == 0:
            return 0.0

        # Return normalized momentum (-1 to 1 range roughly)
        momentum = (current_price - past_price) / 100.0
        return max(-1.0, min(1.0, momentum))

    def _compute_volatility(self, snapshots: list[dict]) -> float:
        """Compute price volatility from recent snapshots."""
        prices = [self._get_price(s) for s in snapshots[:self._config.volatility_window]]
        if len(prices) < 2:
            return 0.0

        # Normalize by 100 to get 0-1 scale
        return float(np.std(prices) / 100.0)

    def _compute_spread(self, snapshot: dict) -> float:
        """Compute normalized spread."""
        spread = snapshot.get("spread")
        if spread is not None:
            return min(1.0, spread / 20.0)  # Normalize, cap at 20 cents

        yes_bid = snapshot.get("yes_bid")
        yes_ask = snapshot.get("yes_ask")
        if yes_bid is not None and yes_ask is not None:
            return min(1.0, (yes_ask - yes_bid) / 20.0)

        return 0.5  # Default if unknown

    def _compute_volume_momentum(self, snapshots: list[dict]) -> float:
        """Compute change in trading volume."""
        if len(snapshots) < 2:
            return 0.0

        # Compare recent vs older volume
        recent_volume = sum(s.get("volume", 0) for s in snapshots[:6])
        older_volume = sum(s.get("volume", 0) for s in snapshots[6:12])

        if older_volume == 0:
            return 0.0 if recent_volume == 0 else 1.0

        ratio = recent_volume / older_volume
        # Normalize to -1 to 1 range (1.0 = same volume, 2.0 = doubled, etc.)
        return max(-1.0, min(1.0, (ratio - 1.0)))

    def _infer_category(self, event_ticker: str, title: str = "") -> str:
        """Infer market category from event ticker and title."""
        text = f"{event_ticker} {title}".lower()

        for category, keywords in self.CATEGORY_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                return category

        return "other"

    def _parse_datetime(self, value) -> datetime | None:
        """Parse datetime from various formats, returning naive UTC datetime."""
        if value is None:
            return None
        if isinstance(value, datetime):
            # Strip timezone info if present
            if value.tzinfo is not None:
                return value.replace(tzinfo=None)
            return value
        if isinstance(value, str):
            try:
                # Handle ISO format with various timezone indicators
                clean = value.replace("Z", "+00:00")
                dt = datetime.fromisoformat(clean)
                # Strip timezone info to get naive UTC datetime
                if dt.tzinfo is not None:
                    return dt.replace(tzinfo=None)
                return dt
            except ValueError:
                return None
        return None

    @staticmethod
    def get_feature_names(include_weather: bool = True) -> list[str]:
        """Get list of feature names in array order."""
        if include_weather:
            return ALL_FEATURE_NAMES.copy()
        return FEATURE_NAMES.copy()

    @staticmethod
    def get_weather_feature_names() -> list[str]:
        """Get list of weather-specific feature names."""
        return WEATHER_FEATURE_NAMES.copy()
