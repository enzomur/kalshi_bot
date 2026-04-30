"""Mean reversion strategy for prediction markets.

This strategy detects when market prices deviate significantly from their
historical moving average and signals trades expecting reversion to the mean.

Key assumptions:
1. Markets with sufficient history exhibit mean-reverting behavior
2. Extreme price moves (>2 std dev) are likely to revert
3. Mean reversion is stronger in markets far from settlement

Usage:
    strategy = MeanReversionStrategy(
        lookback_periods=24,  # Hours of history
        z_score_threshold=2.0,  # Standard deviations
    )
    signals = await strategy.generate_signals(markets)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from statistics import mean, stdev
from typing import TYPE_CHECKING, Any

from src.core.types import Signal
from src.observability.logging import get_logger
from src.strategies.base import Strategy

if TYPE_CHECKING:
    from src.ledger.database import Database

logger = get_logger(__name__)


@dataclass
class PriceHistory:
    """Tracks price history for a single market."""

    market_ticker: str
    prices: list[float] = field(default_factory=list)
    timestamps: list[datetime] = field(default_factory=list)
    max_history: int = 168  # 1 week of hourly data

    def add_price(self, price: float, timestamp: datetime | None = None) -> None:
        """Add a new price observation."""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        self.prices.append(price)
        self.timestamps.append(timestamp)

        # Trim to max history
        if len(self.prices) > self.max_history:
            self.prices = self.prices[-self.max_history :]
            self.timestamps = self.timestamps[-self.max_history :]

    def get_recent(self, periods: int) -> list[float]:
        """Get the most recent N prices."""
        return self.prices[-periods:] if len(self.prices) >= periods else self.prices

    @property
    def current_price(self) -> float | None:
        """Get the most recent price."""
        return self.prices[-1] if self.prices else None

    @property
    def sample_count(self) -> int:
        """Number of price samples."""
        return len(self.prices)


@dataclass
class MeanReversionSignalData:
    """Data about a mean reversion opportunity."""

    market_ticker: str
    current_price: float
    moving_average: float
    standard_deviation: float
    z_score: float
    direction: str  # "yes" = expect price to rise, "no" = expect price to fall
    expected_reversion: float  # Target price after reversion
    sample_count: int
    lookback_hours: int


class MeanReversionStrategy(Strategy):
    """
    Mean reversion strategy for prediction markets.

    Tracks price history for each market and identifies opportunities when
    prices deviate significantly from their historical moving average.

    Parameters:
        lookback_periods: Number of periods for moving average (default: 24 hours)
        z_score_threshold: Standard deviations required for signal (default: 2.0)
        min_samples: Minimum samples required before trading (default: 12)
        max_hours_to_settlement: Ignore markets settling too soon (default: 72)
        reversion_target: Fraction of reversion to expect (default: 0.5 = 50%)
    """

    # Strategy defaults
    DEFAULT_LOOKBACK = 24
    DEFAULT_Z_THRESHOLD = 2.0
    DEFAULT_MIN_SAMPLES = 12
    DEFAULT_MAX_HOURS = 72.0
    DEFAULT_REVERSION = 0.5

    def __init__(
        self,
        db: Database | None = None,
        enabled: bool = True,
        min_edge: float = 0.05,
        min_confidence: float = 0.55,
        lookback_periods: int = DEFAULT_LOOKBACK,
        z_score_threshold: float = DEFAULT_Z_THRESHOLD,
        min_samples: int = DEFAULT_MIN_SAMPLES,
        max_hours_to_settlement: float = DEFAULT_MAX_HOURS,
        reversion_target: float = DEFAULT_REVERSION,
    ) -> None:
        """Initialize the mean reversion strategy."""
        super().__init__(
            db=db,
            enabled=enabled,
            min_edge=min_edge,
            min_confidence=min_confidence,
        )
        self._lookback_periods = lookback_periods
        self._z_score_threshold = z_score_threshold
        self._min_samples = min_samples
        self._max_hours_to_settlement = max_hours_to_settlement
        self._reversion_target = reversion_target

        # Price history storage
        self._price_history: dict[str, PriceHistory] = defaultdict(
            lambda: PriceHistory(market_ticker="")
        )

        # Statistics
        self._markets_analyzed = 0
        self._opportunities_found = 0

    @property
    def name(self) -> str:
        """Strategy name."""
        return "mean_reversion"

    def update_price(
        self, market_ticker: str, price: float, timestamp: datetime | None = None
    ) -> None:
        """
        Update price history for a market.

        Call this method with each price update to build history.

        Args:
            market_ticker: Market identifier
            price: Current price (0-100 cents or 0-1 probability)
            timestamp: Optional timestamp (defaults to now)
        """
        # Normalize price to 0-100 range
        if price <= 1:
            price = price * 100

        if market_ticker not in self._price_history:
            self._price_history[market_ticker] = PriceHistory(
                market_ticker=market_ticker
            )

        self._price_history[market_ticker].add_price(price, timestamp)

    def get_price_history(self, market_ticker: str) -> PriceHistory | None:
        """Get price history for a market."""
        return self._price_history.get(market_ticker)

    async def generate_signals(
        self,
        markets: list[dict[str, Any]],
    ) -> list[Signal]:
        """
        Generate mean reversion signals from market data.

        Args:
            markets: List of market data dictionaries

        Returns:
            List of Signal objects for mean reversion opportunities
        """
        if not self.enabled:
            return []

        signals: list[Signal] = []
        self._markets_analyzed = 0
        self._opportunities_found = 0

        for market in markets:
            try:
                signal = self._analyze_market(market)
                if signal:
                    signals.append(signal)
                    self._opportunities_found += 1
            except Exception as e:
                logger.warning(f"Error analyzing market {market.get('ticker')}: {e}")

        self.record_run()
        self.update_metrics(
            markets_analyzed=self._markets_analyzed,
            opportunities_found=self._opportunities_found,
            markets_with_history=len(self._price_history),
        )

        if signals:
            logger.info(
                f"Mean reversion: {len(signals)} signals from "
                f"{self._markets_analyzed} markets analyzed"
            )

        return signals

    def _analyze_market(self, market: dict[str, Any]) -> Signal | None:
        """Analyze a single market for mean reversion opportunities."""
        ticker = market.get("ticker", "")
        if not ticker:
            return None

        self._markets_analyzed += 1

        # Get current price
        yes_bid = market.get("yes_bid")
        yes_ask = market.get("yes_ask")

        if yes_bid is None or yes_ask is None:
            return None

        current_price = (yes_bid + yes_ask) / 2

        # Update price history
        self.update_price(ticker, current_price)

        # Check if market is settling too soon
        close_time_str = market.get("close_time")
        if close_time_str:
            try:
                close_time = datetime.fromisoformat(
                    close_time_str.replace("Z", "+00:00")
                )
                hours_to_close = (
                    close_time - datetime.now(timezone.utc)
                ).total_seconds() / 3600

                if hours_to_close < 0 or hours_to_close > self._max_hours_to_settlement:
                    return None
            except (ValueError, TypeError):
                pass

        # Get price history
        history = self._price_history.get(ticker)
        if not history or history.sample_count < self._min_samples:
            return None

        # Get lookback prices
        recent_prices = history.get_recent(self._lookback_periods)
        if len(recent_prices) < self._min_samples:
            return None

        # Calculate statistics
        avg_price = mean(recent_prices)
        if len(recent_prices) < 2:
            return None

        std_price = stdev(recent_prices)
        if std_price < 0.5:  # Very low volatility, skip
            return None

        # Calculate z-score
        z_score = (current_price - avg_price) / std_price

        # Check if z-score exceeds threshold
        if abs(z_score) < self._z_score_threshold:
            return None

        # Determine direction and expected reversion
        if z_score > self._z_score_threshold:
            # Price is high, expect it to fall (buy NO)
            direction = "no"
            expected_reversion = avg_price + (current_price - avg_price) * (
                1 - self._reversion_target
            )
        else:
            # Price is low, expect it to rise (buy YES)
            direction = "yes"
            expected_reversion = avg_price - (avg_price - current_price) * (
                1 - self._reversion_target
            )

        # Calculate edge and confidence
        price_diff = abs(current_price - expected_reversion)
        edge = price_diff / 100  # Convert to probability

        # Confidence based on z-score strength and sample count
        z_confidence = min(abs(z_score) / 4.0, 1.0)  # Max confidence at z=4
        sample_confidence = min(history.sample_count / 48, 1.0)  # Max at 48 samples
        confidence = (z_confidence + sample_confidence) / 2

        # Calculate target and market probabilities
        market_probability = current_price / 100
        if direction == "yes":
            target_probability = expected_reversion / 100
        else:
            target_probability = (100 - expected_reversion) / 100

        # Create signal data
        signal_data = MeanReversionSignalData(
            market_ticker=ticker,
            current_price=current_price,
            moving_average=avg_price,
            standard_deviation=std_price,
            z_score=z_score,
            direction=direction,
            expected_reversion=expected_reversion,
            sample_count=history.sample_count,
            lookback_hours=self._lookback_periods,
        )

        # Create signal using base class helper
        signal = self.create_signal(
            market_ticker=ticker,
            direction=direction,
            target_probability=target_probability,
            market_probability=market_probability,
            confidence=confidence,
            max_position=25,  # Conservative position size
            metadata={
                "strategy_type": "mean_reversion",
                "z_score": round(z_score, 2),
                "moving_average": round(avg_price, 2),
                "std_deviation": round(std_price, 2),
                "expected_reversion": round(expected_reversion, 2),
                "sample_count": history.sample_count,
                "lookback_periods": self._lookback_periods,
            },
            expires_in_hours=2.0,  # Short expiry for mean reversion
        )

        if signal:
            logger.debug(
                f"Mean reversion signal: {ticker} {direction} "
                f"(z={z_score:.2f}, price={current_price:.0f}, avg={avg_price:.0f})"
            )

        return signal

    def calculate_z_score(self, market_ticker: str) -> float | None:
        """
        Calculate current z-score for a market.

        Useful for monitoring and debugging.

        Args:
            market_ticker: Market identifier

        Returns:
            Z-score or None if insufficient history
        """
        history = self._price_history.get(market_ticker)
        if not history or history.sample_count < self._min_samples:
            return None

        recent_prices = history.get_recent(self._lookback_periods)
        if len(recent_prices) < self._min_samples:
            return None

        avg_price = mean(recent_prices)
        std_price = stdev(recent_prices) if len(recent_prices) > 1 else 0

        if std_price == 0:
            return 0.0

        current = history.current_price
        if current is None:
            return None

        return (current - avg_price) / std_price

    def get_market_stats(self, market_ticker: str) -> dict[str, Any] | None:
        """
        Get statistics for a market.

        Args:
            market_ticker: Market identifier

        Returns:
            Statistics dictionary or None
        """
        history = self._price_history.get(market_ticker)
        if not history or history.sample_count < 2:
            return None

        recent_prices = history.get_recent(self._lookback_periods)
        if len(recent_prices) < 2:
            return None

        return {
            "ticker": market_ticker,
            "current_price": history.current_price,
            "sample_count": history.sample_count,
            "moving_average": round(mean(recent_prices), 2),
            "std_deviation": round(stdev(recent_prices), 2),
            "z_score": self.calculate_z_score(market_ticker),
            "min_price": min(recent_prices),
            "max_price": max(recent_prices),
        }

    def clear_history(self, market_ticker: str | None = None) -> None:
        """
        Clear price history.

        Args:
            market_ticker: Specific market to clear, or None for all
        """
        if market_ticker:
            self._price_history.pop(market_ticker, None)
        else:
            self._price_history.clear()

    def load_historical_prices(
        self, market_ticker: str, prices: list[tuple[float, datetime]]
    ) -> None:
        """
        Bulk load historical prices for backtesting.

        Args:
            market_ticker: Market identifier
            prices: List of (price, timestamp) tuples, oldest first
        """
        history = PriceHistory(market_ticker=market_ticker)
        for price, timestamp in prices:
            history.add_price(price, timestamp)

        self._price_history[market_ticker] = history

    def get_status(self) -> dict[str, Any]:
        """Get strategy status including mean reversion specific metrics."""
        base_status = super().get_status()
        base_status["metrics"].update(
            {
                "lookback_periods": self._lookback_periods,
                "z_score_threshold": self._z_score_threshold,
                "markets_tracked": len(self._price_history),
                "total_samples": sum(
                    h.sample_count for h in self._price_history.values()
                ),
            }
        )
        return base_status

    @classmethod
    def from_config(cls, config: dict[str, Any], db: Database | None = None) -> MeanReversionStrategy:
        """
        Create a MeanReversionStrategy from configuration.

        Args:
            config: Configuration dictionary
            db: Optional database connection

        Returns:
            Configured MeanReversionStrategy instance
        """
        return cls(
            db=db,
            enabled=config.get("enabled", True),
            min_edge=config.get("min_edge", 0.05),
            min_confidence=config.get("min_confidence", 0.55),
            lookback_periods=config.get("lookback_periods", cls.DEFAULT_LOOKBACK),
            z_score_threshold=config.get("z_score_threshold", cls.DEFAULT_Z_THRESHOLD),
            min_samples=config.get("min_samples", cls.DEFAULT_MIN_SAMPLES),
            max_hours_to_settlement=config.get(
                "max_hours_to_settlement", cls.DEFAULT_MAX_HOURS
            ),
            reversion_target=config.get("reversion_target", cls.DEFAULT_REVERSION),
        )
