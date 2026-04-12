"""Weather Research Agent - fetches NWS forecasts and generates probability estimates."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from kalshi_bot.agents.base import BaseAgent
from kalshi_bot.agents.weather.market_mapper import WeatherMarketMapper
from kalshi_bot.agents.weather.nws_client import NWSClient
from kalshi_bot.agents.weather.probability_calc import (
    ProbabilityEstimate,
    WeatherProbabilityCalculator,
)
from kalshi_bot.utils.logging import get_logger

if TYPE_CHECKING:
    from kalshi_bot.api.client import KalshiAPIClient
    from kalshi_bot.persistence.database import Database

logger = get_logger(__name__)


class WeatherResearchAgent(BaseAgent):
    """
    Agent that fetches NWS forecasts and generates probability estimates
    for Kalshi weather markets.

    Runs periodically to:
    1. Fetch current forecasts from NWS for enabled locations
    2. Store forecasts in the database
    3. Calculate probability estimates for active weather markets
    4. Provide features to the EdgePredictor
    """

    def __init__(
        self,
        db: "Database",
        api_client: "KalshiAPIClient",
        enabled_locations: list[str] | None = None,
        update_interval_minutes: int = 15,
        enabled: bool = True,
    ) -> None:
        """
        Initialize the Weather Research Agent.

        Args:
            db: Database connection
            api_client: Kalshi API client for fetching markets
            enabled_locations: List of location codes to track
            update_interval_minutes: How often to fetch forecasts
            enabled: Whether the agent is enabled
        """
        super().__init__(
            db=db,
            name="weather_research",
            update_interval_seconds=update_interval_minutes * 60,
            enabled=enabled,
        )

        self._api_client = api_client
        self._enabled_locations = enabled_locations or ["NYC", "CHI", "LAX", "MIA"]

        self._nws_client = NWSClient()
        self._market_mapper = WeatherMarketMapper()
        self._probability_calc = WeatherProbabilityCalculator()

        # Cache for recent probability estimates
        self._probability_cache: dict[str, ProbabilityEstimate] = {}
        self._cache_expiry: datetime | None = None

    async def _run_cycle(self) -> None:
        """Execute one cycle of weather data collection."""
        logger.info(f"Weather agent: fetching forecasts for {self._enabled_locations}")

        # Fetch forecasts from NWS
        forecasts = await self._nws_client.get_forecasts_batch(self._enabled_locations)

        if not forecasts:
            logger.warning("Weather agent: no forecasts received")
            return

        # Store forecasts in database
        await self._store_forecasts(forecasts)

        # Get active weather markets from Kalshi
        weather_markets = await self._get_weather_markets()

        if not weather_markets:
            logger.debug("Weather agent: no active weather markets found")
            return

        # Calculate probabilities for each market
        estimates = []
        for ticker, mapping in weather_markets.items():
            forecast = forecasts.get(mapping.location_code)
            if not forecast:
                continue

            estimate = self._probability_calc.calculate_probability(mapping, forecast)
            if estimate:
                estimates.append(estimate)
                self._probability_cache[ticker] = estimate

        # Store probability estimates
        await self._store_probability_estimates(estimates)
        self._cache_expiry = datetime.utcnow() + timedelta(minutes=15)

        # Update metrics
        self._status.metrics = {
            "forecasts_fetched": len(forecasts),
            "markets_analyzed": len(weather_markets),
            "estimates_generated": len(estimates),
            "locations": list(forecasts.keys()),
        }

        logger.info(
            f"Weather agent: fetched {len(forecasts)} forecasts, "
            f"generated {len(estimates)} probability estimates"
        )

    async def _store_forecasts(self, forecasts: dict) -> None:
        """Store forecasts in the database."""
        for location_code, forecast in forecasts.items():
            for period in forecast.periods:
                await self._db.execute(
                    """
                    INSERT OR REPLACE INTO weather_forecasts
                    (location_code, forecast_time, temperature_f, precipitation_prob,
                     confidence, fetched_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        location_code,
                        period.start_time.isoformat(),
                        period.temperature,
                        period.precipitation_probability or 0,
                        1.0,  # NWS forecasts are our primary source
                        datetime.utcnow().isoformat(),
                    ),
                )

    async def _get_weather_markets(self) -> dict:
        """Get active weather markets from Kalshi."""
        try:
            # Fetch markets from Kalshi API
            response = await self._api_client.get_events(
                status="open",
                limit=200,
            )
        except Exception as e:
            logger.error(f"Failed to fetch markets: {e}")
            return {}

        weather_markets = {}

        # Extract events list from response dict
        events = response.get("events", []) if isinstance(response, dict) else []
        for event in events:
            markets = event.get("markets", [])
            for market in markets:
                ticker = market.get("ticker", "")
                mapping = self._market_mapper.parse_ticker(ticker)

                if mapping and mapping.location_code in self._enabled_locations:
                    weather_markets[ticker] = mapping

        return weather_markets

    async def _store_probability_estimates(
        self,
        estimates: list[ProbabilityEstimate],
    ) -> None:
        """Store probability estimates in the database."""
        for estimate in estimates:
            # Get the mapping for this estimate
            mapping = self._market_mapper.parse_ticker(estimate.ticker)
            if not mapping:
                continue

            await self._db.execute(
                """
                INSERT OR REPLACE INTO weather_market_mappings
                (ticker, location_code, weather_type, threshold_value,
                 threshold_direction, event_date, agent_probability, agent_confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    estimate.ticker,
                    mapping.location_code,
                    mapping.weather_type.value,
                    mapping.threshold_value,
                    mapping.threshold_direction.value,
                    mapping.event_date.isoformat(),
                    estimate.probability,
                    estimate.confidence,
                ),
            )

    def get_probability(self, ticker: str) -> ProbabilityEstimate | None:
        """
        Get cached probability estimate for a ticker.

        Args:
            ticker: Market ticker

        Returns:
            ProbabilityEstimate or None if not available
        """
        if self._cache_expiry and datetime.utcnow() > self._cache_expiry:
            self._probability_cache.clear()
            return None

        return self._probability_cache.get(ticker)

    async def get_weather_features(self, ticker: str) -> dict[str, float] | None:
        """
        Get weather-specific features for ML model.

        Args:
            ticker: Market ticker

        Returns:
            Dict of weather features or None if not a weather market
        """
        estimate = self.get_probability(ticker)
        if not estimate:
            # Try to fetch from database
            row = await self._db.fetch_one(
                """
                SELECT agent_probability, agent_confidence
                FROM weather_market_mappings
                WHERE ticker = ?
                """,
                (ticker,),
            )
            if not row:
                return None

            return {
                "nws_probability": row["agent_probability"],
                "nws_confidence": row["agent_confidence"],
                "nws_temp_forecast": 0.0,  # Not available from DB
                "forecast_hours_out": 0.0,
                "forecast_recency": 0.0,
            }

        return {
            "nws_probability": estimate.probability,
            "nws_confidence": estimate.confidence,
            "nws_temp_forecast": (estimate.forecast_temp or 70) / 100.0,  # Normalize
            "forecast_hours_out": min(estimate.hours_until_event, 168) / 168.0,  # Normalize to 7 days
            "forecast_recency": 1.0,  # Fresh from cache
        }

    def is_weather_market(self, ticker: str) -> bool:
        """Check if a ticker is a weather market."""
        return self._market_mapper.is_weather_market(ticker)

    async def close(self) -> None:
        """Clean up resources."""
        await self._nws_client.close()

    def get_status(self) -> dict[str, Any]:
        """Get agent status with weather-specific metrics."""
        status = super().get_status()
        status["cache_size"] = len(self._probability_cache)
        status["cache_valid"] = (
            self._cache_expiry is not None and
            datetime.utcnow() < self._cache_expiry
        )
        return status
