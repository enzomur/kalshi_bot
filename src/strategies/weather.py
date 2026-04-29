"""Weather Trading Strategy.

Uses National Weather Service (NWS) forecasts to find edge in weather markets.
When NWS probability differs significantly from market price, we emit a Signal.

Market patterns:
- Temperature: KXHIGHNY-25JAN01-B79.5 (NYC high temp above 79.5F on Jan 1)
- Precipitation: KXRAINNA-25JAN01-B0.1 (Nashville rainfall above 0.1 inches)

NWS provides:
- Temperature forecasts with uncertainty bounds
- Precipitation probability and amounts
"""

from __future__ import annotations

import re
import ssl
import certifi
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.request import urlopen, Request
import json

from src.core.types import Signal
from src.strategies.base import Strategy


# Location codes to NWS grid points
NWS_GRID_POINTS: dict[str, tuple[str, int, int]] = {
    "NYC": ("OKX", 33, 37),  # New York City
    "LAX": ("LOX", 154, 44),  # Los Angeles
    "CHI": ("LOT", 76, 73),  # Chicago
    "MIA": ("MFL", 110, 50),  # Miami
    "DEN": ("BOU", 62, 60),  # Denver
    "SEA": ("SEW", 124, 67),  # Seattle
    "ATL": ("FFC", 52, 88),  # Atlanta
    "BOS": ("BOX", 71, 90),  # Boston
    "DFW": ("FWD", 80, 108),  # Dallas
    "PHX": ("PSR", 160, 59),  # Phoenix
}

# Ticker pattern matchers
TEMP_HIGH_PATTERN = re.compile(r"KXHIGH([A-Z]{2,3})-(\d{2})([A-Z]{3})(\d{2})-[BT]([\d.]+)")
TEMP_LOW_PATTERN = re.compile(r"KXLOW([A-Z]{2,3})-(\d{2})([A-Z]{3})(\d{2})-[BT]([\d.]+)")
PRECIP_PATTERN = re.compile(r"KXRAIN([A-Z]{2,3})-(\d{2})([A-Z]{3})(\d{2})-[BT]([\d.]+)")


@dataclass
class WeatherForecast:
    """NWS forecast data for a location."""

    location: str
    forecast_time: datetime
    high_temp: float | None = None
    low_temp: float | None = None
    high_temp_uncertainty: float = 3.0  # Degrees F
    low_temp_uncertainty: float = 3.0
    precip_probability: float | None = None  # 0-1
    precip_amount: float | None = None  # Inches
    valid_start: datetime | None = None
    valid_end: datetime | None = None


@dataclass
class ParsedWeatherTicker:
    """Parsed weather market ticker."""

    ticker: str
    location: str
    market_type: str  # "high_temp", "low_temp", "precip"
    target_date: datetime
    threshold: float
    is_above: bool  # True = above threshold, False = below


class WeatherStrategy(Strategy):
    """
    Strategy that trades weather markets using NWS forecasts.

    Compares NWS forecast probabilities to market prices and emits
    Signals when there's significant edge.
    """

    # Minimum edge to trade (higher than usual - NWS is authoritative)
    DEFAULT_MIN_EDGE = 0.10

    # Temperature uncertainty increases with days out
    TEMP_UNCERTAINTY_PER_DAY = 2.0  # Additional degrees F per day

    # Only trade markets within this window
    MAX_HOURS_TO_EVENT = 72

    # Price bounds (only trade uncertain markets)
    MIN_PRICE_CENTS = 20
    MAX_PRICE_CENTS = 80

    def __init__(
        self,
        db=None,
        enabled: bool = True,
        min_edge: float = DEFAULT_MIN_EDGE,
        min_confidence: float = 0.60,
        enabled_locations: list[str] | None = None,
    ) -> None:
        """
        Initialize weather strategy.

        Args:
            db: Optional database connection.
            enabled: Whether strategy is active.
            min_edge: Minimum edge to generate signals.
            min_confidence: Minimum confidence threshold.
            enabled_locations: Locations to trade (default: all).
        """
        super().__init__(
            db=db,
            enabled=enabled,
            min_edge=min_edge,
            min_confidence=min_confidence,
        )
        self._enabled_locations = enabled_locations or list(NWS_GRID_POINTS.keys())
        self._forecast_cache: dict[str, WeatherForecast] = {}
        self._ssl_context = ssl.create_default_context(cafile=certifi.where())

    @property
    def name(self) -> str:
        return "weather"

    async def generate_signals(
        self,
        markets: list[dict[str, Any]],
    ) -> list[Signal]:
        """
        Analyze weather markets and generate signals.

        Args:
            markets: Market data from API.

        Returns:
            List of Signal objects.
        """
        if not self._enabled:
            return []

        signals = []
        weather_markets = 0
        analyzed = 0

        for market in markets:
            ticker = market.get("ticker", "")

            # Parse ticker to check if it's a weather market
            parsed = self._parse_ticker(ticker)
            if parsed is None:
                continue

            weather_markets += 1

            # Check if location is enabled
            if parsed.location not in self._enabled_locations:
                continue

            # Check time to event
            hours_to_event = (parsed.target_date - datetime.now(timezone.utc)).total_seconds() / 3600
            if hours_to_event <= 0 or hours_to_event > self.MAX_HOURS_TO_EVENT:
                continue

            analyzed += 1

            # Get market price
            market_price = self._get_market_price(market)
            if market_price is None:
                continue

            # Check price bounds
            if market_price < self.MIN_PRICE_CENTS or market_price > self.MAX_PRICE_CENTS:
                continue

            # Get or fetch forecast
            forecast = await self._get_forecast(parsed.location, parsed.target_date)
            if forecast is None:
                continue

            # Calculate NWS probability
            nws_prob, confidence = self._calculate_probability(parsed, forecast, hours_to_event)
            if nws_prob is None:
                continue

            # Check for edge
            market_prob = market_price / 100.0
            signal = self.create_signal(
                market_ticker=ticker,
                direction="yes" if nws_prob > market_prob else "no",
                target_probability=nws_prob if nws_prob > market_prob else (1 - nws_prob),
                market_probability=market_prob if nws_prob > market_prob else (1 - market_prob),
                confidence=confidence,
                max_position=100,  # Let Risk Engine decide actual size
                metadata={
                    "nws_probability": nws_prob,
                    "forecast_type": parsed.market_type,
                    "threshold": parsed.threshold,
                    "hours_to_event": hours_to_event,
                    "location": parsed.location,
                },
                expires_in_hours=min(hours_to_event, 4.0),  # Expire before event
            )

            if signal is not None:
                signals.append(signal)

        self.update_metrics(
            weather_markets_found=weather_markets,
            markets_analyzed=analyzed,
            signals_generated=len(signals),
        )
        self.record_run()

        return signals

    def _parse_ticker(self, ticker: str) -> ParsedWeatherTicker | None:
        """Parse a weather market ticker."""
        # Try high temperature
        match = TEMP_HIGH_PATTERN.match(ticker)
        if match:
            return self._parse_temp_ticker(ticker, match, "high_temp")

        # Try low temperature
        match = TEMP_LOW_PATTERN.match(ticker)
        if match:
            return self._parse_temp_ticker(ticker, match, "low_temp")

        # Try precipitation
        match = PRECIP_PATTERN.match(ticker)
        if match:
            return self._parse_precip_ticker(ticker, match)

        return None

    def _parse_temp_ticker(
        self, ticker: str, match: re.Match, market_type: str
    ) -> ParsedWeatherTicker | None:
        """Parse a temperature ticker."""
        try:
            location = match.group(1)
            year = 2000 + int(match.group(2))
            month_str = match.group(3)
            day = int(match.group(4))
            threshold = float(match.group(5))

            # Convert month
            months = {
                "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4,
                "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8,
                "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
            }
            month = months.get(month_str)
            if month is None:
                return None

            target_date = datetime(year, month, day, 23, 59, tzinfo=timezone.utc)

            return ParsedWeatherTicker(
                ticker=ticker,
                location=location,
                market_type=market_type,
                target_date=target_date,
                threshold=threshold,
                is_above="B" in ticker or "T" in ticker,  # Above threshold
            )
        except (ValueError, IndexError):
            return None

    def _parse_precip_ticker(
        self, ticker: str, match: re.Match
    ) -> ParsedWeatherTicker | None:
        """Parse a precipitation ticker."""
        try:
            location = match.group(1)
            year = 2000 + int(match.group(2))
            month_str = match.group(3)
            day = int(match.group(4))
            threshold = float(match.group(5))

            months = {
                "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4,
                "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8,
                "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
            }
            month = months.get(month_str)
            if month is None:
                return None

            target_date = datetime(year, month, day, 23, 59, tzinfo=timezone.utc)

            return ParsedWeatherTicker(
                ticker=ticker,
                location=location,
                market_type="precip",
                target_date=target_date,
                threshold=threshold,
                is_above=True,
            )
        except (ValueError, IndexError):
            return None

    def _get_market_price(self, market: dict[str, Any]) -> int | None:
        """Get market price in cents."""
        # Try last_price first
        price = market.get("last_price")
        if price is not None:
            return int(price)

        # Calculate mid from bid/ask
        yes_bid = market.get("yes_bid")
        yes_ask = market.get("yes_ask")
        if yes_bid is not None and yes_ask is not None:
            return (int(yes_bid) + int(yes_ask)) // 2

        return None

    async def _get_forecast(
        self, location: str, target_date: datetime
    ) -> WeatherForecast | None:
        """
        Fetch NWS forecast for a location.

        Caches forecasts for 15 minutes to avoid hammering the API.
        """
        cache_key = f"{location}:{target_date.date()}"

        # Check cache
        if cache_key in self._forecast_cache:
            cached = self._forecast_cache[cache_key]
            cache_age = (datetime.now(timezone.utc) - cached.forecast_time).total_seconds()
            if cache_age < 900:  # 15 minute cache
                return cached

        # Fetch from NWS
        grid_point = NWS_GRID_POINTS.get(location)
        if grid_point is None:
            return None

        office, grid_x, grid_y = grid_point
        url = f"https://api.weather.gov/gridpoints/{office}/{grid_x},{grid_y}/forecast"

        try:
            request = Request(url, headers={"User-Agent": "KalshiBot/1.0"})
            with urlopen(request, timeout=10, context=self._ssl_context) as response:
                data = json.loads(response.read().decode())

            # Parse forecast periods
            forecast = self._parse_nws_response(location, data, target_date)
            if forecast:
                self._forecast_cache[cache_key] = forecast
            return forecast

        except Exception:
            return None

    def _parse_nws_response(
        self, location: str, data: dict, target_date: datetime
    ) -> WeatherForecast | None:
        """Parse NWS API response."""
        periods = data.get("properties", {}).get("periods", [])
        if not periods:
            return None

        # Find the period matching our target date
        target_day = target_date.date()

        for period in periods:
            start_str = period.get("startTime", "")
            if not start_str:
                continue

            try:
                # Parse ISO format
                start_time = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
                if start_time.date() != target_day:
                    continue

                is_daytime = period.get("isDaytime", True)
                temp = period.get("temperature")

                forecast = WeatherForecast(
                    location=location,
                    forecast_time=datetime.now(timezone.utc),
                    valid_start=start_time,
                )

                if is_daytime:
                    forecast.high_temp = float(temp) if temp else None
                else:
                    forecast.low_temp = float(temp) if temp else None

                # Parse precipitation probability
                prob_precip = period.get("probabilityOfPrecipitation", {})
                if isinstance(prob_precip, dict):
                    precip_val = prob_precip.get("value")
                    if precip_val is not None:
                        forecast.precip_probability = float(precip_val) / 100.0

                return forecast

            except (ValueError, TypeError):
                continue

        return None

    def _calculate_probability(
        self,
        parsed: ParsedWeatherTicker,
        forecast: WeatherForecast,
        hours_to_event: float,
    ) -> tuple[float | None, float]:
        """
        Calculate probability from NWS forecast.

        Returns (probability, confidence) tuple.
        """
        if parsed.market_type == "high_temp":
            return self._calc_temp_probability(
                forecast.high_temp,
                parsed.threshold,
                hours_to_event,
                forecast.high_temp_uncertainty,
            )
        elif parsed.market_type == "low_temp":
            return self._calc_temp_probability(
                forecast.low_temp,
                parsed.threshold,
                hours_to_event,
                forecast.low_temp_uncertainty,
            )
        elif parsed.market_type == "precip":
            return self._calc_precip_probability(
                forecast.precip_probability,
                parsed.threshold,
            )

        return None, 0.0

    def _calc_temp_probability(
        self,
        forecast_temp: float | None,
        threshold: float,
        hours_to_event: float,
        base_uncertainty: float,
    ) -> tuple[float | None, float]:
        """
        Calculate probability of temperature exceeding threshold.

        Uses normal CDF approximation with uncertainty that increases
        with time to event.
        """
        if forecast_temp is None:
            return None, 0.0

        # Uncertainty increases with days out
        days_out = hours_to_event / 24.0
        uncertainty = base_uncertainty + (days_out * self.TEMP_UNCERTAINTY_PER_DAY)

        # Calculate z-score
        z = (threshold - forecast_temp) / uncertainty

        # Normal CDF approximation (probability of exceeding threshold)
        # P(X > threshold) = 1 - Phi(z)
        prob_above = self._normal_cdf_complement(z)

        # Confidence decreases with uncertainty
        confidence = max(0.5, 1.0 - (uncertainty / 15.0))

        return prob_above, confidence

    def _calc_precip_probability(
        self,
        precip_prob: float | None,
        threshold: float,
    ) -> tuple[float | None, float]:
        """
        Calculate probability of precipitation exceeding threshold.

        NWS gives probability of measurable precipitation (>0.01").
        For higher thresholds, we discount accordingly.
        """
        if precip_prob is None:
            return None, 0.0

        # For trace amounts (>0.01"), use NWS probability directly
        if threshold <= 0.05:
            return precip_prob, 0.75

        # For higher amounts, discount probability
        # This is approximate - heavier rain is less likely
        if threshold <= 0.25:
            prob = precip_prob * 0.6
        elif threshold <= 0.50:
            prob = precip_prob * 0.4
        else:
            prob = precip_prob * 0.25

        return prob, 0.60

    def _normal_cdf_complement(self, z: float) -> float:
        """
        Approximate P(Z > z) for standard normal Z.

        Uses Abramowitz & Stegun approximation.
        """
        import math

        # Handle extreme values
        if z < -5:
            return 1.0
        if z > 5:
            return 0.0

        # Approximation constants
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        p = 0.3275911

        sign = 1 if z >= 0 else -1
        z = abs(z)

        t = 1.0 / (1.0 + p * z)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-z * z / 2)

        cdf = 0.5 * (1.0 + sign * y)
        return 1.0 - cdf  # Return complement (probability of exceeding)
