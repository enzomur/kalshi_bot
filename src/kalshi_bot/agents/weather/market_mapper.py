"""Maps Kalshi weather markets to NWS forecast locations and thresholds."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import Any

from kalshi_bot.utils.logging import get_logger

logger = get_logger(__name__)


class WeatherType(str, Enum):
    """Types of weather events tracked."""

    TEMPERATURE = "temperature"
    RAIN = "rain"
    SNOW = "snow"
    WIND = "wind"


class ThresholdDirection(str, Enum):
    """Direction of threshold comparison."""

    ABOVE = "above"
    BELOW = "below"
    BETWEEN = "between"


@dataclass
class WeatherMarketMapping:
    """Parsed information from a Kalshi weather market ticker."""

    ticker: str
    location_code: str
    weather_type: WeatherType
    threshold_value: float
    threshold_direction: ThresholdDirection
    event_date: date
    threshold_high: float | None = None  # For BETWEEN thresholds

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            "ticker": self.ticker,
            "location_code": self.location_code,
            "weather_type": self.weather_type.value,
            "threshold_value": self.threshold_value,
            "threshold_direction": self.threshold_direction.value,
            "event_date": self.event_date.isoformat(),
            "threshold_high": self.threshold_high,
        }


# Kalshi ticker patterns for weather markets
# Examples:
# - HIGHNYC-25APR12-T90 (NYC high temp above 90F on Apr 12, 2025)
# - LOWCHI-25APR12-T35 (Chicago low temp below 35F)
# - RAINNYC-25APR12 (Rain in NYC on Apr 12)

LOCATION_ALIASES = {
    # Ticker abbreviation -> NWS location code
    "NYC": "NYC",
    "NEWYORK": "NYC",
    "NY": "NYC",
    "CHI": "CHI",
    "CHICAGO": "CHI",
    "LAX": "LAX",
    "LA": "LAX",
    "LOSANGELES": "LAX",
    "MIA": "MIA",
    "MIAMI": "MIA",
    "DFW": "DFW",
    "DALLAS": "DFW",
    "PHX": "PHX",
    "PHOENIX": "PHX",
    "HOU": "HOU",
    "HOUSTON": "HOU",
    "ATL": "ATL",
    "ATLANTA": "ATL",
    "BOS": "BOS",
    "BOSTON": "BOS",
    "SEA": "SEA",
    "SEATTLE": "SEA",
    "DEN": "DEN",
    "DENVER": "DEN",
    "PHL": "PHL",
    "PHILLY": "PHL",
    "PHILADELPHIA": "PHL",
    "SFO": "SFO",
    "SF": "SFO",
    "SANFRANCISCO": "SFO",
    "DCA": "DCA",
    "DC": "DCA",
    "WASHINGTON": "DCA",
    "MSP": "MSP",
    "MINNEAPOLIS": "MSP",
}


class WeatherMarketMapper:
    """
    Parses Kalshi weather market tickers to extract location and threshold info.

    Kalshi weather markets follow patterns like:
    - HIGH{CITY}-{DATE}-T{TEMP} - High temperature above threshold
    - LOW{CITY}-{DATE}-T{TEMP} - Low temperature below threshold
    - RAIN{CITY}-{DATE} - Any precipitation
    """

    # Regex patterns for parsing tickers
    TEMP_PATTERN = re.compile(
        r"^(HIGH|LOW)([A-Z]+)-(\d{2})([A-Z]{3})(\d{2})-T(\d+)$",
        re.IGNORECASE,
    )

    RAIN_PATTERN = re.compile(
        r"^RAIN([A-Z]+)-(\d{2})([A-Z]{3})(\d{2})$",
        re.IGNORECASE,
    )

    SNOW_PATTERN = re.compile(
        r"^SNOW([A-Z]+)-(\d{2})([A-Z]{3})(\d{2})(-(\d+))?$",
        re.IGNORECASE,
    )

    # Month abbreviation mapping
    MONTHS = {
        "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4,
        "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8,
        "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
    }

    def parse_ticker(self, ticker: str) -> WeatherMarketMapping | None:
        """
        Parse a Kalshi ticker to extract weather market info.

        Args:
            ticker: Kalshi market ticker

        Returns:
            WeatherMarketMapping or None if not a weather market
        """
        # Try temperature pattern first
        temp_match = self.TEMP_PATTERN.match(ticker)
        if temp_match:
            return self._parse_temp_ticker(ticker, temp_match)

        # Try rain pattern
        rain_match = self.RAIN_PATTERN.match(ticker)
        if rain_match:
            return self._parse_rain_ticker(ticker, rain_match)

        # Try snow pattern
        snow_match = self.SNOW_PATTERN.match(ticker)
        if snow_match:
            return self._parse_snow_ticker(ticker, snow_match)

        return None

    def _parse_temp_ticker(
        self,
        ticker: str,
        match: re.Match,
    ) -> WeatherMarketMapping | None:
        """Parse a temperature market ticker."""
        high_low = match.group(1).upper()
        city_abbr = match.group(2).upper()
        year = int(match.group(3)) + 2000
        month_abbr = match.group(4).upper()
        day = int(match.group(5))
        threshold = int(match.group(6))

        location_code = LOCATION_ALIASES.get(city_abbr)
        if not location_code:
            logger.debug(f"Unknown city abbreviation: {city_abbr}")
            return None

        month = self.MONTHS.get(month_abbr)
        if not month:
            logger.debug(f"Unknown month abbreviation: {month_abbr}")
            return None

        try:
            event_date = date(year, month, day)
        except ValueError:
            logger.debug(f"Invalid date in ticker: {ticker}")
            return None

        return WeatherMarketMapping(
            ticker=ticker,
            location_code=location_code,
            weather_type=WeatherType.TEMPERATURE,
            threshold_value=float(threshold),
            threshold_direction=(
                ThresholdDirection.ABOVE if high_low == "HIGH"
                else ThresholdDirection.BELOW
            ),
            event_date=event_date,
        )

    def _parse_rain_ticker(
        self,
        ticker: str,
        match: re.Match,
    ) -> WeatherMarketMapping | None:
        """Parse a rain market ticker."""
        city_abbr = match.group(1).upper()
        year = int(match.group(2)) + 2000
        month_abbr = match.group(3).upper()
        day = int(match.group(4))

        location_code = LOCATION_ALIASES.get(city_abbr)
        if not location_code:
            return None

        month = self.MONTHS.get(month_abbr)
        if not month:
            return None

        try:
            event_date = date(year, month, day)
        except ValueError:
            return None

        return WeatherMarketMapping(
            ticker=ticker,
            location_code=location_code,
            weather_type=WeatherType.RAIN,
            threshold_value=0.01,  # Any measurable precipitation
            threshold_direction=ThresholdDirection.ABOVE,
            event_date=event_date,
        )

    def _parse_snow_ticker(
        self,
        ticker: str,
        match: re.Match,
    ) -> WeatherMarketMapping | None:
        """Parse a snow market ticker."""
        city_abbr = match.group(1).upper()
        year = int(match.group(2)) + 2000
        month_abbr = match.group(3).upper()
        day = int(match.group(4))
        threshold = match.group(6)  # Optional threshold

        location_code = LOCATION_ALIASES.get(city_abbr)
        if not location_code:
            return None

        month = self.MONTHS.get(month_abbr)
        if not month:
            return None

        try:
            event_date = date(year, month, day)
        except ValueError:
            return None

        return WeatherMarketMapping(
            ticker=ticker,
            location_code=location_code,
            weather_type=WeatherType.SNOW,
            threshold_value=float(threshold) if threshold else 0.1,
            threshold_direction=ThresholdDirection.ABOVE,
            event_date=event_date,
        )

    def is_weather_market(self, ticker: str) -> bool:
        """Check if a ticker is a weather market."""
        return self.parse_ticker(ticker) is not None

    def get_location_from_ticker(self, ticker: str) -> str | None:
        """Extract just the location code from a ticker."""
        mapping = self.parse_ticker(ticker)
        return mapping.location_code if mapping else None
