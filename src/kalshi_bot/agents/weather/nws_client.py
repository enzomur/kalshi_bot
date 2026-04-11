"""NWS (National Weather Service) API client."""

from __future__ import annotations

import asyncio
import ssl
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import aiohttp
import certifi

from kalshi_bot.utils.logging import get_logger

logger = get_logger(__name__)


# Major city coordinates for NWS API
CITY_COORDINATES = {
    "NYC": (40.7128, -74.0060),   # New York City
    "CHI": (41.8781, -87.6298),   # Chicago
    "LAX": (34.0522, -118.2437),  # Los Angeles
    "MIA": (25.7617, -80.1918),   # Miami
    "DFW": (32.7767, -96.7970),   # Dallas-Fort Worth
    "PHX": (33.4484, -112.0740),  # Phoenix
    "HOU": (29.7604, -95.3698),   # Houston
    "ATL": (33.7490, -84.3880),   # Atlanta
    "BOS": (42.3601, -71.0589),   # Boston
    "SEA": (47.6062, -122.3321),  # Seattle
    "DEN": (39.7392, -104.9903),  # Denver
    "PHL": (39.9526, -75.1652),   # Philadelphia
    "SFO": (37.7749, -122.4194),  # San Francisco
    "DCA": (38.9072, -77.0369),   # Washington DC
    "MSP": (44.9778, -93.2650),   # Minneapolis
}


@dataclass
class ForecastPeriod:
    """A single forecast period from NWS."""

    name: str  # e.g., "Tonight", "Monday", "Monday Night"
    start_time: datetime
    end_time: datetime
    temperature: int  # Fahrenheit
    temperature_unit: str
    is_daytime: bool
    wind_speed: str
    wind_direction: str
    short_forecast: str
    detailed_forecast: str
    precipitation_probability: int | None  # 0-100 percentage

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> ForecastPeriod:
        """Create from NWS API response."""
        return cls(
            name=data.get("name", ""),
            start_time=datetime.fromisoformat(data["startTime"].replace("Z", "+00:00")).replace(tzinfo=None),
            end_time=datetime.fromisoformat(data["endTime"].replace("Z", "+00:00")).replace(tzinfo=None),
            temperature=data.get("temperature", 0),
            temperature_unit=data.get("temperatureUnit", "F"),
            is_daytime=data.get("isDaytime", True),
            wind_speed=data.get("windSpeed", ""),
            wind_direction=data.get("windDirection", ""),
            short_forecast=data.get("shortForecast", ""),
            detailed_forecast=data.get("detailedForecast", ""),
            precipitation_probability=data.get("probabilityOfPrecipitation", {}).get("value"),
        )


@dataclass
class Forecast:
    """Complete forecast for a location."""

    location_code: str
    latitude: float
    longitude: float
    updated_at: datetime
    periods: list[ForecastPeriod]

    def get_temperature_at(self, target_time: datetime) -> tuple[int, float] | None:
        """
        Get forecasted temperature at a specific time.

        Returns:
            Tuple of (temperature, confidence) or None if no forecast available
        """
        for period in self.periods:
            if period.start_time <= target_time < period.end_time:
                # Confidence decreases with forecast distance
                hours_out = (target_time - datetime.utcnow()).total_seconds() / 3600
                confidence = max(0.5, 1.0 - (hours_out / 168))  # 7-day decay
                return (period.temperature, confidence)
        return None

    def get_high_low_for_date(self, date: datetime) -> tuple[int | None, int | None]:
        """Get forecasted high and low temperatures for a specific date."""
        high = None
        low = None

        for period in self.periods:
            if period.start_time.date() == date.date():
                if period.is_daytime:
                    high = period.temperature
                else:
                    low = period.temperature

        return (high, low)

    def get_precipitation_probability(self, target_time: datetime) -> int | None:
        """Get precipitation probability at a specific time."""
        for period in self.periods:
            if period.start_time <= target_time < period.end_time:
                return period.precipitation_probability
        return None


class NWSClient:
    """
    Client for the National Weather Service API.

    The NWS API is free and requires no authentication.
    Rate limit is approximately 100 requests per minute.

    API docs: https://www.weather.gov/documentation/services-web-api
    """

    BASE_URL = "https://api.weather.gov"
    USER_AGENT = "KalshiWeatherBot/1.0 (contact@example.com)"

    def __init__(
        self,
        request_timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        """
        Initialize the NWS client.

        Args:
            request_timeout: Request timeout in seconds
            max_retries: Maximum number of retries per request
        """
        self._timeout = aiohttp.ClientTimeout(total=request_timeout)
        self._max_retries = max_retries
        self._session: aiohttp.ClientSession | None = None

        # Cache for grid points (they don't change)
        self._grid_cache: dict[str, tuple[str, int, int]] = {}

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            # Create SSL context with certifi certificates
            ssl_context = ssl.create_default_context(cafile=certifi.where())
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            self._session = aiohttp.ClientSession(
                timeout=self._timeout,
                headers={"User-Agent": self.USER_AGENT},
                connector=connector,
            )
        return self._session

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def _request(self, url: str) -> dict[str, Any] | None:
        """Make an HTTP request with retries."""
        session = await self._get_session()

        for attempt in range(self._max_retries):
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 503:
                        # Service unavailable, retry with backoff
                        await asyncio.sleep(2 ** attempt)
                    else:
                        logger.warning(f"NWS API error: {response.status} for {url}")
                        return None
            except asyncio.TimeoutError:
                logger.warning(f"NWS API timeout (attempt {attempt + 1})")
            except aiohttp.ClientError as e:
                logger.warning(f"NWS API client error: {e}")
                await asyncio.sleep(1)

        return None

    async def _get_grid_point(self, location_code: str) -> tuple[str, int, int] | None:
        """
        Get the NWS grid point for a location.

        Returns:
            Tuple of (office, gridX, gridY) or None
        """
        if location_code in self._grid_cache:
            return self._grid_cache[location_code]

        coords = CITY_COORDINATES.get(location_code)
        if not coords:
            logger.warning(f"Unknown location code: {location_code}")
            return None

        lat, lon = coords
        url = f"{self.BASE_URL}/points/{lat},{lon}"

        data = await self._request(url)
        if not data:
            return None

        try:
            props = data["properties"]
            grid_point = (
                props["gridId"],
                props["gridX"],
                props["gridY"],
            )
            self._grid_cache[location_code] = grid_point
            return grid_point
        except KeyError as e:
            logger.error(f"Invalid NWS grid response: {e}")
            return None

    async def get_forecast(self, location_code: str) -> Forecast | None:
        """
        Get the 7-day forecast for a location.

        Args:
            location_code: Location code (e.g., "NYC", "CHI")

        Returns:
            Forecast object or None if unavailable
        """
        grid_point = await self._get_grid_point(location_code)
        if not grid_point:
            return None

        office, grid_x, grid_y = grid_point
        url = f"{self.BASE_URL}/gridpoints/{office}/{grid_x},{grid_y}/forecast"

        data = await self._request(url)
        if not data:
            return None

        try:
            periods_data = data["properties"]["periods"]
            periods = [ForecastPeriod.from_api_response(p) for p in periods_data]

            coords = CITY_COORDINATES.get(location_code, (0, 0))

            return Forecast(
                location_code=location_code,
                latitude=coords[0],
                longitude=coords[1],
                updated_at=datetime.utcnow(),
                periods=periods,
            )
        except (KeyError, ValueError) as e:
            logger.error(f"Error parsing NWS forecast: {e}")
            return None

    async def get_forecasts_batch(
        self,
        location_codes: list[str],
    ) -> dict[str, Forecast]:
        """
        Get forecasts for multiple locations.

        Args:
            location_codes: List of location codes

        Returns:
            Dict mapping location code to Forecast
        """
        results = {}

        for location in location_codes:
            forecast = await self.get_forecast(location)
            if forecast:
                results[location] = forecast
            # Small delay to respect rate limits
            await asyncio.sleep(0.1)

        return results

    @staticmethod
    def get_supported_locations() -> list[str]:
        """Get list of supported location codes."""
        return list(CITY_COORDINATES.keys())
