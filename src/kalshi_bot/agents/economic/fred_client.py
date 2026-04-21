"""FRED API client for economic data."""

from __future__ import annotations

import os
import ssl
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import aiohttp
import certifi

from kalshi_bot.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EconomicDataPoint:
    """A single economic data point from FRED."""

    series_id: str
    date: datetime
    value: float
    release_date: datetime | None = None


@dataclass
class SeriesInfo:
    """Information about a FRED series."""

    series_id: str
    title: str
    frequency: str
    units: str
    seasonal_adjustment: str
    last_updated: datetime | None = None


class FREDClient:
    """
    Client for the Federal Reserve Economic Data (FRED) API.

    Used to fetch economic indicators for predicting market movements.

    Common series:
    - GDP: Gross Domestic Product
    - CPIAUCSL: Consumer Price Index (CPI)
    - PAYEMS: Nonfarm Payrolls
    - UNRATE: Unemployment Rate
    - FEDFUNDS: Federal Funds Rate
    """

    BASE_URL = "https://api.stlouisfed.org/fred"

    # Key economic series for Kalshi markets
    KEY_SERIES = {
        "GDP": "GDP",
        "GDPC1": "Real GDP",
        "CPIAUCSL": "CPI All Items",
        "CPILFESL": "Core CPI",
        "PAYEMS": "Nonfarm Payrolls",
        "UNRATE": "Unemployment Rate",
        "FEDFUNDS": "Fed Funds Rate",
        "T10Y2Y": "10Y-2Y Treasury Spread",
        "INDPRO": "Industrial Production",
        "RSXFS": "Retail Sales",
    }

    def __init__(
        self,
        api_key: str | None = None,
    ) -> None:
        """
        Initialize FRED client.

        Args:
            api_key: FRED API key (or set FRED_API_KEY env var)
        """
        self._api_key = api_key or os.getenv("FRED_API_KEY", "")
        self._session: aiohttp.ClientSession | None = None
        self._cache: dict[str, list[EconomicDataPoint]] = {}
        self._cache_expiry: dict[str, datetime] = {}

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            # Use certifi for proper SSL certificate handling on macOS
            ssl_context = ssl.create_default_context(cafile=certifi.where())
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            self._session = aiohttp.ClientSession(connector=connector)
        return self._session

    async def close(self) -> None:
        """Close the client session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def get_series(
        self,
        series_id: str,
        observation_start: datetime | None = None,
        observation_end: datetime | None = None,
        limit: int = 100,
    ) -> list[EconomicDataPoint]:
        """
        Get observations for a FRED series.

        Args:
            series_id: FRED series ID
            observation_start: Start date for observations
            observation_end: End date for observations
            limit: Maximum observations to return

        Returns:
            List of EconomicDataPoint objects
        """
        # Check cache
        cache_key = f"{series_id}_{limit}"
        if cache_key in self._cache:
            expiry = self._cache_expiry.get(cache_key)
            if expiry and datetime.utcnow() < expiry:
                return self._cache[cache_key]

        if not self._api_key:
            logger.warning("No FRED API key configured")
            return []

        params = {
            "series_id": series_id,
            "api_key": self._api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": limit,
        }

        if observation_start:
            params["observation_start"] = observation_start.strftime("%Y-%m-%d")
        if observation_end:
            params["observation_end"] = observation_end.strftime("%Y-%m-%d")

        try:
            session = await self._get_session()
            async with session.get(
                f"{self.BASE_URL}/series/observations",
                params=params,
            ) as response:
                if response.status != 200:
                    logger.error(f"FRED API error: {response.status}")
                    return []

                data = await response.json()

        except Exception as e:
            logger.error(f"FRED API request failed: {e}")
            return []

        observations = data.get("observations", [])
        data_points = []

        for obs in observations:
            try:
                value_str = obs.get("value", ".")
                if value_str == ".":
                    continue  # Missing data

                data_points.append(
                    EconomicDataPoint(
                        series_id=series_id,
                        date=datetime.strptime(obs["date"], "%Y-%m-%d"),
                        value=float(value_str),
                    )
                )
            except (ValueError, KeyError):
                continue

        # Cache for 1 hour
        self._cache[cache_key] = data_points
        self._cache_expiry[cache_key] = datetime.utcnow() + timedelta(hours=1)

        return data_points

    async def get_latest_value(self, series_id: str) -> EconomicDataPoint | None:
        """Get the most recent value for a series."""
        data_points = await self.get_series(series_id, limit=1)
        return data_points[0] if data_points else None

    async def get_series_info(self, series_id: str) -> SeriesInfo | None:
        """Get metadata about a FRED series."""
        if not self._api_key:
            return None

        params = {
            "series_id": series_id,
            "api_key": self._api_key,
            "file_type": "json",
        }

        try:
            session = await self._get_session()
            async with session.get(
                f"{self.BASE_URL}/series",
                params=params,
            ) as response:
                if response.status != 200:
                    return None

                data = await response.json()

        except Exception as e:
            logger.error(f"FRED API request failed: {e}")
            return None

        series_list = data.get("seriess", [])
        if not series_list:
            return None

        series = series_list[0]
        return SeriesInfo(
            series_id=series["id"],
            title=series.get("title", ""),
            frequency=series.get("frequency", ""),
            units=series.get("units", ""),
            seasonal_adjustment=series.get("seasonal_adjustment", ""),
        )

    async def get_release_dates(
        self,
        series_id: str,
        limit: int = 10,
    ) -> list[datetime]:
        """Get upcoming release dates for a series."""
        if not self._api_key:
            return []

        # First get the release ID for this series
        params = {
            "series_id": series_id,
            "api_key": self._api_key,
            "file_type": "json",
        }

        try:
            session = await self._get_session()
            async with session.get(
                f"{self.BASE_URL}/series/release",
                params=params,
            ) as response:
                if response.status != 200:
                    return []

                data = await response.json()

        except Exception as e:
            logger.error(f"FRED API request failed: {e}")
            return []

        releases = data.get("releases", [])
        if not releases:
            return []

        release_id = releases[0].get("id")
        if not release_id:
            return []

        # Get release dates
        params = {
            "release_id": release_id,
            "api_key": self._api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": limit,
        }

        try:
            async with session.get(
                f"{self.BASE_URL}/release/dates",
                params=params,
            ) as response:
                if response.status != 200:
                    return []

                data = await response.json()

        except Exception as e:
            logger.error(f"FRED API request failed: {e}")
            return []

        dates = []
        for rd in data.get("release_dates", []):
            try:
                dates.append(datetime.strptime(rd["date"], "%Y-%m-%d"))
            except (ValueError, KeyError):
                continue

        return dates

    def get_trend(
        self,
        data_points: list[EconomicDataPoint],
        periods: int = 3,
    ) -> str:
        """
        Calculate trend direction from recent data points.

        Args:
            data_points: List of data points (newest first)
            periods: Number of periods to consider

        Returns:
            'up', 'down', or 'flat'
        """
        if len(data_points) < 2:
            return "flat"

        recent = data_points[:periods]
        if len(recent) < 2:
            return "flat"

        # Calculate average change
        changes = []
        for i in range(len(recent) - 1):
            if recent[i + 1].value != 0:
                pct_change = (recent[i].value - recent[i + 1].value) / abs(
                    recent[i + 1].value
                )
                changes.append(pct_change)

        if not changes:
            return "flat"

        avg_change = sum(changes) / len(changes)

        if avg_change > 0.01:  # More than 1% average increase
            return "up"
        elif avg_change < -0.01:
            return "down"
        else:
            return "flat"
