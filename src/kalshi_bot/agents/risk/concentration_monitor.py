"""Monitors concentration of weather market positions."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from kalshi_bot.agents.weather.market_mapper import WeatherMarketMapper
from kalshi_bot.utils.logging import get_logger

if TYPE_CHECKING:
    from kalshi_bot.persistence.database import Database

logger = get_logger(__name__)


@dataclass
class ConcentrationSnapshot:
    """Snapshot of weather exposure concentration."""

    timestamp: datetime
    total_weather_exposure: float
    total_portfolio_value: float
    weather_pct: float
    by_location: dict[str, float] = field(default_factory=dict)
    by_type: dict[str, float] = field(default_factory=dict)
    max_location_pct: float = 0.0
    max_location: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "total_weather_exposure": self.total_weather_exposure,
            "total_portfolio_value": self.total_portfolio_value,
            "weather_pct": self.weather_pct,
            "by_location": self.by_location,
            "by_type": self.by_type,
            "max_location_pct": self.max_location_pct,
            "max_location": self.max_location,
        }


@dataclass
class ConcentrationLimits:
    """Concentration limits for weather positions."""

    max_weather_exposure_pct: float = 0.30  # Max 30% portfolio in weather
    max_single_location_pct: float = 0.15   # Max 15% in one city
    max_single_type_pct: float = 0.20       # Max 20% in one weather type
    max_correlated_exposure_pct: float = 0.25  # Max 25% in correlated locations


class ConcentrationMonitor:
    """
    Monitors concentration of weather market positions.

    Tracks:
    - Total weather market exposure as % of portfolio
    - Exposure per location (city)
    - Exposure per weather type (temp, rain, snow)
    - Exposure in correlated locations
    """

    def __init__(
        self,
        db: "Database",
        limits: ConcentrationLimits | None = None,
    ) -> None:
        """
        Initialize the concentration monitor.

        Args:
            db: Database connection
            limits: Concentration limits
        """
        self._db = db
        self._limits = limits or ConcentrationLimits()
        self._market_mapper = WeatherMarketMapper()
        self._current_snapshot: ConcentrationSnapshot | None = None

    def compute_concentration(
        self,
        positions: list[dict],
        portfolio_value: float,
    ) -> ConcentrationSnapshot:
        """
        Compute current concentration of weather positions.

        Args:
            positions: List of position dicts with 'ticker' and 'value' keys
            portfolio_value: Total portfolio value

        Returns:
            ConcentrationSnapshot
        """
        by_location: dict[str, float] = {}
        by_type: dict[str, float] = {}
        total_weather = 0.0

        for position in positions:
            ticker = position.get("ticker", "")
            value = abs(position.get("value", 0))

            mapping = self._market_mapper.parse_ticker(ticker)
            if not mapping:
                continue

            total_weather += value
            location = mapping.location_code
            weather_type = mapping.weather_type.value

            by_location[location] = by_location.get(location, 0) + value
            by_type[weather_type] = by_type.get(weather_type, 0) + value

        # Calculate percentages
        weather_pct = total_weather / portfolio_value if portfolio_value > 0 else 0

        # Find max location
        max_location = ""
        max_location_value = 0.0
        for loc, val in by_location.items():
            if val > max_location_value:
                max_location = loc
                max_location_value = val

        max_location_pct = max_location_value / portfolio_value if portfolio_value > 0 else 0

        snapshot = ConcentrationSnapshot(
            timestamp=datetime.utcnow(),
            total_weather_exposure=total_weather,
            total_portfolio_value=portfolio_value,
            weather_pct=weather_pct,
            by_location=by_location,
            by_type=by_type,
            max_location_pct=max_location_pct,
            max_location=max_location,
        )

        self._current_snapshot = snapshot
        return snapshot

    def check_limits(
        self,
        snapshot: ConcentrationSnapshot,
    ) -> list[str]:
        """
        Check if concentration limits are violated.

        Args:
            snapshot: Current concentration snapshot

        Returns:
            List of violation messages (empty if all OK)
        """
        violations = []

        if snapshot.weather_pct > self._limits.max_weather_exposure_pct:
            violations.append(
                f"Weather exposure {snapshot.weather_pct:.1%} exceeds "
                f"limit {self._limits.max_weather_exposure_pct:.1%}"
            )

        if snapshot.max_location_pct > self._limits.max_single_location_pct:
            violations.append(
                f"Location {snapshot.max_location} at {snapshot.max_location_pct:.1%} "
                f"exceeds limit {self._limits.max_single_location_pct:.1%}"
            )

        for weather_type, value in snapshot.by_type.items():
            type_pct = value / snapshot.total_portfolio_value if snapshot.total_portfolio_value > 0 else 0
            if type_pct > self._limits.max_single_type_pct:
                violations.append(
                    f"Weather type {weather_type} at {type_pct:.1%} "
                    f"exceeds limit {self._limits.max_single_type_pct:.1%}"
                )

        return violations

    def get_available_capacity(
        self,
        location: str,
        weather_type: str,
        portfolio_value: float,
    ) -> float:
        """
        Get available capacity for a new position.

        Args:
            location: Location code
            weather_type: Weather type
            portfolio_value: Total portfolio value

        Returns:
            Maximum additional exposure allowed ($)
        """
        if not self._current_snapshot:
            # No existing positions, full capacity available
            return min(
                portfolio_value * self._limits.max_weather_exposure_pct,
                portfolio_value * self._limits.max_single_location_pct,
            )

        snapshot = self._current_snapshot

        # Calculate remaining capacity for each limit
        weather_remaining = (
            self._limits.max_weather_exposure_pct * portfolio_value
            - snapshot.total_weather_exposure
        )

        location_current = snapshot.by_location.get(location, 0)
        location_remaining = (
            self._limits.max_single_location_pct * portfolio_value
            - location_current
        )

        type_current = snapshot.by_type.get(weather_type, 0)
        type_remaining = (
            self._limits.max_single_type_pct * portfolio_value
            - type_current
        )

        return max(0, min(weather_remaining, location_remaining, type_remaining))

    async def save_snapshot(self, snapshot: ConcentrationSnapshot) -> None:
        """Save concentration snapshot to database."""
        import json

        await self._db.execute(
            """
            INSERT INTO weather_exposure_snapshots
            (snapshot_at, total_weather_exposure, exposure_by_location,
             exposure_by_type, portfolio_pct)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                snapshot.timestamp.isoformat(),
                snapshot.total_weather_exposure,
                json.dumps(snapshot.by_location),
                json.dumps(snapshot.by_type),
                snapshot.weather_pct,
            ),
        )

    def get_current_snapshot(self) -> ConcentrationSnapshot | None:
        """Get most recent concentration snapshot."""
        return self._current_snapshot

    def get_limits(self) -> ConcentrationLimits:
        """Get current concentration limits."""
        return self._limits

    def set_limits(self, limits: ConcentrationLimits) -> None:
        """Update concentration limits."""
        self._limits = limits
