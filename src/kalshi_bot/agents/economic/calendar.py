"""Economic release calendar for tracking upcoming data releases."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kalshi_bot.persistence.database import Database


@dataclass
class ScheduledRelease:
    """An upcoming economic data release."""

    release_name: str
    series_id: str
    release_time: datetime
    days_until: float
    expected_value: float | None = None
    prediction_direction: str | None = None
    prediction_confidence: float | None = None


class EconomicCalendar:
    """
    Tracks upcoming economic data releases.

    Used to identify when economic markets will settle and
    prepare trading positions in advance.
    """

    # Known economic release schedules (approximate)
    # Format: (series_id, name, day_of_month, hour_utc)
    RELEASE_SCHEDULE = {
        "GDP": ("GDP", "GDP", [28, 29, 30], 13),  # End of month
        "CPI": ("CPIAUCSL", "CPI", [10, 11, 12, 13], 13),  # Mid-month
        "JOBS": ("PAYEMS", "Nonfarm Payrolls", [1, 2, 3, 4, 5, 6, 7], 13),  # First Friday
        "UNRATE": ("UNRATE", "Unemployment Rate", [1, 2, 3, 4, 5, 6, 7], 13),
    }

    def __init__(self, db: "Database") -> None:
        """
        Initialize economic calendar.

        Args:
            db: Database connection
        """
        self._db = db
        self._upcoming_releases: list[ScheduledRelease] = []

    async def load_releases(self) -> None:
        """Load upcoming releases from database."""
        now = datetime.utcnow()

        rows = await self._db.fetch_all(
            """
            SELECT release_name, series_id, release_time,
                   expected_value, prediction_direction, prediction_confidence
            FROM economic_releases
            WHERE release_time > ?
            ORDER BY release_time ASC
            LIMIT 20
            """,
            (now.isoformat(),),
        )

        self._upcoming_releases = []
        for row in rows:
            try:
                release_time = datetime.fromisoformat(row["release_time"])
                days_until = (release_time - now).total_seconds() / 86400
            except (ValueError, TypeError):
                continue

            self._upcoming_releases.append(
                ScheduledRelease(
                    release_name=row["release_name"],
                    series_id=row["series_id"],
                    release_time=release_time,
                    days_until=days_until,
                    expected_value=row["expected_value"],
                    prediction_direction=row["prediction_direction"],
                    prediction_confidence=row["prediction_confidence"],
                )
            )

    async def add_release(
        self,
        release_name: str,
        series_id: str,
        release_time: datetime,
        expected_value: float | None = None,
    ) -> None:
        """Add an upcoming release to the calendar."""
        await self._db.execute(
            """
            INSERT OR REPLACE INTO economic_releases
            (release_name, series_id, release_time, expected_value)
            VALUES (?, ?, ?, ?)
            """,
            (release_name, series_id, release_time.isoformat(), expected_value),
        )

    async def update_prediction(
        self,
        series_id: str,
        release_time: datetime,
        direction: str,
        confidence: float,
    ) -> None:
        """Update prediction for a release."""
        await self._db.execute(
            """
            UPDATE economic_releases
            SET prediction_direction = ?, prediction_confidence = ?
            WHERE series_id = ? AND release_time = ?
            """,
            (direction, confidence, series_id, release_time.isoformat()),
        )

    async def record_actual(
        self,
        series_id: str,
        release_time: datetime,
        actual_value: float,
    ) -> None:
        """Record the actual released value."""
        # Get prediction
        row = await self._db.fetch_one(
            """
            SELECT prediction_direction, expected_value
            FROM economic_releases
            WHERE series_id = ? AND release_time = ?
            """,
            (series_id, release_time.isoformat()),
        )

        was_correct = None
        if row and row["prediction_direction"] and row["expected_value"]:
            expected = row["expected_value"]
            direction = row["prediction_direction"]

            if direction == "above" and actual_value > expected:
                was_correct = 1
            elif direction == "below" and actual_value < expected:
                was_correct = 1
            else:
                was_correct = 0

        await self._db.execute(
            """
            UPDATE economic_releases
            SET actual_value = ?, was_correct = ?
            WHERE series_id = ? AND release_time = ?
            """,
            (actual_value, was_correct, series_id, release_time.isoformat()),
        )

    def get_upcoming(
        self,
        max_days: float = 7.0,
    ) -> list[ScheduledRelease]:
        """Get releases within the specified number of days."""
        return [r for r in self._upcoming_releases if r.days_until <= max_days]

    def get_release_for_series(
        self,
        series_id: str,
    ) -> ScheduledRelease | None:
        """Get the next release for a specific series."""
        for release in self._upcoming_releases:
            if release.series_id == series_id:
                return release
        return None

    async def estimate_next_releases(self) -> list[ScheduledRelease]:
        """
        Estimate upcoming release dates based on historical patterns.

        This is a fallback when exact dates aren't available.
        """
        now = datetime.utcnow()
        estimates = []

        for key, (series_id, name, days, hour) in self.RELEASE_SCHEDULE.items():
            # Find next occurrence
            current_month = now.replace(day=1, hour=hour, minute=30, second=0)

            for day in sorted(days):
                try:
                    candidate = current_month.replace(day=day)
                    if candidate > now:
                        estimates.append(
                            ScheduledRelease(
                                release_name=name,
                                series_id=series_id,
                                release_time=candidate,
                                days_until=(candidate - now).total_seconds() / 86400,
                            )
                        )
                        break
                except ValueError:
                    continue
            else:
                # Try next month
                next_month = (current_month.replace(day=28) + timedelta(days=5)).replace(
                    day=1
                )
                for day in sorted(days):
                    try:
                        candidate = next_month.replace(day=day, hour=hour)
                        estimates.append(
                            ScheduledRelease(
                                release_name=name,
                                series_id=series_id,
                                release_time=candidate,
                                days_until=(candidate - now).total_seconds() / 86400,
                            )
                        )
                        break
                    except ValueError:
                        continue

        return sorted(estimates, key=lambda r: r.release_time)

    @property
    def upcoming_releases(self) -> list[ScheduledRelease]:
        """Get all tracked upcoming releases."""
        return self._upcoming_releases.copy()
