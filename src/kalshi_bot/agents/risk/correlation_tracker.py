"""Tracks correlations between weather events at different locations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

from kalshi_bot.utils.logging import get_logger

if TYPE_CHECKING:
    from kalshi_bot.persistence.database import Database

logger = get_logger(__name__)


@dataclass
class LocationPairCorrelation:
    """Correlation data between two locations."""

    location_a: str
    location_b: str
    correlation: float
    sample_size: int
    last_updated: datetime

    @property
    def pair_key(self) -> str:
        """Get canonical key for this pair (sorted alphabetically)."""
        return ":".join(sorted([self.location_a, self.location_b]))


# Pre-computed weather correlations based on geographic proximity
# and historical weather pattern similarity
WEATHER_CORRELATIONS = {
    # Northeast corridor (high correlation)
    ("BOS", "NYC"): 0.75,
    ("NYC", "PHL"): 0.80,
    ("PHL", "DCA"): 0.75,
    ("BOS", "PHL"): 0.65,
    ("NYC", "DCA"): 0.70,
    ("BOS", "DCA"): 0.55,

    # Southeast
    ("ATL", "MIA"): 0.40,

    # West Coast (moderate correlation)
    ("LAX", "SFO"): 0.40,
    ("SEA", "SFO"): 0.35,

    # Midwest
    ("CHI", "MSP"): 0.60,
    ("CHI", "DEN"): 0.35,

    # Texas
    ("DFW", "HOU"): 0.50,

    # Cross-region (low correlation)
    ("NYC", "LAX"): 0.15,
    ("NYC", "CHI"): 0.35,
    ("LAX", "MIA"): 0.20,
    ("CHI", "ATL"): 0.25,
    ("SEA", "NYC"): 0.10,
    ("DEN", "ATL"): 0.20,
    ("PHX", "MIA"): 0.25,  # Both hot, but different climates
}


class CorrelationTracker:
    """
    Tracks and computes correlations between weather events.

    Used to:
    - Identify correlated positions that amplify risk
    - Adjust position limits based on correlation
    - Detect portfolio concentration in correlated markets
    """

    def __init__(
        self,
        db: "Database",
    ) -> None:
        """
        Initialize the correlation tracker.

        Args:
            db: Database connection
        """
        self._db = db
        self._correlations: dict[str, float] = {}
        self._load_builtin_correlations()

    def _load_builtin_correlations(self) -> None:
        """Load pre-computed correlations."""
        for (loc_a, loc_b), corr in WEATHER_CORRELATIONS.items():
            key = ":".join(sorted([loc_a, loc_b]))
            self._correlations[key] = corr

    def get_correlation(
        self,
        location_a: str,
        location_b: str,
    ) -> float:
        """
        Get correlation between two locations.

        Args:
            location_a: First location code
            location_b: Second location code

        Returns:
            Correlation coefficient (-1 to 1), defaults to 0.1 if unknown
        """
        if location_a == location_b:
            return 1.0

        key = ":".join(sorted([location_a, location_b]))
        return self._correlations.get(key, 0.1)  # Low default correlation

    def compute_portfolio_correlation(
        self,
        positions: dict[str, float],
    ) -> float:
        """
        Compute average pairwise correlation of portfolio positions.

        Args:
            positions: Dict mapping location_code to position value

        Returns:
            Average correlation (0-1)
        """
        if len(positions) < 2:
            return 0.0

        locations = list(positions.keys())
        total_corr = 0.0
        pair_count = 0

        for i, loc_a in enumerate(locations):
            for loc_b in locations[i + 1:]:
                weight_a = abs(positions[loc_a])
                weight_b = abs(positions[loc_b])
                total_weight = sum(abs(v) for v in positions.values())

                if total_weight == 0:
                    continue

                # Weight correlation by position sizes
                pair_weight = (weight_a + weight_b) / total_weight
                corr = self.get_correlation(loc_a, loc_b)
                total_corr += corr * pair_weight
                pair_count += 1

        if pair_count == 0:
            return 0.0

        return total_corr / pair_count

    def get_correlated_locations(
        self,
        location: str,
        min_correlation: float = 0.50,
    ) -> list[tuple[str, float]]:
        """
        Get locations correlated with the given location.

        Args:
            location: Location code
            min_correlation: Minimum correlation threshold

        Returns:
            List of (location, correlation) tuples
        """
        correlated = []

        for key, corr in self._correlations.items():
            if corr < min_correlation:
                continue

            loc_a, loc_b = key.split(":")
            if loc_a == location:
                correlated.append((loc_b, corr))
            elif loc_b == location:
                correlated.append((loc_a, corr))

        return sorted(correlated, key=lambda x: x[1], reverse=True)

    async def update_correlation(
        self,
        location_a: str,
        location_b: str,
        correlation: float,
        sample_size: int,
    ) -> None:
        """
        Update correlation for a location pair.

        Args:
            location_a: First location
            location_b: Second location
            correlation: New correlation value
            sample_size: Number of samples used
        """
        key = ":".join(sorted([location_a, location_b]))
        self._correlations[key] = correlation

        # Persist to database
        await self._db.execute(
            """
            INSERT OR REPLACE INTO weather_correlations
            (location_pair, correlation_coefficient, sample_size, last_updated)
            VALUES (?, ?, ?, ?)
            """,
            (key, correlation, sample_size, datetime.utcnow().isoformat()),
        )

    async def load_from_database(self) -> None:
        """Load correlations from database."""
        rows = await self._db.fetch_all(
            "SELECT location_pair, correlation_coefficient FROM weather_correlations"
        )

        for row in rows:
            self._correlations[row["location_pair"]] = row["correlation_coefficient"]

        logger.info(f"Loaded {len(rows)} correlations from database")

    def get_all_correlations(self) -> dict[str, float]:
        """Get all correlation values."""
        return self._correlations.copy()
