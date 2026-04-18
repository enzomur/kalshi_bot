"""Fair value estimation for new markets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kalshi_bot.persistence.database import Database


@dataclass
class FairValueEstimate:
    """Fair value estimate for a market."""

    ticker: str
    category: str
    fair_value: float  # Probability 0-1
    confidence: float
    sample_count: int
    explanation: str


class FairValueEstimator:
    """
    Estimates fair value for new markets based on historical data.

    Uses historical settlement data to estimate what price a market
    should trade at based on its category and characteristics.
    """

    # Default priors by category (base rate of YES settlement)
    CATEGORY_PRIORS = {
        "weather": 0.50,  # Weather is ~50/50 on average
        "economic_gdp": 0.50,
        "economic_cpi": 0.50,
        "economic_jobs": 0.50,
        "politics": 0.50,
        "sports": 0.50,
        "other": 0.50,
    }

    # Minimum samples needed to trust historical data
    MIN_SAMPLES = 10

    def __init__(
        self,
        db: "Database",
        min_samples: int = 10,
    ) -> None:
        """
        Initialize fair value estimator.

        Args:
            db: Database connection
            min_samples: Minimum samples for historical estimate
        """
        self._db = db
        self._min_samples = min_samples

        # Cache of historical fair values by category
        self._category_rates: dict[str, tuple[float, int]] = {}

    async def load_historical_data(self) -> None:
        """Load historical settlement rates by category."""
        # Load from historical_fair_values table
        rows = await self._db.fetch_all(
            """
            SELECT category, yes_rate, sample_count
            FROM historical_fair_values
            WHERE sample_count >= ?
            """,
            (self._min_samples,),
        )

        for row in rows:
            category = row["category"]
            self._category_rates[category] = (
                row["yes_rate"],
                row["sample_count"],
            )

    async def estimate_fair_value(
        self,
        ticker: str,
        category: str,
        current_price: int | None = None,
    ) -> FairValueEstimate:
        """
        Estimate fair value for a market.

        Args:
            ticker: Market ticker
            category: Market category
            current_price: Current market price (optional)

        Returns:
            FairValueEstimate with fair value and confidence
        """
        # Check for historical data
        if category in self._category_rates:
            rate, count = self._category_rates[category]
            confidence = min(count / 100.0, 0.8)  # Cap at 80% confidence

            return FairValueEstimate(
                ticker=ticker,
                category=category,
                fair_value=rate,
                confidence=confidence,
                sample_count=count,
                explanation=f"Historical {category} settlement rate: {rate:.0%} (n={count})",
            )

        # Fall back to category prior
        prior = self.CATEGORY_PRIORS.get(category, 0.50)

        return FairValueEstimate(
            ticker=ticker,
            category=category,
            fair_value=prior,
            confidence=0.3,  # Low confidence for prior
            sample_count=0,
            explanation=f"Using category prior: {prior:.0%}",
        )

    def calculate_edge(
        self,
        fair_value: float,
        current_price: int,
    ) -> tuple[float, str]:
        """
        Calculate edge from fair value vs current price.

        Args:
            fair_value: Estimated fair value (0-1)
            current_price: Current market price in cents

        Returns:
            Tuple of (edge, side) where edge is signed and side is 'yes' or 'no'
        """
        market_prob = current_price / 100.0
        edge = fair_value - market_prob

        if edge > 0:
            return edge, "yes"  # Market underprices YES
        else:
            return abs(edge), "no"  # Market underprices NO

    async def update_settlement(
        self,
        ticker: str,
        category: str,
        settled_yes: bool,
    ) -> None:
        """
        Update historical data with a settlement.

        Args:
            ticker: Market ticker
            category: Market category
            settled_yes: Whether the market settled YES
        """
        # Update historical_fair_values table
        existing = await self._db.fetch_one(
            """
            SELECT yes_rate, sample_count FROM historical_fair_values
            WHERE category = ? AND pattern = 'all'
            """,
            (category,),
        )

        if existing:
            old_rate = existing["yes_rate"]
            old_count = existing["sample_count"]
            new_count = old_count + 1
            new_rate = (old_rate * old_count + (1 if settled_yes else 0)) / new_count

            await self._db.execute(
                """
                UPDATE historical_fair_values
                SET yes_rate = ?, sample_count = ?, updated_at = datetime('now')
                WHERE category = ? AND pattern = 'all'
                """,
                (new_rate, new_count, category),
            )
        else:
            await self._db.execute(
                """
                INSERT INTO historical_fair_values
                (category, pattern, yes_rate, sample_count)
                VALUES (?, 'all', ?, 1)
                """,
                (category, 1.0 if settled_yes else 0.0),
            )

        # Update cache
        if category in self._category_rates:
            old_rate, old_count = self._category_rates[category]
            new_count = old_count + 1
            new_rate = (old_rate * old_count + (1 if settled_yes else 0)) / new_count
            self._category_rates[category] = (new_rate, new_count)
