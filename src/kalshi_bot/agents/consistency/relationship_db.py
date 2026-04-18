"""Database for market relationships used in consistency checking."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kalshi_bot.persistence.database import Database


class RelationshipType(str, Enum):
    """Types of market relationships."""

    # P(A and B) <= min(P(A), P(B))
    IMPLIES = "implies"

    # P(by date X) <= P(by date Y) where X < Y
    TEMPORAL_SUBSET = "temporal_subset"

    # P(A) + P(B) <= 1 for mutually exclusive events
    MUTUALLY_EXCLUSIVE = "mutually_exclusive"

    # P(A) + P(B) + ... = 1 for exhaustive outcomes
    EXHAUSTIVE = "exhaustive"


@dataclass
class MarketRelationship:
    """A relationship between two markets."""

    market_a: str
    market_b: str
    relationship_type: RelationshipType
    constraint_value: float | None = None  # For exhaustive: expected sum
    last_violation_at: datetime | None = None
    violation_count: int = 0


class RelationshipDB:
    """
    Manages market relationships for consistency checking.

    Relationships are discovered from:
    1. Event structure (markets in same event)
    2. Ticker patterns (e.g., GDP-Q1 implies GDP-H1)
    3. Manual configuration
    """

    def __init__(self, db: "Database") -> None:
        """
        Initialize relationship database.

        Args:
            db: Database connection
        """
        self._db = db
        self._relationships: list[MarketRelationship] = []

    async def load_relationships(self) -> None:
        """Load relationships from database."""
        rows = await self._db.fetch_all(
            """
            SELECT market_a, market_b, relationship_type, constraint_value,
                   last_violation_at, violation_count
            FROM market_relationships
            """
        )

        self._relationships = []
        for row in rows:
            try:
                rel_type = RelationshipType(row["relationship_type"])
            except ValueError:
                continue

            self._relationships.append(
                MarketRelationship(
                    market_a=row["market_a"],
                    market_b=row["market_b"],
                    relationship_type=rel_type,
                    constraint_value=row["constraint_value"],
                    violation_count=row["violation_count"] or 0,
                )
            )

    async def discover_relationships(
        self,
        markets: list[dict],
    ) -> list[MarketRelationship]:
        """
        Discover relationships from current markets.

        Args:
            markets: List of market dicts from API

        Returns:
            List of newly discovered relationships
        """
        new_relationships = []

        # Group markets by event
        events: dict[str, list] = {}
        for market in markets:
            event = market.event_ticker if hasattr(market, "event_ticker") else market.get("event_ticker", "")
            if event:
                events.setdefault(event, []).append(market)

        # Find temporal subset relationships (by date markets)
        new_relationships.extend(self._find_temporal_relationships(markets))

        # Find exhaustive relationships (same event, different outcomes)
        for event_ticker, event_markets in events.items():
            new_relationships.extend(
                self._find_exhaustive_relationships(event_ticker, event_markets)
            )

        # Store new relationships
        for rel in new_relationships:
            await self._store_relationship(rel)
            self._relationships.append(rel)

        return new_relationships

    def _find_temporal_relationships(
        self,
        markets: list,
    ) -> list[MarketRelationship]:
        """Find temporal subset relationships (earlier date implies later date)."""
        relationships = []

        # Group by base ticker pattern
        date_markets: dict[str, list[tuple[str, str]]] = {}

        for market in markets:
            ticker = market.ticker if hasattr(market, "ticker") else market.get("ticker", "")

            # Look for date patterns in tickers
            # e.g., GDP-25Q1, GDP-25Q2, GDP-25H1
            parts = ticker.split("-")
            if len(parts) >= 2:
                base = parts[0]
                date_part = parts[-1]

                # Quarter patterns
                if len(date_part) >= 3 and date_part[-2] == "Q":
                    date_markets.setdefault(base, []).append((ticker, date_part))
                # Half-year patterns
                elif len(date_part) >= 3 and date_part[-2] == "H":
                    date_markets.setdefault(base, []).append((ticker, date_part))

        # Create relationships
        for base, tickers in date_markets.items():
            # Sort by date
            sorted_tickers = sorted(tickers, key=lambda x: x[1])

            for i in range(len(sorted_tickers)):
                for j in range(i + 1, len(sorted_tickers)):
                    earlier = sorted_tickers[i][0]
                    later = sorted_tickers[j][0]

                    # Earlier date implies later date
                    # P(by earlier) <= P(by later)
                    rel = MarketRelationship(
                        market_a=earlier,
                        market_b=later,
                        relationship_type=RelationshipType.TEMPORAL_SUBSET,
                    )

                    if not self._relationship_exists(rel):
                        relationships.append(rel)

        return relationships

    def _find_exhaustive_relationships(
        self,
        event_ticker: str,
        markets: list,
    ) -> list[MarketRelationship]:
        """Find exhaustive relationships (probabilities must sum to 1)."""
        relationships = []

        # Check if markets look like exhaustive outcomes
        # e.g., "Will X be above 50?", "Will X be above 60?", etc.
        if len(markets) < 2:
            return relationships

        # For now, only detect obvious exhaustive sets
        # (This could be enhanced with more sophisticated pattern matching)

        return relationships

    def _relationship_exists(self, rel: MarketRelationship) -> bool:
        """Check if a relationship already exists."""
        for existing in self._relationships:
            if (
                existing.market_a == rel.market_a
                and existing.market_b == rel.market_b
                and existing.relationship_type == rel.relationship_type
            ):
                return True
        return False

    async def _store_relationship(self, rel: MarketRelationship) -> None:
        """Store a relationship in the database."""
        await self._db.execute(
            """
            INSERT OR IGNORE INTO market_relationships
            (market_a, market_b, relationship_type, constraint_value)
            VALUES (?, ?, ?, ?)
            """,
            (
                rel.market_a,
                rel.market_b,
                rel.relationship_type.value,
                rel.constraint_value,
            ),
        )

    async def record_violation(
        self,
        market_a: str,
        market_b: str,
        relationship_type: RelationshipType,
    ) -> None:
        """Record a constraint violation."""
        await self._db.execute(
            """
            UPDATE market_relationships
            SET violation_count = violation_count + 1,
                last_violation_at = datetime('now')
            WHERE market_a = ? AND market_b = ? AND relationship_type = ?
            """,
            (market_a, market_b, relationship_type.value),
        )

    def get_relationships_for_market(
        self,
        ticker: str,
    ) -> list[MarketRelationship]:
        """Get all relationships involving a market."""
        return [
            rel
            for rel in self._relationships
            if rel.market_a == ticker or rel.market_b == ticker
        ]

    @property
    def all_relationships(self) -> list[MarketRelationship]:
        """Get all relationships."""
        return self._relationships.copy()
