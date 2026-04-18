"""Consistency Arbitrage Agent - finds mathematically inconsistent prices."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from kalshi_bot.agents.base import BaseAgent
from kalshi_bot.agents.consistency.constraint_checker import (
    ConstraintChecker,
    ConstraintViolation,
)
from kalshi_bot.agents.consistency.relationship_db import RelationshipDB
from kalshi_bot.core.types import ArbitrageOpportunity, ArbitrageType, Side
from kalshi_bot.utils.logging import get_logger

if TYPE_CHECKING:
    from kalshi_bot.api.client import KalshiAPIClient
    from kalshi_bot.persistence.database import Database

logger = get_logger(__name__)


@dataclass
class ConsistencyOpportunity:
    """A trading opportunity from pricing inconsistency."""

    violation: ConstraintViolation
    quantity: int
    expected_profit: float

    def to_arbitrage_opportunity(self) -> ArbitrageOpportunity:
        """Convert to ArbitrageOpportunity for execution."""
        opportunity_id = (
            f"consistency_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{str(uuid4())[:8]}"
        )

        rel = self.violation.relationship
        side_a = Side.YES if self.violation.trade_side_a == "yes" else Side.NO
        side_b = Side.YES if self.violation.trade_side_b == "yes" else Side.NO

        price_a = (
            self.violation.price_a
            if side_a == Side.YES
            else (100 - self.violation.price_a)
        )
        price_b = (
            self.violation.price_b
            if side_b == Side.YES
            else (100 - self.violation.price_b)
        )

        legs = [
            {
                "market_ticker": rel.market_a,
                "side": side_a.value,
                "price": price_a,
                "quantity": self.quantity,
                "action": "buy",
            },
            {
                "market_ticker": rel.market_b,
                "side": side_b.value,
                "price": price_b,
                "quantity": self.quantity,
                "action": "buy",
            },
        ]

        total_cost = (price_a + price_b) * self.quantity / 100
        fees = sum(
            0.07 * (p / 100) * (1 - p / 100) * self.quantity for p in [price_a, price_b]
        )

        return ArbitrageOpportunity(
            opportunity_id=opportunity_id,
            arbitrage_type=ArbitrageType.CROSS_MARKET,
            markets=[rel.market_a, rel.market_b],
            expected_profit=self.expected_profit * 100,
            expected_profit_pct=self.violation.guaranteed_profit,
            confidence=0.95,  # High confidence for pure arbitrage
            legs=legs,
            max_quantity=self.quantity,
            total_cost=total_cost,
            fees=fees,
            net_profit=self.expected_profit,
            metadata={
                "source": "consistency",
                "relationship_type": rel.relationship_type.value,
                "violation_magnitude": self.violation.violation_magnitude,
                "explanation": self.violation.explanation,
            },
        )


class ConsistencyAgent(BaseAgent):
    """
    Agent that finds mathematically inconsistent prices across related markets.

    Strategy (pure arbitrage):
    - P(A and B) must be <= min(P(A), P(B))
    - P(by earlier date) must be <= P(by later date)
    - Mutually exclusive events: P(A) + P(B) <= 1

    When constraints are violated, there's guaranteed profit.
    """

    # Minimum violation to trade
    MIN_VIOLATION = 0.05

    # Only trade if profit is guaranteed
    REQUIRE_GUARANTEED = True

    # Max percentage of capital per trade
    MAX_POSITION_PCT = 0.05  # Can be higher for true arbitrage

    def __init__(
        self,
        db: "Database",
        api_client: "KalshiAPIClient",
        min_violation_magnitude: float = 0.05,
        require_guaranteed_profit: bool = True,
        update_interval_minutes: int = 10,
        enabled: bool = True,
    ) -> None:
        """
        Initialize Consistency Agent.

        Args:
            db: Database connection
            api_client: Kalshi API client
            min_violation_magnitude: Minimum constraint violation to trade
            require_guaranteed_profit: Only trade guaranteed arbitrage
            update_interval_minutes: How often to check for violations
            enabled: Whether the agent is enabled
        """
        super().__init__(
            db=db,
            name="consistency",
            update_interval_seconds=update_interval_minutes * 60,
            enabled=enabled,
        )

        self._api_client = api_client
        self._min_violation = min_violation_magnitude
        self._require_guaranteed = require_guaranteed_profit

        self._relationship_db = RelationshipDB(db)
        self._checker = ConstraintChecker(
            min_violation_magnitude=min_violation_magnitude,
            require_guaranteed_profit=require_guaranteed_profit,
        )

        # Cache of current violations
        self._violations: list[ConstraintViolation] = []

    async def _run_cycle(self) -> None:
        """Execute one cycle of consistency checking."""
        logger.info("Consistency agent: checking for pricing violations")

        # Load relationships if first run
        if not self._relationship_db.all_relationships:
            await self._relationship_db.load_relationships()

        # Get current markets
        try:
            markets, _ = await self._api_client.get_markets(
                status="open",
                limit=200,
            )
        except Exception as e:
            logger.error(f"Failed to fetch markets: {e}")
            return

        # Discover new relationships
        new_rels = await self._relationship_db.discover_relationships(markets)
        if new_rels:
            logger.info(f"Discovered {len(new_rels)} new market relationships")

        # Build price map
        price_map = {}
        for market in markets:
            ticker = market.ticker
            price = market.last_price
            if price is None:
                if market.yes_bid is not None and market.yes_ask is not None:
                    price = (market.yes_bid + market.yes_ask) // 2
            if ticker and price is not None:
                price_map[ticker] = price

        # Check all relationships for violations
        violations = []
        for rel in self._relationship_db.all_relationships:
            if rel.market_a not in price_map or rel.market_b not in price_map:
                continue

            price_a = price_map[rel.market_a]
            price_b = price_map[rel.market_b]

            violation = self._checker.check_constraint(rel, price_a, price_b)
            if violation:
                violations.append(violation)
                await self._relationship_db.record_violation(
                    rel.market_a,
                    rel.market_b,
                    rel.relationship_type,
                )

        self._violations = violations

        # Update metrics
        self._status.metrics = {
            "relationships_checked": len(self._relationship_db.all_relationships),
            "violations_found": len(violations),
            "guaranteed_arbitrage": sum(
                1 for v in violations if v.guaranteed_profit > 0
            ),
        }

        if violations:
            logger.info(
                f"Consistency agent: found {len(violations)} constraint violations"
            )
        else:
            logger.debug("No constraint violations found")

    async def get_trading_opportunities(
        self,
        available_capital: float,
        max_opportunities: int = 5,
    ) -> list[ArbitrageOpportunity]:
        """
        Get consistency arbitrage opportunities for the bot.

        Args:
            available_capital: Available capital for trading
            max_opportunities: Maximum opportunities to return

        Returns:
            List of ArbitrageOpportunity objects
        """
        opportunities = []

        # Sort by guaranteed profit
        sorted_violations = sorted(
            self._violations,
            key=lambda v: v.guaranteed_profit,
            reverse=True,
        )

        for violation in sorted_violations[:max_opportunities]:
            opp = self._create_opportunity(violation, available_capital)
            if opp:
                opportunities.append(opp.to_arbitrage_opportunity())

        return opportunities

    def _create_opportunity(
        self,
        violation: ConstraintViolation,
        available_capital: float,
    ) -> ConsistencyOpportunity | None:
        """Create a trading opportunity from a violation."""
        if violation.guaranteed_profit <= 0 and self._require_guaranteed:
            return None

        # Calculate position size
        # For arbitrage, we can size more aggressively
        position_value = available_capital * self.MAX_POSITION_PCT

        # Cost per "arbitrage unit" (buying both legs)
        rel = violation.relationship
        price_a = (
            violation.price_a
            if violation.trade_side_a == "yes"
            else (100 - violation.price_a)
        )
        price_b = (
            violation.price_b
            if violation.trade_side_b == "yes"
            else (100 - violation.price_b)
        )

        cost_per_unit = (price_a + price_b) / 100.0
        quantity = int(position_value / cost_per_unit) if cost_per_unit > 0 else 0

        if quantity < 1:
            return None

        # Guaranteed profit per unit
        profit_per_unit = violation.guaranteed_profit
        expected_profit = profit_per_unit * quantity

        # Subtract fees
        fees = sum(
            0.07 * (p / 100) * (1 - p / 100) * quantity for p in [price_a, price_b]
        )
        expected_profit -= fees

        if expected_profit <= 0:
            return None

        logger.info(
            f"Consistency opportunity: {rel.market_a} + {rel.market_b}, "
            f"violation={violation.violation_magnitude:.1%}, "
            f"profit=${expected_profit:.2f}, qty={quantity}"
        )

        return ConsistencyOpportunity(
            violation=violation,
            quantity=quantity,
            expected_profit=expected_profit,
        )

    def get_status(self) -> dict[str, Any]:
        """Get agent status with consistency-specific metrics."""
        status = super().get_status()
        status["relationships"] = len(self._relationship_db.all_relationships)
        status["active_violations"] = len(self._violations)
        status["guaranteed_arbitrage"] = sum(
            1 for v in self._violations if v.guaranteed_profit > 0
        )
        return status
