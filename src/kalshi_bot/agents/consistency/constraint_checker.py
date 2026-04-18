"""Constraint checker for finding pricing inconsistencies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from kalshi_bot.agents.consistency.relationship_db import (
    MarketRelationship,
    RelationshipType,
)

if TYPE_CHECKING:
    pass


@dataclass
class ConstraintViolation:
    """A detected constraint violation."""

    relationship: MarketRelationship
    price_a: int  # Price of market A in cents
    price_b: int  # Price of market B in cents
    violation_magnitude: float  # How much the constraint is violated
    trade_side_a: str  # 'yes' or 'no' for market A
    trade_side_b: str  # 'yes' or 'no' for market B
    guaranteed_profit: float  # Profit per contract if arbitrage
    explanation: str


class ConstraintChecker:
    """
    Checks for pricing constraint violations across related markets.

    Constraints:
    1. Temporal subset: P(by earlier date) <= P(by later date)
    2. Implies: P(A and B) <= min(P(A), P(B))
    3. Mutually exclusive: P(A) + P(B) <= 1
    4. Exhaustive: P(A) + P(B) + ... = 1
    """

    # Minimum violation magnitude to consider
    MIN_VIOLATION = 0.05  # 5 cents

    def __init__(
        self,
        min_violation_magnitude: float = 0.05,
        require_guaranteed_profit: bool = True,
    ) -> None:
        """
        Initialize constraint checker.

        Args:
            min_violation_magnitude: Minimum violation to flag
            require_guaranteed_profit: Only flag if arbitrage is possible
        """
        self._min_violation = min_violation_magnitude
        self._require_guaranteed = require_guaranteed_profit

    def check_constraint(
        self,
        relationship: MarketRelationship,
        price_a: int,
        price_b: int,
    ) -> ConstraintViolation | None:
        """
        Check if a constraint is violated.

        Args:
            relationship: The market relationship
            price_a: Price of market A in cents
            price_b: Price of market B in cents

        Returns:
            ConstraintViolation if violated, None otherwise
        """
        if relationship.relationship_type == RelationshipType.TEMPORAL_SUBSET:
            return self._check_temporal_subset(relationship, price_a, price_b)
        elif relationship.relationship_type == RelationshipType.IMPLIES:
            return self._check_implies(relationship, price_a, price_b)
        elif relationship.relationship_type == RelationshipType.MUTUALLY_EXCLUSIVE:
            return self._check_mutually_exclusive(relationship, price_a, price_b)
        elif relationship.relationship_type == RelationshipType.EXHAUSTIVE:
            return self._check_exhaustive(relationship, price_a, price_b)

        return None

    def _check_temporal_subset(
        self,
        rel: MarketRelationship,
        price_earlier: int,
        price_later: int,
    ) -> ConstraintViolation | None:
        """
        Check temporal subset constraint.

        P(by earlier date) <= P(by later date)

        If earlier date price > later date price, there's an arbitrage:
        - Buy YES on later date (cheaper)
        - Buy NO on earlier date (or sell YES if you have it)

        Example: If "GDP growth by Q1" is priced at 60c but "GDP growth by Q2"
        is priced at 50c, that's inconsistent because Q1 is a subset of Q2.
        """
        if price_earlier <= price_later:
            return None  # No violation

        violation_magnitude = (price_earlier - price_later) / 100.0

        if violation_magnitude < self._min_violation:
            return None

        # Arbitrage strategy:
        # Buy YES on later (cheaper, price_later)
        # Buy NO on earlier (100 - price_earlier)
        # If earlier settles YES -> later must settle YES -> win on later
        # If earlier settles NO -> win on earlier

        cost_later_yes = price_later
        cost_earlier_no = 100 - price_earlier

        # Total cost
        total_cost = cost_later_yes + cost_earlier_no

        # If total cost < 100, guaranteed profit
        if total_cost >= 100:
            if self._require_guaranteed:
                return None
            guaranteed_profit = 0.0
        else:
            guaranteed_profit = (100 - total_cost) / 100.0

        return ConstraintViolation(
            relationship=rel,
            price_a=price_earlier,
            price_b=price_later,
            violation_magnitude=violation_magnitude,
            trade_side_a="no",  # Buy NO on earlier
            trade_side_b="yes",  # Buy YES on later
            guaranteed_profit=guaranteed_profit,
            explanation=(
                f"Temporal inconsistency: {rel.market_a} (earlier) at {price_earlier}c > "
                f"{rel.market_b} (later) at {price_later}c. "
                f"Buy NO@{100-price_earlier}c + YES@{price_later}c = {total_cost}c cost, "
                f"guaranteed {100-total_cost}c profit."
            ),
        )

    def _check_implies(
        self,
        rel: MarketRelationship,
        price_a: int,
        price_b: int,
    ) -> ConstraintViolation | None:
        """
        Check implication constraint.

        If A implies B, then P(A) <= P(B)
        P(A and B) <= min(P(A), P(B))
        """
        # A implies B means P(A) <= P(B)
        if price_a <= price_b:
            return None

        violation_magnitude = (price_a - price_b) / 100.0

        if violation_magnitude < self._min_violation:
            return None

        # Similar arbitrage to temporal
        cost_b_yes = price_b
        cost_a_no = 100 - price_a
        total_cost = cost_b_yes + cost_a_no

        if total_cost >= 100:
            if self._require_guaranteed:
                return None
            guaranteed_profit = 0.0
        else:
            guaranteed_profit = (100 - total_cost) / 100.0

        return ConstraintViolation(
            relationship=rel,
            price_a=price_a,
            price_b=price_b,
            violation_magnitude=violation_magnitude,
            trade_side_a="no",
            trade_side_b="yes",
            guaranteed_profit=guaranteed_profit,
            explanation=(
                f"Implication inconsistency: {rel.market_a} at {price_a}c > "
                f"{rel.market_b} at {price_b}c (A implies B)."
            ),
        )

    def _check_mutually_exclusive(
        self,
        rel: MarketRelationship,
        price_a: int,
        price_b: int,
    ) -> ConstraintViolation | None:
        """
        Check mutual exclusivity constraint.

        P(A) + P(B) <= 1 for mutually exclusive events

        If sum > 100, arbitrage by selling both (buying NO on both).
        """
        total = price_a + price_b

        if total <= 100:
            return None

        violation_magnitude = (total - 100) / 100.0

        if violation_magnitude < self._min_violation:
            return None

        # Arbitrage: Buy NO on both
        cost_a_no = 100 - price_a
        cost_b_no = 100 - price_b
        total_cost = cost_a_no + cost_b_no

        # At least one must settle NO, so we get at least 100
        # Total cost < 100 means guaranteed profit
        guaranteed_profit = (100 - total_cost) / 100.0 if total_cost < 100 else 0.0

        if guaranteed_profit <= 0 and self._require_guaranteed:
            return None

        return ConstraintViolation(
            relationship=rel,
            price_a=price_a,
            price_b=price_b,
            violation_magnitude=violation_magnitude,
            trade_side_a="no",
            trade_side_b="no",
            guaranteed_profit=guaranteed_profit,
            explanation=(
                f"Mutual exclusivity violation: {rel.market_a}@{price_a}c + "
                f"{rel.market_b}@{price_b}c = {total}c > 100c. "
                f"Buy NO on both for guaranteed profit."
            ),
        )

    def _check_exhaustive(
        self,
        rel: MarketRelationship,
        price_a: int,
        price_b: int,
    ) -> ConstraintViolation | None:
        """
        Check exhaustive constraint.

        For exhaustive outcomes, sum should equal 100.
        If sum < 100, buy YES on all.
        If sum > 100, buy NO on all.

        Note: This simplified version only handles 2-market exhaustive sets.
        """
        total = price_a + price_b
        expected = rel.constraint_value or 100

        violation = abs(total - expected)
        violation_magnitude = violation / 100.0

        if violation_magnitude < self._min_violation:
            return None

        if total < expected:
            # Buy YES on both - at least one must settle YES
            guaranteed_profit = (expected - total) / 100.0
            trade_side = "yes"
        else:
            # Buy NO on both - at least one must settle NO
            cost_a_no = 100 - price_a
            cost_b_no = 100 - price_b
            guaranteed_profit = (100 - cost_a_no - cost_b_no) / 100.0
            trade_side = "no"

        if guaranteed_profit <= 0 and self._require_guaranteed:
            return None

        return ConstraintViolation(
            relationship=rel,
            price_a=price_a,
            price_b=price_b,
            violation_magnitude=violation_magnitude,
            trade_side_a=trade_side,
            trade_side_b=trade_side,
            guaranteed_profit=guaranteed_profit,
            explanation=(
                f"Exhaustive constraint violation: sum={total}c, expected={expected}c. "
                f"Buy {trade_side.upper()} on both."
            ),
        )
