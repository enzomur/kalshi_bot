"""Position sizing coordinator combining Kelly and constraints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from kalshi_bot.config.settings import Settings
from kalshi_bot.core.types import ArbitrageOpportunity
from kalshi_bot.optimization.bregman import BregmanProjection
from kalshi_bot.optimization.frank_wolfe import FrankWolfeOptimizer
from kalshi_bot.optimization.kelly import KellyCriterion
from kalshi_bot.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SizingResult:
    """Result of position sizing calculation."""

    opportunity_id: str
    recommended_quantity: int
    allocated_capital: float
    expected_profit: float
    expected_roi: float
    kelly_fraction: float
    constraints_applied: list[str]


class PositionSizer:
    """
    Coordinates position sizing across multiple optimization methods.

    Combines:
    - Kelly criterion for optimal sizing
    - Frank-Wolfe for portfolio optimization
    - Bregman projection for constraint satisfaction
    - Risk limits and position constraints

    The sizing process:
    1. Calculate Kelly optimal fraction for each opportunity
    2. Apply position limits (max size, max concentration)
    3. Apply portfolio constraints (total exposure)
    4. Project onto feasible region if necessary
    """

    def __init__(self, settings: Settings) -> None:
        """
        Initialize position sizer.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self._kelly = KellyCriterion(settings)
        self._bregman = BregmanProjection()
        self._frank_wolfe = FrankWolfeOptimizer()

        self._max_position_size = settings.trading.max_position_size
        self._max_position_pct = settings.trading.max_position_pct

    def size_single_opportunity(
        self,
        opportunity: ArbitrageOpportunity,
        available_balance: float,
        current_exposure: float = 0.0,
    ) -> SizingResult:
        """
        Calculate position size for a single opportunity.

        Args:
            opportunity: The arbitrage opportunity
            available_balance: Available trading balance
            current_exposure: Current total market exposure

        Returns:
            SizingResult with recommended quantity
        """
        constraints_applied: list[str] = []

        cost_per_contract = opportunity.total_cost / opportunity.max_quantity
        profit_per_contract = opportunity.net_profit / opportunity.max_quantity

        total_cost_cents = int(cost_per_contract * 100)
        kelly_contracts = self._kelly.calculate_arbitrage_size(
            total_cost_cents,
            100,
            opportunity.max_quantity,
            available_balance,
        )

        recommended = kelly_contracts
        constraints_applied.append("kelly")

        if recommended > self._max_position_size:
            recommended = self._max_position_size
            constraints_applied.append("max_position_size")

        max_by_balance = int(available_balance / cost_per_contract)
        if recommended > max_by_balance:
            recommended = max_by_balance
            constraints_applied.append("available_balance")

        if recommended > opportunity.max_quantity:
            recommended = opportunity.max_quantity
            constraints_applied.append("liquidity")

        max_by_concentration = int(
            available_balance * self._max_position_pct / cost_per_contract
        )
        if recommended > max_by_concentration:
            recommended = max_by_concentration
            constraints_applied.append("concentration_limit")

        recommended = max(0, recommended)

        allocated_capital = recommended * cost_per_contract
        expected_profit = recommended * profit_per_contract
        expected_roi = expected_profit / allocated_capital if allocated_capital > 0 else 0

        kelly_fraction = (recommended * cost_per_contract) / available_balance if available_balance > 0 else 0

        return SizingResult(
            opportunity_id=opportunity.opportunity_id,
            recommended_quantity=recommended,
            allocated_capital=allocated_capital,
            expected_profit=expected_profit,
            expected_roi=expected_roi,
            kelly_fraction=kelly_fraction,
            constraints_applied=constraints_applied,
        )

    def size_portfolio(
        self,
        opportunities: list[ArbitrageOpportunity],
        available_balance: float,
        max_opportunities: int = 5,
    ) -> list[SizingResult]:
        """
        Calculate optimal sizing across multiple opportunities.

        Uses portfolio optimization to allocate capital efficiently
        while respecting all constraints.

        Args:
            opportunities: List of opportunities to consider
            available_balance: Total available balance
            max_opportunities: Maximum number to allocate to

        Returns:
            List of SizingResults for each opportunity
        """
        if not opportunities:
            return []

        ranked = sorted(opportunities, key=lambda x: x.roi, reverse=True)
        selected = ranked[:max_opportunities]

        n = len(selected)
        if n == 0:
            return []

        expected_returns = np.array([opp.roi for opp in selected])

        max_per_opp = self._max_position_pct
        position_limits = np.array([max_per_opp] * n)

        min_allocation = 0.01
        lower_bounds = np.array([min_allocation] * n)
        upper_bounds = position_limits

        if np.sum(lower_bounds) > 1.0:
            lower_bounds = lower_bounds * (0.9 / np.sum(lower_bounds))

        result = self._frank_wolfe.optimize_box_simplex(
            lambda x: -np.dot(expected_returns, x),
            lambda x: -expected_returns,
            n,
            lower_bounds,
            upper_bounds,
        )

        allocations = result.solution

        results: list[SizingResult] = []
        remaining_balance = available_balance

        for i, opp in enumerate(selected):
            allocation_pct = allocations[i]
            allocated_capital = available_balance * allocation_pct

            cost_per_contract = opp.total_cost / opp.max_quantity if opp.max_quantity > 0 else 1
            profit_per_contract = opp.net_profit / opp.max_quantity if opp.max_quantity > 0 else 0

            contracts = int(allocated_capital / cost_per_contract)
            contracts = min(contracts, opp.max_quantity, self._max_position_size)
            contracts = max(0, contracts)

            actual_cost = contracts * cost_per_contract
            if actual_cost > remaining_balance:
                contracts = int(remaining_balance / cost_per_contract)
                actual_cost = contracts * cost_per_contract

            remaining_balance -= actual_cost

            results.append(SizingResult(
                opportunity_id=opp.opportunity_id,
                recommended_quantity=contracts,
                allocated_capital=actual_cost,
                expected_profit=contracts * profit_per_contract,
                expected_roi=profit_per_contract / cost_per_contract if cost_per_contract > 0 else 0,
                kelly_fraction=actual_cost / available_balance if available_balance > 0 else 0,
                constraints_applied=["portfolio_optimization", "frank_wolfe"],
            ))

        return results

    def rebalance_portfolio(
        self,
        current_positions: dict[str, int],
        opportunities: list[ArbitrageOpportunity],
        available_balance: float,
        total_portfolio_value: float,
    ) -> dict[str, int]:
        """
        Calculate rebalancing trades to optimize portfolio.

        Args:
            current_positions: Current position quantities by ticker
            opportunities: Available opportunities
            available_balance: Cash available for new positions
            total_portfolio_value: Total portfolio value

        Returns:
            Dict of recommended position changes (positive = buy, negative = sell)
        """
        changes: dict[str, int] = {}

        new_sizing = self.size_portfolio(opportunities, available_balance)

        sizing_map = {r.opportunity_id: r for r in new_sizing}

        for opp in opportunities:
            sizing = sizing_map.get(opp.opportunity_id)
            if sizing is None:
                continue

            current_qty = current_positions.get(opp.markets[0], 0)
            target_qty = sizing.recommended_quantity

            if target_qty != current_qty:
                changes[opp.markets[0]] = target_qty - current_qty

        return changes

    def get_status(self) -> dict[str, Any]:
        """Get position sizer status."""
        return {
            "max_position_size": self._max_position_size,
            "max_position_pct": self._max_position_pct,
            "kelly_fraction": self.settings.trading.kelly_fraction,
            "execution_confidence": self.settings.trading.execution_confidence,
            "min_edge": self.settings.trading.min_edge,
        }
