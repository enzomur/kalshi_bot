"""Kelly criterion for optimal bet sizing."""

from __future__ import annotations

from dataclasses import dataclass

from kalshi_bot.config.settings import Settings
from kalshi_bot.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class KellyResult:
    """Result of Kelly criterion calculation."""

    optimal_fraction: float
    adjusted_fraction: float
    expected_value: float
    edge: float
    win_probability: float
    loss_probability: float
    win_payout: float
    loss_payout: float


class KellyCriterion:
    """
    Implements Kelly criterion for optimal position sizing.

    The Kelly criterion calculates the optimal fraction of bankroll to bet
    to maximize long-term growth while minimizing risk of ruin.

    Standard Kelly formula:
        f* = (bp - q) / b

    Where:
        f* = fraction of bankroll to bet
        b = odds received on the bet (net payout per dollar bet)
        p = probability of winning
        q = probability of losing (1 - p)

    For prediction markets:
        b = (1 - price) / price  (for YES contracts at given price)
        p = estimated true probability
    """

    def __init__(self, settings: Settings) -> None:
        """
        Initialize Kelly calculator.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self._kelly_fraction = settings.trading.kelly_fraction
        self._execution_confidence = settings.trading.execution_confidence
        self._min_edge = settings.trading.min_edge

    def calculate(
        self,
        win_probability: float,
        price_cents: int,
        is_yes: bool = True,
    ) -> KellyResult:
        """
        Calculate Kelly criterion for a single bet.

        Args:
            win_probability: Estimated probability of winning (0-1)
            price_cents: Price in cents (1-99)
            is_yes: Whether this is a YES contract

        Returns:
            KellyResult with optimal sizing
        """
        price = price_cents / 100.0

        if is_yes:
            payout = 1.0 - price
            cost = price
        else:
            payout = price
            cost = 1.0 - price

        b = payout / cost

        p = win_probability
        q = 1.0 - p

        edge = b * p - q
        expected_value = p * payout - q * cost

        if edge <= 0:
            return KellyResult(
                optimal_fraction=0.0,
                adjusted_fraction=0.0,
                expected_value=expected_value,
                edge=edge,
                win_probability=p,
                loss_probability=q,
                win_payout=payout,
                loss_payout=cost,
            )

        f_star = edge / b

        f_adjusted = f_star * self._kelly_fraction * self._execution_confidence

        f_adjusted = max(0.0, min(f_adjusted, 0.25))

        return KellyResult(
            optimal_fraction=f_star,
            adjusted_fraction=f_adjusted,
            expected_value=expected_value,
            edge=edge,
            win_probability=p,
            loss_probability=q,
            win_payout=payout,
            loss_payout=cost,
        )

    def calculate_arbitrage_size(
        self,
        total_cost_cents: int,
        guaranteed_payout: int,
        available_quantity: int,
        bankroll: float,
    ) -> int:
        """
        Calculate position size for an arbitrage opportunity.

        For pure arbitrage (guaranteed profit), Kelly suggests betting
        the entire bankroll. However, we apply conservative constraints.

        Args:
            total_cost_cents: Total cost per contract set in cents
            guaranteed_payout: Guaranteed payout per contract set (usually 100)
            available_quantity: Maximum contracts available
            bankroll: Available trading balance in dollars

        Returns:
            Recommended number of contracts
        """
        profit_cents = guaranteed_payout - total_cost_cents
        if profit_cents <= 0:
            return 0

        edge = profit_cents / total_cost_cents

        if edge < self._min_edge:
            logger.debug(f"Edge {edge:.4f} below minimum {self._min_edge}")
            return 0

        cost_per_contract = total_cost_cents / 100.0

        max_by_bankroll = int(bankroll / cost_per_contract)

        position_limit = int(bankroll * self.settings.trading.max_position_pct / cost_per_contract)

        kelly_fraction = min(edge * self._kelly_fraction * self._execution_confidence, 0.25)
        kelly_contracts = int(bankroll * kelly_fraction / cost_per_contract)

        recommended = min(
            available_quantity,
            max_by_bankroll,
            position_limit,
            kelly_contracts,
            self.settings.trading.max_position_size,
        )

        return max(0, recommended)

    def calculate_portfolio_allocation(
        self,
        opportunities: list[dict],
        bankroll: float,
    ) -> dict[str, int]:
        """
        Calculate optimal allocation across multiple opportunities.

        Uses a simplified multi-asset Kelly approach where each opportunity
        is treated independently with position limits.

        Args:
            opportunities: List of opportunity dicts with cost and profit
            bankroll: Total available bankroll

        Returns:
            Dict mapping opportunity_id to recommended contracts
        """
        allocations: dict[str, int] = {}
        remaining_bankroll = bankroll

        sorted_opps = sorted(
            opportunities,
            key=lambda x: x.get("edge", 0),
            reverse=True,
        )

        max_per_opp = bankroll * self.settings.trading.max_position_pct

        for opp in sorted_opps:
            if remaining_bankroll <= 0:
                break

            opp_id = opp.get("opportunity_id", "")
            total_cost_cents = opp.get("total_cost_cents", 0)
            profit_cents = opp.get("profit_cents", 0)
            available = opp.get("max_quantity", 0)

            if total_cost_cents <= 0:
                continue

            edge = profit_cents / total_cost_cents
            if edge < self._min_edge:
                continue

            usable_bankroll = min(remaining_bankroll, max_per_opp)

            contracts = self.calculate_arbitrage_size(
                total_cost_cents,
                100,
                available,
                usable_bankroll,
            )

            if contracts > 0:
                allocations[opp_id] = contracts
                cost = (total_cost_cents * contracts) / 100
                remaining_bankroll -= cost

        return allocations
