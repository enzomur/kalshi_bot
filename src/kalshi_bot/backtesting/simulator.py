"""Trade simulator for backtesting."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from kalshi_bot.arbitrage.strategies.base import ArbitrageStrategy
from kalshi_bot.core.types import ArbitrageOpportunity, Side, Trade
from kalshi_bot.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SimulatedTrade:
    """Represents a simulated trade execution."""

    trade_id: str
    opportunity_id: str
    market_ticker: str
    side: Side
    action: str  # "buy" or "sell"
    price: int  # In cents
    quantity: int
    fee: float  # In dollars
    cost: float  # Total cost in dollars (price * quantity / 100)
    executed_at: datetime
    strategy: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_cost(self) -> float:
        """Total cost including fees in dollars."""
        return self.cost + self.fee


@dataclass
class SimulationResult:
    """Result of simulating an opportunity."""

    success: bool
    trades: list[SimulatedTrade]
    total_cost: float
    total_fees: float
    error_message: str | None = None


class TradeSimulator:
    """
    Simulates trade execution for backtesting.

    Handles:
    - Balance validation
    - Position limit validation
    - Fee calculation using Kalshi formula
    - Trade recording
    """

    def __init__(
        self,
        initial_balance: float = 10000.0,
        max_position_per_market: int = 1000,
        max_position_pct: float = 0.10,
    ) -> None:
        """
        Initialize the trade simulator.

        Args:
            initial_balance: Starting balance in dollars
            max_position_per_market: Maximum contracts per market
            max_position_pct: Maximum position size as % of balance
        """
        self._initial_balance = initial_balance
        self._balance = initial_balance
        self._max_position_per_market = max_position_per_market
        self._max_position_pct = max_position_pct

        self._trades: list[SimulatedTrade] = []
        self._total_fees = 0.0
        self._total_cost = 0.0

    @property
    def balance(self) -> float:
        """Current available balance."""
        return self._balance

    @property
    def initial_balance(self) -> float:
        """Initial balance."""
        return self._initial_balance

    @property
    def trades(self) -> list[SimulatedTrade]:
        """List of all simulated trades."""
        return self._trades

    @property
    def total_fees(self) -> float:
        """Total fees paid."""
        return self._total_fees

    @property
    def total_invested(self) -> float:
        """Total amount invested (cost of positions)."""
        return self._total_cost

    def reset(self, balance: float | None = None) -> None:
        """
        Reset simulator state.

        Args:
            balance: New balance, uses initial_balance if not provided
        """
        self._balance = balance if balance is not None else self._initial_balance
        self._trades = []
        self._total_fees = 0.0
        self._total_cost = 0.0

    def calculate_fee(self, price: int, quantity: int = 1) -> float:
        """
        Calculate Kalshi trading fee.

        Uses the formula: 0.07 * P * (1-P) per contract
        Where P is price/100 (price as decimal)

        Args:
            price: Price in cents (1-99)
            quantity: Number of contracts

        Returns:
            Total fee in dollars
        """
        return ArbitrageStrategy.calculate_fee(price, quantity)

    def validate_trade(
        self,
        cost: float,
        quantity: int,
        market_ticker: str,
    ) -> tuple[bool, str | None]:
        """
        Validate if a trade can be executed.

        Args:
            cost: Total cost in dollars
            quantity: Number of contracts
            market_ticker: Market being traded

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check balance
        if cost > self._balance:
            return False, f"Insufficient balance: need ${cost:.2f}, have ${self._balance:.2f}"

        # Check position size
        if quantity > self._max_position_per_market:
            return False, f"Position too large: {quantity} > {self._max_position_per_market}"

        # Check position as % of balance
        max_cost = self._balance * self._max_position_pct
        if cost > max_cost:
            return False, f"Position exceeds {self._max_position_pct:.0%} of balance"

        return True, None

    def simulate_opportunity(
        self,
        opportunity: ArbitrageOpportunity,
        timestamp: datetime,
        max_quantity: int | None = None,
    ) -> SimulationResult:
        """
        Simulate executing an arbitrage opportunity.

        Args:
            opportunity: The opportunity to execute
            timestamp: Simulation timestamp
            max_quantity: Override max quantity (for position sizing)

        Returns:
            SimulationResult with trades and costs
        """
        # Determine quantity to trade
        quantity = max_quantity or opportunity.max_quantity

        # Adjust quantity based on position limits
        max_cost = self._balance * self._max_position_pct
        if quantity > self._max_position_per_market:
            quantity = self._max_position_per_market

        # Calculate total cost for all legs at this quantity
        total_cost = 0.0
        total_fees = 0.0
        legs_to_execute = []

        for leg in opportunity.legs:
            price = leg["price"]
            leg_quantity = quantity  # Use same quantity for all legs
            fee = self.calculate_fee(price, leg_quantity)
            cost = (price * leg_quantity) / 100

            total_cost += cost
            total_fees += fee

            legs_to_execute.append({
                **leg,
                "quantity": leg_quantity,
                "fee": fee,
                "cost": cost,
            })

        # Validate the trade
        is_valid, error = self.validate_trade(
            total_cost + total_fees,
            quantity,
            opportunity.markets[0],
        )

        if not is_valid:
            return SimulationResult(
                success=False,
                trades=[],
                total_cost=0.0,
                total_fees=0.0,
                error_message=error,
            )

        # Execute the trades
        trades = []
        for leg in legs_to_execute:
            trade = SimulatedTrade(
                trade_id=f"bt-{uuid.uuid4().hex[:8]}",
                opportunity_id=opportunity.opportunity_id,
                market_ticker=leg["market"],
                side=Side(leg["side"]),
                action=leg["action"],
                price=leg["price"],
                quantity=leg["quantity"],
                fee=leg["fee"],
                cost=leg["cost"],
                executed_at=timestamp,
                strategy=opportunity.arbitrage_type.value,
                metadata={
                    "opportunity_type": opportunity.arbitrage_type.value,
                    "expected_profit": opportunity.expected_profit,
                },
            )
            trades.append(trade)
            self._trades.append(trade)

        # Update balance and totals
        self._balance -= (total_cost + total_fees)
        self._total_cost += total_cost
        self._total_fees += total_fees

        logger.debug(
            f"Simulated {len(trades)} trades for {opportunity.opportunity_id}: "
            f"cost=${total_cost:.2f}, fees=${total_fees:.4f}"
        )

        return SimulationResult(
            success=True,
            trades=trades,
            total_cost=total_cost,
            total_fees=total_fees,
        )

    def add_funds(self, amount: float) -> None:
        """
        Add funds to balance (e.g., from settlement payout).

        Args:
            amount: Amount in dollars
        """
        self._balance += amount

    def get_trades_by_market(self, ticker: str) -> list[SimulatedTrade]:
        """Get all trades for a specific market."""
        return [t for t in self._trades if t.market_ticker == ticker]

    def get_trades_by_strategy(self, strategy: str) -> list[SimulatedTrade]:
        """Get all trades for a specific strategy."""
        return [t for t in self._trades if t.strategy == strategy]

    def get_status(self) -> dict:
        """Get simulator status."""
        return {
            "initial_balance": self._initial_balance,
            "current_balance": self._balance,
            "total_trades": len(self._trades),
            "total_fees": self._total_fees,
            "total_invested": self._total_cost,
        }
