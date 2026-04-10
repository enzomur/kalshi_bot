"""Position tracking for backtesting."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from kalshi_bot.backtesting.simulator import SimulatedTrade
from kalshi_bot.core.types import Side
from kalshi_bot.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BacktestPosition:
    """A position held during backtesting."""

    market_ticker: str
    side: Side
    quantity: int
    average_price: float  # In cents
    total_cost: float  # In dollars (including fees)
    opened_at: datetime
    opportunity_id: str
    trades: list[SimulatedTrade] = field(default_factory=list)

    @property
    def cost_basis(self) -> float:
        """Cost basis in dollars (price only, no fees)."""
        return (self.average_price * self.quantity) / 100

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "market_ticker": self.market_ticker,
            "side": self.side.value,
            "quantity": self.quantity,
            "average_price": self.average_price,
            "total_cost": self.total_cost,
            "opened_at": self.opened_at.isoformat(),
            "opportunity_id": self.opportunity_id,
        }


@dataclass
class SettlementResult:
    """Result of a position settlement."""

    market_ticker: str
    side: Side
    quantity: int
    cost_basis: float  # In dollars
    payout: float  # In dollars
    profit: float  # In dollars
    outcome: str  # "yes" or "no"
    settled_at: datetime
    opportunity_id: str

    @property
    def return_pct(self) -> float:
        """Return percentage."""
        if self.cost_basis <= 0:
            return 0.0
        return (self.payout - self.cost_basis) / self.cost_basis

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "market_ticker": self.market_ticker,
            "side": self.side.value,
            "quantity": self.quantity,
            "cost_basis": self.cost_basis,
            "payout": self.payout,
            "profit": self.profit,
            "outcome": self.outcome,
            "settled_at": self.settled_at.isoformat(),
            "opportunity_id": self.opportunity_id,
            "return_pct": self.return_pct,
        }


class PositionTracker:
    """
    Tracks positions and settlements during backtesting.

    Handles:
    - Recording positions from simulated trades
    - Settling positions when markets resolve
    - Calculating P&L for each position
    - Tracking position history
    """

    def __init__(self) -> None:
        """Initialize the position tracker."""
        # Active positions: ticker -> list of positions
        self._positions: dict[str, list[BacktestPosition]] = {}
        # Settled positions
        self._settlements: list[SettlementResult] = []
        # All positions ever opened
        self._position_history: list[BacktestPosition] = []

    @property
    def open_positions(self) -> dict[str, list[BacktestPosition]]:
        """Get all open positions."""
        return self._positions

    @property
    def settlements(self) -> list[SettlementResult]:
        """Get all settlement results."""
        return self._settlements

    @property
    def position_history(self) -> list[BacktestPosition]:
        """Get all positions ever opened."""
        return self._position_history

    @property
    def total_open_positions(self) -> int:
        """Total number of open positions."""
        return sum(len(positions) for positions in self._positions.values())

    @property
    def total_open_quantity(self) -> int:
        """Total contracts in open positions."""
        return sum(
            pos.quantity
            for positions in self._positions.values()
            for pos in positions
        )

    @property
    def total_exposure(self) -> float:
        """Total dollar exposure in open positions."""
        return sum(
            pos.total_cost
            for positions in self._positions.values()
            for pos in positions
        )

    def reset(self) -> None:
        """Reset all position tracking state."""
        self._positions = {}
        self._settlements = []
        self._position_history = []

    def record_trade(self, trade: SimulatedTrade, timestamp: datetime) -> BacktestPosition:
        """
        Record a trade and create/update position.

        Args:
            trade: The simulated trade
            timestamp: Trade timestamp

        Returns:
            The created or updated position
        """
        ticker = trade.market_ticker
        side = trade.side

        # Create new position for this trade
        position = BacktestPosition(
            market_ticker=ticker,
            side=side,
            quantity=trade.quantity,
            average_price=trade.price,
            total_cost=trade.total_cost,
            opened_at=timestamp,
            opportunity_id=trade.opportunity_id,
            trades=[trade],
        )

        # Add to positions
        if ticker not in self._positions:
            self._positions[ticker] = []
        self._positions[ticker].append(position)
        self._position_history.append(position)

        logger.debug(
            f"Recorded position: {ticker} {side.value} {trade.quantity}@{trade.price}"
        )

        return position

    def record_trades(
        self,
        trades: list[SimulatedTrade],
        timestamp: datetime,
    ) -> list[BacktestPosition]:
        """
        Record multiple trades.

        Args:
            trades: List of trades
            timestamp: Trade timestamp

        Returns:
            List of created positions
        """
        return [self.record_trade(trade, timestamp) for trade in trades]

    def settle_market(
        self,
        ticker: str,
        outcome: str,
        settled_at: datetime,
    ) -> list[SettlementResult]:
        """
        Settle all positions in a market.

        Args:
            ticker: Market ticker
            outcome: Settlement outcome ("yes" or "no")
            settled_at: Settlement timestamp

        Returns:
            List of settlement results
        """
        if ticker not in self._positions:
            return []

        positions = self._positions.pop(ticker)
        results = []

        for position in positions:
            # Calculate payout
            # YES wins: YES holders get $1 per contract, NO holders get $0
            # NO wins: NO holders get $1 per contract, YES holders get $0
            if (outcome == "yes" and position.side == Side.YES) or \
               (outcome == "no" and position.side == Side.NO):
                payout = position.quantity  # $1 per contract
            else:
                payout = 0.0

            # Cost basis is just the price paid (total_cost includes fees)
            cost_basis = position.cost_basis
            profit = payout - position.total_cost  # Subtract total cost including fees

            result = SettlementResult(
                market_ticker=ticker,
                side=position.side,
                quantity=position.quantity,
                cost_basis=cost_basis,
                payout=payout,
                profit=profit,
                outcome=outcome,
                settled_at=settled_at,
                opportunity_id=position.opportunity_id,
            )

            results.append(result)
            self._settlements.append(result)

            logger.debug(
                f"Settled {ticker}: {position.side.value} {position.quantity} -> "
                f"payout=${payout:.2f}, profit=${profit:.2f}"
            )

        return results

    def get_positions_for_market(self, ticker: str) -> list[BacktestPosition]:
        """Get all open positions for a market."""
        return self._positions.get(ticker, [])

    def get_net_position(self, ticker: str) -> tuple[int, int]:
        """
        Get net position for a market.

        Returns:
            Tuple of (yes_quantity, no_quantity)
        """
        positions = self._positions.get(ticker, [])
        yes_qty = sum(p.quantity for p in positions if p.side == Side.YES)
        no_qty = sum(p.quantity for p in positions if p.side == Side.NO)
        return yes_qty, no_qty

    def get_total_pnl(self) -> float:
        """Get total realized P&L from all settlements."""
        return sum(s.profit for s in self._settlements)

    def get_winning_trades(self) -> list[SettlementResult]:
        """Get all winning (profitable) settlements."""
        return [s for s in self._settlements if s.profit > 0]

    def get_losing_trades(self) -> list[SettlementResult]:
        """Get all losing settlements."""
        return [s for s in self._settlements if s.profit < 0]

    def get_settlements_by_opportunity(self, opportunity_id: str) -> list[SettlementResult]:
        """Get all settlements for an opportunity."""
        return [s for s in self._settlements if s.opportunity_id == opportunity_id]

    def get_opportunity_pnl(self, opportunity_id: str) -> float:
        """Get total P&L for an opportunity (all legs combined)."""
        settlements = self.get_settlements_by_opportunity(opportunity_id)
        return sum(s.profit for s in settlements)

    def get_status(self) -> dict[str, Any]:
        """Get tracker status summary."""
        return {
            "open_positions": self.total_open_positions,
            "open_quantity": self.total_open_quantity,
            "total_exposure": self.total_exposure,
            "total_settlements": len(self._settlements),
            "realized_pnl": self.get_total_pnl(),
            "winning_trades": len(self.get_winning_trades()),
            "losing_trades": len(self.get_losing_trades()),
        }
