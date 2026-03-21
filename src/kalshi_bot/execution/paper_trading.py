"""Paper trading executor for risk-free testing."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from kalshi_bot.config.settings import Settings
from kalshi_bot.core.types import (
    ArbitrageOpportunity,
    Order,
    OrderStatus,
    OrderType,
    Side,
    Trade,
)
from kalshi_bot.execution.executor import ExecutionResult, LegExecution
from kalshi_bot.persistence.database import Database
from kalshi_bot.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PaperPosition:
    """Simulated position for paper trading."""

    market_ticker: str
    side: Side
    quantity: int
    average_price: float
    opened_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class PaperTradeRecord:
    """Record of a paper trade."""

    trade_id: str
    opportunity_id: str
    markets: list[str]
    legs: list[dict[str, Any]]
    total_cost: float
    fees: float
    net_profit: float
    quantity: int
    executed_at: datetime = field(default_factory=datetime.utcnow)
    is_paper: bool = True


class PaperTradingExecutor:
    """
    Simulates trade execution without real API calls.

    Used for testing strategies in a risk-free environment.
    Tracks simulated P&L and positions, records paper trades
    to database for analysis, and applies same validation as
    the real executor.
    """

    def __init__(
        self,
        settings: Settings,
        db: Database | None = None,
    ) -> None:
        """
        Initialize paper trading executor.

        Args:
            settings: Application settings
            db: Database for persisting paper trades
        """
        self.settings = settings
        self.db = db

        self._positions: dict[str, PaperPosition] = {}
        self._trade_history: list[PaperTradeRecord] = []
        self._simulated_balance: float = 0.0
        self._initial_balance: float = 0.0
        self._total_pnl: float = 0.0
        self._total_fees: float = 0.0
        self._trade_count: int = 0

        # Use same slippage settings as real executor
        self._max_slippage_cents = getattr(
            settings.trading, 'max_slippage_cents', 2
        )
        self._min_net_profit_pct = getattr(
            settings.trading, 'min_net_profit_pct', 0.02
        )

    def set_initial_balance(self, balance: float) -> None:
        """Set initial simulated balance."""
        self._initial_balance = balance
        self._simulated_balance = balance
        logger.info(f"Paper trading initialized with ${balance:.2f} balance")

    async def execute_opportunity(
        self,
        opportunity: ArbitrageOpportunity,
        quantity: int,
    ) -> ExecutionResult:
        """
        Simulate execution of an arbitrage opportunity.

        Applies same validation as real executor but doesn't
        call the actual API.

        Args:
            opportunity: The opportunity to execute
            quantity: Number of contracts per leg

        Returns:
            ExecutionResult with simulated details
        """
        start_time = datetime.utcnow()

        # Validate minimum net profit percentage
        if opportunity.total_cost > 0:
            net_profit_pct = opportunity.net_profit / opportunity.total_cost
            if net_profit_pct < self._min_net_profit_pct:
                logger.warning(
                    f"Paper trade rejected: net profit {net_profit_pct:.2%} "
                    f"< minimum {self._min_net_profit_pct:.2%}"
                )
                return ExecutionResult(
                    success=False,
                    opportunity_id=opportunity.opportunity_id,
                    orders=[],
                    trades=[],
                    total_cost=0.0,
                    total_fees=0.0,
                    filled_quantity=0,
                    partial_fill=False,
                    error_message=f"Net profit {net_profit_pct:.2%} below minimum {self._min_net_profit_pct:.2%}",
                    execution_time_ms=0.0,
                )

        # Simulate leg executions
        orders: list[Order] = []
        trades: list[Trade] = []
        leg_results: list[LegExecution] = []

        logger.info(
            f"[PAPER] Executing opportunity {opportunity.opportunity_id}: "
            f"{len(opportunity.legs)} legs, {quantity} contracts each"
        )

        # Simulate each leg
        for leg in opportunity.legs:
            order = Order(
                order_id=f"paper-{uuid.uuid4().hex[:12]}",
                market_ticker=leg["market"],
                side=Side(leg["side"]),
                order_type=OrderType.LIMIT,
                price=leg["price"],
                quantity=quantity,
                status=OrderStatus.FILLED,
                filled_quantity=quantity,
                remaining_quantity=0,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )
            orders.append(order)

            trade = Trade(
                trade_id=f"paper-trade-{uuid.uuid4().hex[:8]}",
                order_id=order.order_id,
                market_ticker=leg["market"],
                side=Side(leg["side"]),
                price=leg["price"],
                quantity=quantity,
                fee=0.0,  # Fees calculated separately
                executed_at=datetime.utcnow(),
            )
            trades.append(trade)

            leg_results.append(LegExecution(
                market_ticker=leg["market"],
                side=Side(leg["side"]),
                order=order,
                filled=True,
                filled_quantity=quantity,
                average_price=float(leg["price"]),
                error=None,
            ))

            # Update simulated positions
            position_key = f"{leg['market']}_{leg['side']}"
            if position_key in self._positions:
                existing = self._positions[position_key]
                total_qty = existing.quantity + quantity
                avg_price = (
                    (existing.average_price * existing.quantity) +
                    (leg["price"] * quantity)
                ) / total_qty
                existing.quantity = total_qty
                existing.average_price = avg_price
            else:
                self._positions[position_key] = PaperPosition(
                    market_ticker=leg["market"],
                    side=Side(leg["side"]),
                    quantity=quantity,
                    average_price=float(leg["price"]),
                )

        # Calculate costs using the opportunity's pre-calculated values
        # Scale by actual quantity vs max quantity
        scale_factor = quantity / opportunity.max_quantity if opportunity.max_quantity > 0 else 1.0
        total_cost = opportunity.total_cost * scale_factor
        total_fees = opportunity.fees * scale_factor
        net_profit = opportunity.net_profit * scale_factor

        # Update simulated balance
        self._simulated_balance -= total_cost
        self._total_pnl += net_profit
        self._total_fees += total_fees
        self._trade_count += 1

        execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        # Record paper trade
        paper_trade = PaperTradeRecord(
            trade_id=f"paper-{uuid.uuid4().hex[:12]}",
            opportunity_id=opportunity.opportunity_id,
            markets=opportunity.markets,
            legs=opportunity.legs,
            total_cost=total_cost,
            fees=total_fees,
            net_profit=net_profit,
            quantity=quantity,
            executed_at=datetime.utcnow(),
        )
        self._trade_history.append(paper_trade)

        # Persist to database if available
        if self.db:
            await self._persist_paper_trade(paper_trade)

        logger.info(
            f"[PAPER] Execution complete: success=True, "
            f"filled={quantity}, cost=${total_cost:.2f}, "
            f"net_profit=${net_profit:.4f}, time={execution_time:.0f}ms"
        )

        return ExecutionResult(
            success=True,
            opportunity_id=opportunity.opportunity_id,
            orders=orders,
            trades=trades,
            total_cost=total_cost,
            total_fees=total_fees,
            filled_quantity=quantity,
            partial_fill=False,
            error_message=None,
            execution_time_ms=execution_time,
        )

    async def _persist_paper_trade(self, trade: PaperTradeRecord) -> None:
        """Persist paper trade to database."""
        if not self.db:
            return

        try:
            async with self.db.get_session() as session:
                await session.execute(
                    """
                    INSERT INTO paper_trades (
                        trade_id, opportunity_id, markets, legs,
                        total_cost, fees, net_profit, quantity,
                        executed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        trade.trade_id,
                        trade.opportunity_id,
                        ",".join(trade.markets),
                        str(trade.legs),
                        trade.total_cost,
                        trade.fees,
                        trade.net_profit,
                        trade.quantity,
                        trade.executed_at.isoformat(),
                    ),
                )
                await session.commit()
        except Exception as e:
            # Table may not exist, just log
            logger.debug(f"Could not persist paper trade: {e}")

    async def cancel_all_pending(self) -> int:
        """No-op for paper trading - no pending orders."""
        return 0

    def get_simulated_pnl(self) -> float:
        """Get total simulated P&L."""
        return self._total_pnl

    def get_simulated_balance(self) -> float:
        """Get current simulated balance."""
        return self._simulated_balance + self._total_pnl

    def get_status(self) -> dict[str, Any]:
        """Get paper trading status."""
        return {
            "mode": "paper",
            "initial_balance": self._initial_balance,
            "current_balance": self.get_simulated_balance(),
            "total_pnl": self._total_pnl,
            "total_fees": self._total_fees,
            "trade_count": self._trade_count,
            "open_positions": len(self._positions),
            "max_slippage_cents": self._max_slippage_cents,
            "min_net_profit_pct": self._min_net_profit_pct,
        }

    def get_trade_history(self) -> list[PaperTradeRecord]:
        """Get paper trade history."""
        return self._trade_history.copy()

    def reset(self) -> None:
        """Reset paper trading state."""
        self._positions.clear()
        self._trade_history.clear()
        self._simulated_balance = self._initial_balance
        self._total_pnl = 0.0
        self._total_fees = 0.0
        self._trade_count = 0
        logger.info("Paper trading state reset")
