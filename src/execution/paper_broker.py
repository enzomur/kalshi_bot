"""Paper trading broker with realistic fill simulation.

This broker simulates trade execution without making real API calls.
It provides realistic fill behavior including:
- Checking orderbook liquidity
- Applying slippage based on order size
- Supporting partial fills
- Tracking positions and P&L
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.core.types import (
    Order,
    Fill,
    Position,
    Side,
    OrderType,
    OrderStatus,
    OrderBook,
)
from src.execution.base import BaseBroker, ExecutionResult
from src.ledger.database import Database
from src.observability.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PaperPosition:
    """Internal position tracking for paper trading."""

    market_ticker: str
    side: Side
    quantity: int
    average_price: float
    total_cost: float
    realized_pnl: float = 0.0

    def to_position(self) -> Position:
        """Convert to core Position type."""
        return Position(
            market_ticker=self.market_ticker,
            side=self.side,
            quantity=self.quantity,
            average_price=self.average_price,
            market_exposure=self.total_cost,
            realized_pnl=self.realized_pnl,
        )


class PaperBroker(BaseBroker):
    """Paper trading broker with realistic fill simulation.

    Features:
    - Simulates order execution with configurable slippage
    - Tracks positions and P&L
    - Persists state to database
    - Simulates partial fills based on liquidity
    """

    DEFAULT_INITIAL_BALANCE = 1000.0
    DEFAULT_SLIPPAGE_BPS = 10  # 0.1% slippage
    FEE_PER_CONTRACT = 0.01  # $0.01 per contract

    def __init__(
        self,
        db: Database | None = None,
        initial_balance: float = DEFAULT_INITIAL_BALANCE,
        slippage_bps: int = DEFAULT_SLIPPAGE_BPS,
    ) -> None:
        """
        Initialize paper broker.

        Args:
            db: Database for persistence (optional)
            initial_balance: Starting balance in dollars
            slippage_bps: Slippage in basis points (100 = 1%)
        """
        self._db = db
        self._initial_balance = initial_balance
        self._balance = initial_balance
        self._slippage_bps = slippage_bps

        self._positions: dict[str, PaperPosition] = {}
        self._orders: dict[str, Order] = {}
        self._fills: list[Fill] = []

        self._total_pnl = 0.0
        self._total_fees = 0.0
        self._trade_count = 0
        self._win_count = 0
        self._loss_count = 0

    @property
    def is_paper(self) -> bool:
        return True

    @property
    def balance(self) -> float:
        return self._balance

    async def execute(
        self,
        market_ticker: str,
        side: Side,
        quantity: int,
        price: int,
        decision_id: str | None = None,
    ) -> ExecutionResult:
        """Execute a paper trade with realistic fill simulation."""
        start_time = datetime.now(timezone.utc)

        # Validate inputs
        if quantity <= 0:
            return ExecutionResult(
                success=False,
                order=None,
                fills=[],
                total_cost=0.0,
                total_fees=0.0,
                filled_quantity=0,
                average_price=0.0,
                error_message="Invalid quantity",
            )

        if not 1 <= price <= 99:
            return ExecutionResult(
                success=False,
                order=None,
                fills=[],
                total_cost=0.0,
                total_fees=0.0,
                filled_quantity=0,
                average_price=0.0,
                error_message="Invalid price (must be 1-99)",
            )

        # Calculate expected cost
        expected_cost = (price * quantity) / 100
        fees = self.FEE_PER_CONTRACT * quantity

        if expected_cost + fees > self._balance:
            return ExecutionResult(
                success=False,
                order=None,
                fills=[],
                total_cost=0.0,
                total_fees=0.0,
                filled_quantity=0,
                average_price=0.0,
                error_message=(
                    f"Insufficient balance: need ${expected_cost + fees:.2f}, "
                    f"have ${self._balance:.2f}"
                ),
            )

        # Apply slippage (simulate market impact)
        fill_price = self._apply_slippage(price, side, quantity)

        # Create order
        order_id = f"paper-{uuid.uuid4().hex[:12]}"
        order = Order(
            order_id=order_id,
            market_ticker=market_ticker,
            side=side,
            order_type=OrderType.LIMIT,
            price=fill_price,
            quantity=quantity,
            status=OrderStatus.FILLED,
            filled_quantity=quantity,
            remaining_quantity=0,
        )

        # Create fill
        fill_id = f"fill-{uuid.uuid4().hex[:8]}"
        fill = Fill(
            fill_id=fill_id,
            order_id=order_id,
            market_ticker=market_ticker,
            side=side,
            price=fill_price,
            quantity=quantity,
            fee=fees,
        )

        # Calculate actual cost
        actual_cost = (fill_price * quantity) / 100

        # Update balance
        self._balance -= actual_cost + fees

        # Update position
        position_key = f"{market_ticker}:{side.value}"
        if position_key in self._positions:
            pos = self._positions[position_key]
            new_quantity = pos.quantity + quantity
            new_avg_price = (
                (pos.average_price * pos.quantity) + (fill_price * quantity)
            ) / new_quantity
            pos.quantity = new_quantity
            pos.average_price = new_avg_price
            pos.total_cost += actual_cost
        else:
            self._positions[position_key] = PaperPosition(
                market_ticker=market_ticker,
                side=side,
                quantity=quantity,
                average_price=float(fill_price),
                total_cost=actual_cost,
            )

        # Update metrics
        self._total_fees += fees
        self._trade_count += 1
        self._orders[order_id] = order
        self._fills.append(fill)

        # Persist to database
        if self._db is not None:
            try:
                await self._db.save_order(order, decision_id, is_paper=True)
                await self._db.save_fill(fill, is_paper=True)
                await self._db.update_position(
                    market_ticker=market_ticker,
                    side=side,
                    quantity=self._positions[position_key].quantity,
                    average_price=self._positions[position_key].average_price,
                    market_exposure=self._positions[position_key].total_cost,
                    is_paper=True,
                )
            except Exception as e:
                logger.warning(f"Failed to persist paper trade: {e}")

        execution_time = (
            datetime.now(timezone.utc) - start_time
        ).total_seconds() * 1000

        logger.info(
            f"[PAPER] Executed {side.value} {quantity}x {market_ticker} "
            f"@ {fill_price}c, cost=${actual_cost:.2f}, fees=${fees:.2f}"
        )

        return ExecutionResult(
            success=True,
            order=order,
            fills=[fill],
            total_cost=actual_cost,
            total_fees=fees,
            filled_quantity=quantity,
            average_price=float(fill_price),
            execution_time_ms=execution_time,
            metadata={"decision_id": decision_id} if decision_id else {},
        )

    def _apply_slippage(self, price: int, side: Side, quantity: int) -> int:
        """Apply slippage to the fill price.

        Larger orders get worse prices due to market impact.
        """
        # Base slippage
        slippage_pct = self._slippage_bps / 10000

        # Scale slippage with order size (larger orders = more impact)
        size_factor = 1 + (quantity / 100) * 0.1  # +10% slippage per 100 contracts

        # Random component (market noise)
        random_factor = 1 + random.uniform(-0.2, 0.5)

        total_slippage = slippage_pct * size_factor * random_factor

        # For buys, price goes up; for sells, price goes down
        # Since we're always buying (entering positions), price goes up
        slippage_cents = int(price * total_slippage)
        slippage_cents = max(0, min(slippage_cents, 5))  # Cap at 5 cents

        fill_price = price + slippage_cents

        # Ensure valid price range
        return max(1, min(99, fill_price))

    async def simulate_fill_with_orderbook(
        self,
        market_ticker: str,
        side: Side,
        quantity: int,
        price: int,
        orderbook: OrderBook,
    ) -> tuple[int, int]:
        """Simulate a fill based on actual orderbook liquidity.

        Args:
            market_ticker: Market ticker
            side: YES or NO
            quantity: Desired quantity
            price: Limit price
            orderbook: Current orderbook

        Returns:
            Tuple of (filled_quantity, average_price)
        """
        # Get relevant side of book
        if side == Side.YES:
            levels = orderbook.yes_asks
        else:
            levels = orderbook.no_asks

        if not levels:
            return 0, 0

        filled = 0
        total_cost = 0

        for level in levels:
            if level.price > price:
                break  # Exceeded limit price

            available = min(level.quantity, quantity - filled)
            filled += available
            total_cost += level.price * available

            if filled >= quantity:
                break

        if filled == 0:
            return 0, 0

        avg_price = total_cost / filled
        return filled, int(avg_price)

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order (no-op in paper trading - orders fill instantly)."""
        if order_id in self._orders:
            order = self._orders[order_id]
            if order.status in (OrderStatus.PENDING, OrderStatus.OPEN, OrderStatus.RESTING):
                order.status = OrderStatus.CANCELLED
                return True
        return False

    async def cancel_all_orders(self, market_ticker: str | None = None) -> int:
        """Cancel all pending orders."""
        cancelled = 0
        for order in self._orders.values():
            if market_ticker and order.market_ticker != market_ticker:
                continue
            if order.status in (OrderStatus.PENDING, OrderStatus.OPEN, OrderStatus.RESTING):
                order.status = OrderStatus.CANCELLED
                cancelled += 1
        return cancelled

    async def get_positions(self) -> list[Position]:
        """Get all current positions."""
        return [pos.to_position() for pos in self._positions.values() if pos.quantity > 0]

    async def get_position(self, market_ticker: str, side: Side) -> Position | None:
        """Get a specific position."""
        position_key = f"{market_ticker}:{side.value}"
        if position_key in self._positions:
            pos = self._positions[position_key]
            if pos.quantity > 0:
                return pos.to_position()
        return None

    async def sync_positions(self) -> None:
        """Sync positions from database."""
        if self._db is None:
            return

        try:
            db_positions = await self._db.get_positions(is_paper=True)
            for pos_data in db_positions:
                position_key = f"{pos_data['market_ticker']}:{pos_data['side']}"
                self._positions[position_key] = PaperPosition(
                    market_ticker=pos_data["market_ticker"],
                    side=Side(pos_data["side"]),
                    quantity=pos_data["quantity"],
                    average_price=pos_data["average_price"],
                    total_cost=pos_data["market_exposure"],
                    realized_pnl=pos_data.get("realized_pnl", 0.0),
                )
            logger.info(f"Synced {len(db_positions)} positions from database")
        except Exception as e:
            logger.warning(f"Failed to sync positions: {e}")

    async def settle_position(
        self,
        market_ticker: str,
        side: Side,
        settlement_result: str,  # "yes" or "no"
    ) -> float:
        """Settle a position and calculate P&L.

        Args:
            market_ticker: Market ticker
            side: Position side
            settlement_result: "yes" or "no"

        Returns:
            Realized P&L in dollars
        """
        position_key = f"{market_ticker}:{side.value}"
        if position_key not in self._positions:
            return 0.0

        pos = self._positions[position_key]
        if pos.quantity == 0:
            return 0.0

        # Calculate settlement value
        # If you bought YES and market settles YES, you get $1 per contract
        # If you bought YES and market settles NO, you get $0
        won = (side == Side.YES and settlement_result == "yes") or (
            side == Side.NO and settlement_result == "no"
        )

        if won:
            # Get $1 per contract
            payout = pos.quantity
        else:
            # Get $0
            payout = 0.0

        # P&L = payout - cost
        pnl = payout - pos.total_cost
        pos.realized_pnl = pnl

        # Update balance
        self._balance += payout

        # Update metrics
        self._total_pnl += pnl
        if pnl > 0:
            self._win_count += 1
        else:
            self._loss_count += 1

        # Clear position
        pos.quantity = 0
        pos.total_cost = 0.0

        logger.info(
            f"[PAPER] Settled {side.value} position in {market_ticker}: "
            f"result={settlement_result}, pnl=${pnl:.2f}"
        )

        return pnl

    def get_status(self) -> dict[str, Any]:
        """Get broker status."""
        return {
            "mode": "paper",
            "initial_balance": self._initial_balance,
            "current_balance": self._balance,
            "total_pnl": self._total_pnl,
            "total_fees": self._total_fees,
            "trade_count": self._trade_count,
            "win_count": self._win_count,
            "loss_count": self._loss_count,
            "win_rate": (
                self._win_count / (self._win_count + self._loss_count)
                if (self._win_count + self._loss_count) > 0
                else 0.0
            ),
            "open_positions": len(
                [p for p in self._positions.values() if p.quantity > 0]
            ),
            "slippage_bps": self._slippage_bps,
        }

    async def close(self) -> None:
        """Clean up resources."""
        logger.info("Paper broker closed")

    def reset(self) -> None:
        """Reset paper trading state."""
        self._balance = self._initial_balance
        self._positions.clear()
        self._orders.clear()
        self._fills.clear()
        self._total_pnl = 0.0
        self._total_fees = 0.0
        self._trade_count = 0
        self._win_count = 0
        self._loss_count = 0
        logger.info("Paper broker reset")
