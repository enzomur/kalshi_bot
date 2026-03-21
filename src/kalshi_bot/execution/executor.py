"""Trade executor for submitting and managing orders."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from kalshi_bot.api.client import KalshiAPIClient
from kalshi_bot.config.settings import Settings
from kalshi_bot.core.exceptions import ExecutionError, OrderError
from kalshi_bot.core.types import (
    ArbitrageOpportunity,
    Order,
    OrderStatus,
    OrderType,
    Side,
    Trade,
)
from kalshi_bot.persistence.models import OrderRepository, TradeRepository
from kalshi_bot.utils.logging import get_logger
from kalshi_bot.utils.notifications import send_notification

logger = get_logger(__name__)


@dataclass
class ExecutionResult:
    """Result of executing an arbitrage opportunity."""

    success: bool
    opportunity_id: str
    orders: list[Order]
    trades: list[Trade]
    total_cost: float
    total_fees: float
    filled_quantity: int
    partial_fill: bool
    error_message: str | None = None
    execution_time_ms: float = 0.0


@dataclass
class LegExecution:
    """Execution result for a single leg."""

    market_ticker: str
    side: Side
    order: Order | None
    filled: bool
    filled_quantity: int
    average_price: float
    error: str | None = None


class TradeExecutor:
    """
    Handles execution of arbitrage trades.

    Responsibilities:
    - Submit orders for each leg of an arbitrage
    - Monitor fills and handle partial fills
    - Cancel unfilled orders on failure
    - Calculate actual costs and P&L
    - Persist trade records
    """

    def __init__(
        self,
        settings: Settings,
        api_client: KalshiAPIClient,
        order_repo: OrderRepository | None = None,
        trade_repo: TradeRepository | None = None,
    ) -> None:
        """
        Initialize trade executor.

        Args:
            settings: Application settings
            api_client: Kalshi API client
            order_repo: Order repository
            trade_repo: Trade repository
        """
        self.settings = settings
        self.api_client = api_client
        self.order_repo = order_repo
        self.trade_repo = trade_repo

        self._fill_timeout = 10.0
        self._poll_interval = 0.5
        self._max_retries = 3

        # Slippage and profit protection settings
        self._max_slippage_cents = getattr(
            settings.trading, 'max_slippage_cents', 2
        )
        self._min_net_profit_pct = getattr(
            settings.trading, 'min_net_profit_pct', 0.02
        )

    async def execute_opportunity(
        self,
        opportunity: ArbitrageOpportunity,
        quantity: int,
    ) -> ExecutionResult:
        """
        Execute an arbitrage opportunity.

        Submits orders for all legs and monitors for fills.
        On partial failure, cancels remaining orders.

        Args:
            opportunity: The opportunity to execute
            quantity: Number of contracts per leg

        Returns:
            ExecutionResult with details
        """
        start_time = datetime.utcnow()
        orders: list[Order] = []
        trades: list[Trade] = []
        leg_results: list[LegExecution] = []

        logger.info(
            f"Executing opportunity {opportunity.opportunity_id}: "
            f"{len(opportunity.legs)} legs, {quantity} contracts each"
        )

        # Pre-execution validation: verify all legs are still profitable
        is_valid, validation_error = await self._validate_all_legs_profitable(opportunity)
        if not is_valid:
            logger.warning(
                f"Opportunity {opportunity.opportunity_id} failed pre-execution validation: {validation_error}"
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
                error_message=validation_error,
            )

        try:
            for leg in opportunity.legs:
                leg_result = await self._execute_leg(
                    leg, quantity, opportunity.opportunity_id
                )
                leg_results.append(leg_result)

                if leg_result.order:
                    orders.append(leg_result.order)

                if not leg_result.filled:
                    logger.warning(
                        f"Leg failed: {leg['market']} {leg['side']} - {leg_result.error}"
                    )
                    await self._cancel_remaining_orders(orders)
                    break

            all_filled = all(r.filled for r in leg_results)
            min_filled = min((r.filled_quantity for r in leg_results), default=0)
            partial_fill = not all_filled and min_filled > 0

            total_cost = 0.0
            total_fees = 0.0

            for result in leg_results:
                if result.filled_quantity > 0:
                    cost = (result.average_price * result.filled_quantity) / 100
                    total_cost += cost

                    from kalshi_bot.arbitrage.strategies.base import ArbitrageStrategy
                    fee = ArbitrageStrategy.calculate_fee(
                        int(result.average_price), result.filled_quantity
                    )
                    total_fees += fee

            for order in orders:
                if order.filled_quantity > 0:
                    trade = Trade(
                        trade_id=f"trade-{uuid.uuid4().hex[:8]}",
                        order_id=order.order_id,
                        market_ticker=order.market_ticker,
                        side=order.side,
                        price=order.price,
                        quantity=order.filled_quantity,
                        fee=total_fees / len(orders) if orders else 0,
                        executed_at=datetime.utcnow(),
                    )
                    trades.append(trade)

                    if self.trade_repo:
                        try:
                            await self.trade_repo.save(trade, opportunity.opportunity_id)
                        except Exception as e:
                            logger.error(f"Failed to save trade: {e}")

            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000

            error_msg = None
            if not all_filled:
                failed_legs = [r for r in leg_results if not r.filled]
                if failed_legs:
                    error_msg = f"Failed legs: {[r.error for r in failed_legs]}"

            result = ExecutionResult(
                success=all_filled,
                opportunity_id=opportunity.opportunity_id,
                orders=orders,
                trades=trades,
                total_cost=total_cost,
                total_fees=total_fees,
                filled_quantity=min_filled,
                partial_fill=partial_fill,
                error_message=error_msg,
                execution_time_ms=execution_time,
            )

            logger.info(
                f"Execution complete: success={result.success}, "
                f"filled={result.filled_quantity}, cost=${result.total_cost:.2f}, "
                f"time={result.execution_time_ms:.0f}ms"
            )

            if result.success:
                markets = ", ".join(t.market_ticker for t in trades[:2])
                if len(trades) > 2:
                    markets += f" +{len(trades) - 2} more"
                send_notification(
                    "Trade Executed",
                    f"{result.filled_quantity} contracts | ${result.total_cost:.2f} cost | {markets}",
                )

            return result

        except Exception as e:
            logger.error(f"Execution error: {e}")
            await self._cancel_remaining_orders(orders)

            return ExecutionResult(
                success=False,
                opportunity_id=opportunity.opportunity_id,
                orders=orders,
                trades=trades,
                total_cost=0.0,
                total_fees=0.0,
                filled_quantity=0,
                partial_fill=False,
                error_message=str(e),
            )

    async def _execute_leg(
        self,
        leg: dict[str, Any],
        quantity: int,
        opportunity_id: str,
    ) -> LegExecution:
        """
        Execute a single leg of the arbitrage.

        Args:
            leg: Leg definition with market, side, price
            quantity: Number of contracts
            opportunity_id: Parent opportunity ID

        Returns:
            LegExecution result
        """
        market_ticker = leg["market"]
        side = Side(leg["side"])
        price = leg["price"]

        logger.debug(f"Executing leg: {market_ticker} {side.value} @ {price}c x{quantity}")

        try:
            order = await self.api_client.create_order(
                ticker=market_ticker,
                side=side,
                order_type=OrderType.LIMIT,
                quantity=quantity,
                price=price,
                client_order_id=f"{opportunity_id[-8:]}-{uuid.uuid4().hex[:8]}",
            )

            if self.order_repo:
                await self.order_repo.save(order, opportunity_id)

            # Check if already filled from create response
            if order.status in (OrderStatus.FILLED, OrderStatus.EXECUTED) or order.filled_quantity >= quantity:
                order.status = OrderStatus.FILLED
                filled_order = order
            else:
                filled_order = await self._wait_for_fill(order)

            if self.order_repo:
                await self.order_repo.update_status(
                    order.order_id,
                    filled_order.status,
                    filled_order.filled_quantity,
                    filled_order.remaining_quantity,
                )

            is_filled = filled_order.status == OrderStatus.FILLED
            filled_qty = filled_order.filled_quantity

            return LegExecution(
                market_ticker=market_ticker,
                side=side,
                order=filled_order,
                filled=is_filled,
                filled_quantity=filled_qty,
                average_price=float(price),
                error=None if is_filled else f"Order status: {filled_order.status.value}",
            )

        except Exception as e:
            logger.error(f"Leg execution failed: {market_ticker} - {e}")
            return LegExecution(
                market_ticker=market_ticker,
                side=side,
                order=None,
                filled=False,
                filled_quantity=0,
                average_price=0.0,
                error=str(e),
            )

    async def _wait_for_fill(self, order: Order) -> Order:
        """
        Wait for an order to fill with timeout.

        Args:
            order: Order to monitor

        Returns:
            Updated order with fill status
        """
        deadline = datetime.utcnow().timestamp() + self._fill_timeout
        consecutive_404s = 0

        while datetime.utcnow().timestamp() < deadline:
            try:
                updated = await self.api_client.get_order(order.order_id)
                consecutive_404s = 0  # Reset on success

                if updated.status in (
                    OrderStatus.FILLED,
                    OrderStatus.EXECUTED,
                    OrderStatus.CANCELLED,
                    OrderStatus.EXPIRED,
                    OrderStatus.REJECTED,
                ):
                    return updated

                if updated.filled_quantity >= order.quantity:
                    updated.status = OrderStatus.FILLED
                    return updated

            except Exception as e:
                error_str = str(e)
                if "404" in error_str:
                    consecutive_404s += 1
                    # If we get 3+ consecutive 404s, assume order filled and was removed
                    if consecutive_404s >= 3:
                        logger.info(f"Order {order.order_id} returned 404 - assuming filled")
                        order.status = OrderStatus.FILLED
                        order.filled_quantity = order.quantity
                        return order
                else:
                    logger.warning(f"Error polling order {order.order_id}: {e}")

            await asyncio.sleep(self._poll_interval)

        logger.warning(f"Order {order.order_id} timed out waiting for fill")

        try:
            await self.api_client.cancel_order(order.order_id)
        except Exception as e:
            # 404 on cancel likely means order already filled/gone
            if "404" in str(e):
                logger.info(f"Order {order.order_id} not found on cancel - assuming filled")
                order.status = OrderStatus.FILLED
                order.filled_quantity = order.quantity
                return order
            logger.error(f"Failed to cancel timed-out order: {e}")

        try:
            return await self.api_client.get_order(order.order_id)
        except Exception as e:
            if "404" in str(e):
                order.status = OrderStatus.FILLED
                order.filled_quantity = order.quantity
                return order
            raise

    async def _cancel_remaining_orders(self, orders: list[Order]) -> None:
        """
        Cancel any unfilled orders.

        Args:
            orders: List of orders to potentially cancel
        """
        for order in orders:
            if order.status in (OrderStatus.PENDING, OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED):
                try:
                    await self.api_client.cancel_order(order.order_id)
                    logger.info(f"Cancelled order {order.order_id}")

                    if self.order_repo:
                        await self.order_repo.update_status(
                            order.order_id, OrderStatus.CANCELLED
                        )

                except Exception as e:
                    logger.error(f"Failed to cancel order {order.order_id}: {e}")

    async def cancel_all_pending(self, market_ticker: str | None = None) -> int:
        """
        Cancel all pending orders.

        Args:
            market_ticker: Optional filter by market

        Returns:
            Number of orders cancelled
        """
        try:
            result = await self.api_client.cancel_all_orders(market_ticker)
            cancelled = result.get("cancelled_orders", 0)
            logger.info(f"Cancelled {cancelled} orders")
            return cancelled
        except Exception as e:
            logger.error(f"Failed to cancel all orders: {e}")
            return 0

    async def execute_single_order(
        self,
        market_ticker: str,
        side: Side,
        quantity: int,
        price: int | None = None,
        order_type: OrderType = OrderType.LIMIT,
    ) -> Order:
        """
        Execute a single order (not part of arbitrage).

        Args:
            market_ticker: Market to trade
            side: YES or NO
            quantity: Number of contracts
            price: Price in cents (required for limit orders)
            order_type: LIMIT or MARKET

        Returns:
            Executed order
        """
        order = await self.api_client.create_order(
            ticker=market_ticker,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
        )

        if order_type == OrderType.LIMIT:
            order = await self._wait_for_fill(order)

        if self.order_repo:
            await self.order_repo.save(order)

        return order

    async def _validate_price_current(
        self,
        market_ticker: str,
        expected_price: int,
        side: Side,
    ) -> tuple[bool, int | None]:
        """
        Re-fetch orderbook and validate price hasn't moved too much.

        Args:
            market_ticker: Market to check
            expected_price: Price we expected to execute at
            side: Side of the trade (YES or NO)

        Returns:
            Tuple of (is_valid, current_price)
        """
        try:
            orderbook = await self.api_client.get_orderbook(market_ticker)

            if side == Side.YES:
                current_price = orderbook.best_yes_ask
            else:
                current_price = orderbook.best_no_ask

            if current_price is None:
                return False, None

            slippage = abs(current_price - expected_price)
            if slippage > self._max_slippage_cents:
                logger.warning(
                    f"Price slippage too high for {market_ticker}: "
                    f"expected {expected_price}c, current {current_price}c, "
                    f"slippage {slippage}c > max {self._max_slippage_cents}c"
                )
                return False, current_price

            return True, current_price

        except Exception as e:
            logger.error(f"Failed to validate price for {market_ticker}: {e}")
            return False, None

    async def _validate_all_legs_profitable(
        self,
        opportunity: ArbitrageOpportunity,
    ) -> tuple[bool, str | None]:
        """
        Re-validate ALL legs still sum to < 100 cents and profit is sufficient.

        This is the atomic validation before any execution - ensures the
        arbitrage is still valid at current market prices.

        Args:
            opportunity: The opportunity to validate

        Returns:
            Tuple of (is_valid, error_reason)
        """
        try:
            total_cost_cents = 0

            for leg in opportunity.legs:
                market_ticker = leg["market"]
                side = Side(leg["side"])
                expected_price = leg["price"]

                is_valid, current_price = await self._validate_price_current(
                    market_ticker, expected_price, side
                )

                if not is_valid:
                    return False, f"Price validation failed for {market_ticker}"

                # Use current price for calculation
                total_cost_cents += current_price

            # Check if still profitable (sum < 100 for true arbitrage)
            if total_cost_cents >= 100:
                return False, f"Total cost {total_cost_cents}c >= 100c, no longer profitable"

            # Calculate net profit with current prices
            gross_profit_cents = 100 - total_cost_cents
            quantity = opportunity.max_quantity

            # Recalculate fees with current prices
            from kalshi_bot.arbitrage.strategies.base import ArbitrageStrategy
            total_fees = 0.0
            for leg in opportunity.legs:
                # Approximate current price
                fee = ArbitrageStrategy.calculate_fee(leg["price"], quantity)
                total_fees += fee

            total_cost_dollars = (total_cost_cents * quantity) / 100
            gross_profit_dollars = (gross_profit_cents * quantity) / 100
            net_profit_dollars = gross_profit_dollars - total_fees

            if net_profit_dollars <= 0:
                return False, f"Net profit ${net_profit_dollars:.4f} <= 0 after fees"

            # Check minimum profit percentage (2% default)
            if total_cost_dollars > 0:
                net_profit_pct = net_profit_dollars / total_cost_dollars
                if net_profit_pct < self._min_net_profit_pct:
                    return False, (
                        f"Net profit {net_profit_pct:.2%} < "
                        f"minimum {self._min_net_profit_pct:.2%}"
                    )

            logger.info(
                f"Pre-execution validation passed: "
                f"total_cost={total_cost_cents}c, net_profit=${net_profit_dollars:.4f}"
            )
            return True, None

        except Exception as e:
            logger.error(f"Pre-execution validation error: {e}")
            return False, f"Validation error: {str(e)}"

    def get_status(self) -> dict[str, Any]:
        """Get executor status."""
        return {
            "fill_timeout": self._fill_timeout,
            "poll_interval": self._poll_interval,
            "max_retries": self._max_retries,
            "max_slippage_cents": self._max_slippage_cents,
            "min_net_profit_pct": self._min_net_profit_pct,
        }
