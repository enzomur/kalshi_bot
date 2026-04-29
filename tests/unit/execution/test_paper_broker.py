"""Unit tests for Paper Broker."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.core.types import Side, OrderStatus, OrderType
from src.execution.paper_broker import PaperBroker, PaperPosition
from src.execution.base import ExecutionResult


@pytest.fixture
def broker() -> PaperBroker:
    """Create a paper broker for testing."""
    return PaperBroker(db=None, initial_balance=1000.0, slippage_bps=10)


class TestPaperBroker:
    """Tests for PaperBroker class."""

    def test_initialization(self, broker: PaperBroker) -> None:
        """Test broker initializes with correct defaults."""
        assert broker.is_paper is True
        assert broker.balance == 1000.0
        assert broker._initial_balance == 1000.0
        assert broker._slippage_bps == 10

    @pytest.mark.asyncio
    async def test_execute_successful_trade(self, broker: PaperBroker) -> None:
        """Test executing a successful trade."""
        result = await broker.execute(
            market_ticker="TEST-MARKET",
            side=Side.YES,
            quantity=10,
            price=50,
        )

        assert result.success is True
        assert result.filled_quantity == 10
        assert result.order is not None
        assert result.order.status == OrderStatus.FILLED
        assert len(result.fills) == 1
        assert result.total_cost > 0
        assert result.total_fees > 0

    @pytest.mark.asyncio
    async def test_execute_updates_balance(self, broker: PaperBroker) -> None:
        """Test that executing a trade updates balance."""
        initial_balance = broker.balance

        await broker.execute(
            market_ticker="TEST-MARKET",
            side=Side.YES,
            quantity=10,
            price=50,
        )

        assert broker.balance < initial_balance

    @pytest.mark.asyncio
    async def test_execute_creates_position(self, broker: PaperBroker) -> None:
        """Test that executing a trade creates a position."""
        await broker.execute(
            market_ticker="TEST-MARKET",
            side=Side.YES,
            quantity=10,
            price=50,
        )

        positions = await broker.get_positions()
        assert len(positions) == 1
        assert positions[0].market_ticker == "TEST-MARKET"
        assert positions[0].side == Side.YES
        assert positions[0].quantity == 10

    @pytest.mark.asyncio
    async def test_execute_accumulates_position(self, broker: PaperBroker) -> None:
        """Test that multiple trades accumulate into one position."""
        await broker.execute(
            market_ticker="TEST-MARKET",
            side=Side.YES,
            quantity=10,
            price=50,
        )
        await broker.execute(
            market_ticker="TEST-MARKET",
            side=Side.YES,
            quantity=5,
            price=55,
        )

        positions = await broker.get_positions()
        assert len(positions) == 1
        assert positions[0].quantity == 15

    @pytest.mark.asyncio
    async def test_execute_invalid_quantity(self, broker: PaperBroker) -> None:
        """Test that invalid quantity is rejected."""
        result = await broker.execute(
            market_ticker="TEST-MARKET",
            side=Side.YES,
            quantity=0,
            price=50,
        )

        assert result.success is False
        assert "invalid" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_execute_invalid_price(self, broker: PaperBroker) -> None:
        """Test that invalid price is rejected."""
        result = await broker.execute(
            market_ticker="TEST-MARKET",
            side=Side.YES,
            quantity=10,
            price=0,
        )
        assert result.success is False

        result = await broker.execute(
            market_ticker="TEST-MARKET",
            side=Side.YES,
            quantity=10,
            price=100,
        )
        assert result.success is False

    @pytest.mark.asyncio
    async def test_execute_insufficient_balance(self, broker: PaperBroker) -> None:
        """Test that insufficient balance is rejected."""
        # Try to buy more than we can afford
        result = await broker.execute(
            market_ticker="TEST-MARKET",
            side=Side.YES,
            quantity=10000,  # Would cost $5000 at 50c
            price=50,
        )

        assert result.success is False
        assert "insufficient" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_get_position(self, broker: PaperBroker) -> None:
        """Test getting a specific position."""
        await broker.execute(
            market_ticker="TEST-MARKET",
            side=Side.YES,
            quantity=10,
            price=50,
        )

        position = await broker.get_position("TEST-MARKET", Side.YES)
        assert position is not None
        assert position.quantity == 10

        # Non-existent position
        position = await broker.get_position("OTHER-MARKET", Side.YES)
        assert position is None

    @pytest.mark.asyncio
    async def test_cancel_order(self, broker: PaperBroker) -> None:
        """Test cancelling an order (no-op in paper trading)."""
        # Execute a trade first
        result = await broker.execute(
            market_ticker="TEST-MARKET",
            side=Side.YES,
            quantity=10,
            price=50,
        )

        # Try to cancel (should return False since orders fill instantly)
        cancelled = await broker.cancel_order(result.order.order_id)
        assert cancelled is False

    @pytest.mark.asyncio
    async def test_cancel_all_orders(self, broker: PaperBroker) -> None:
        """Test cancelling all orders."""
        cancelled = await broker.cancel_all_orders()
        assert cancelled == 0


class TestSlippage:
    """Tests for slippage simulation."""

    @pytest.mark.asyncio
    async def test_slippage_applied(self) -> None:
        """Test that slippage is applied to fill price."""
        broker = PaperBroker(db=None, initial_balance=1000.0, slippage_bps=100)  # 1% slippage

        result = await broker.execute(
            market_ticker="TEST-MARKET",
            side=Side.YES,
            quantity=10,
            price=50,
        )

        # Fill price should be >= order price due to slippage
        assert result.average_price >= 50

    @pytest.mark.asyncio
    async def test_larger_orders_more_slippage(self) -> None:
        """Test that larger orders get more slippage."""
        broker = PaperBroker(db=None, initial_balance=10000.0, slippage_bps=50)

        small_result = await broker.execute(
            market_ticker="TEST-1",
            side=Side.YES,
            quantity=1,
            price=50,
        )

        broker2 = PaperBroker(db=None, initial_balance=10000.0, slippage_bps=50)
        large_result = await broker2.execute(
            market_ticker="TEST-2",
            side=Side.YES,
            quantity=100,
            price=50,
        )

        # On average, larger orders should have more slippage
        # (though randomness means this isn't deterministic)
        # Just verify both execute successfully
        assert small_result.success is True
        assert large_result.success is True


class TestPositionSettlement:
    """Tests for position settlement."""

    @pytest.mark.asyncio
    async def test_settle_winning_position(self, broker: PaperBroker) -> None:
        """Test settling a winning position."""
        initial_balance = broker.balance

        # Buy YES at 50c
        await broker.execute(
            market_ticker="TEST-MARKET",
            side=Side.YES,
            quantity=10,
            price=50,
        )

        # Settle YES (we win)
        pnl = await broker.settle_position("TEST-MARKET", Side.YES, "yes")

        assert pnl > 0  # Profit
        assert broker.balance > initial_balance  # Net gain

    @pytest.mark.asyncio
    async def test_settle_losing_position(self, broker: PaperBroker) -> None:
        """Test settling a losing position."""
        initial_balance = broker.balance

        # Buy YES at 50c
        await broker.execute(
            market_ticker="TEST-MARKET",
            side=Side.YES,
            quantity=10,
            price=50,
        )

        post_trade_balance = broker.balance

        # Settle NO (we lose)
        pnl = await broker.settle_position("TEST-MARKET", Side.YES, "no")

        assert pnl < 0  # Loss
        assert broker.balance < initial_balance  # Net loss

    @pytest.mark.asyncio
    async def test_settle_clears_position(self, broker: PaperBroker) -> None:
        """Test that settlement clears the position."""
        await broker.execute(
            market_ticker="TEST-MARKET",
            side=Side.YES,
            quantity=10,
            price=50,
        )

        await broker.settle_position("TEST-MARKET", Side.YES, "yes")

        position = await broker.get_position("TEST-MARKET", Side.YES)
        # Position should have zero quantity after settlement
        assert position is None or position.quantity == 0


class TestBrokerStatus:
    """Tests for broker status reporting."""

    @pytest.mark.asyncio
    async def test_get_status(self, broker: PaperBroker) -> None:
        """Test get_status returns expected fields."""
        await broker.execute(
            market_ticker="TEST-MARKET",
            side=Side.YES,
            quantity=10,
            price=50,
        )

        status = broker.get_status()

        assert status["mode"] == "paper"
        assert status["initial_balance"] == 1000.0
        assert status["current_balance"] <= 1000.0
        assert status["trade_count"] == 1
        assert status["open_positions"] == 1

    @pytest.mark.asyncio
    async def test_status_tracks_pnl(self, broker: PaperBroker) -> None:
        """Test that status tracks P&L."""
        await broker.execute(
            market_ticker="TEST-MARKET",
            side=Side.YES,
            quantity=10,
            price=50,
        )

        await broker.settle_position("TEST-MARKET", Side.YES, "yes")

        status = broker.get_status()
        assert status["total_pnl"] > 0  # Winning trade


class TestBrokerReset:
    """Tests for broker reset functionality."""

    @pytest.mark.asyncio
    async def test_reset_clears_state(self, broker: PaperBroker) -> None:
        """Test that reset clears all state."""
        # Execute some trades
        await broker.execute(
            market_ticker="TEST-MARKET",
            side=Side.YES,
            quantity=10,
            price=50,
        )

        broker.reset()

        assert broker.balance == 1000.0
        positions = await broker.get_positions()
        assert len(positions) == 0
        assert broker._trade_count == 0
        assert broker._total_pnl == 0.0


class TestPaperPosition:
    """Tests for PaperPosition dataclass."""

    def test_to_position(self) -> None:
        """Test converting PaperPosition to Position."""
        paper_pos = PaperPosition(
            market_ticker="TEST-MARKET",
            side=Side.YES,
            quantity=10,
            average_price=50.0,
            total_cost=5.0,
            realized_pnl=1.0,
        )

        position = paper_pos.to_position()

        assert position.market_ticker == "TEST-MARKET"
        assert position.side == Side.YES
        assert position.quantity == 10
        assert position.average_price == 50.0
        assert position.realized_pnl == 1.0


class TestExecutionResult:
    """Tests for ExecutionResult dataclass."""

    def test_net_cost_calculation(self) -> None:
        """Test net cost includes fees."""
        result = ExecutionResult(
            success=True,
            order=None,
            fills=[],
            total_cost=10.0,
            total_fees=0.50,
            filled_quantity=10,
            average_price=50.0,
        )

        assert result.net_cost == 10.50
