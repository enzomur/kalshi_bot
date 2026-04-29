"""Unit tests for Risk Engine."""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.types import Signal, TradingMode, Side, Position
from src.risk.engine import (
    RiskEngine,
    RiskDecision,
    CircuitBreakerState,
    PositionCheck,
)
from src.execution.paper_broker import PaperBroker


@pytest.fixture
def mock_broker() -> PaperBroker:
    """Create a mock paper broker."""
    broker = PaperBroker(db=None, initial_balance=1000.0)
    return broker


@pytest.fixture
def mock_mode_manager() -> MagicMock:
    """Create a mock mode manager."""
    manager = MagicMock()
    manager.current_mode = TradingMode.PAPER
    manager.config = MagicMock()
    manager.config.max_position_dollars = 10000.0
    manager.config.max_daily_loss_dollars = 10000.0
    return manager


@pytest.fixture
def risk_engine(mock_broker: PaperBroker, mock_mode_manager: MagicMock) -> RiskEngine:
    """Create a risk engine for testing."""
    return RiskEngine(
        broker=mock_broker,
        db=None,
        mode_manager=mock_mode_manager,
        kelly_fraction=0.25,
        max_position_contracts=100,
        max_daily_loss=100.0,
        max_drawdown=0.20,
        max_consecutive_losses=5,
    )


@pytest.fixture
def valid_signal() -> Signal:
    """Create a valid trading signal."""
    return Signal.create(
        strategy_name="test_strategy",
        market_ticker="TEST-MARKET",
        direction="yes",
        target_probability=0.70,
        market_probability=0.50,
        confidence=0.80,
        max_position=50,
        metadata={"market_price_cents": 50},
    )


class TestRiskEngine:
    """Tests for RiskEngine class."""

    @pytest.mark.asyncio
    async def test_approve_valid_signal(
        self, risk_engine: RiskEngine, valid_signal: Signal
    ) -> None:
        """Test that a valid signal is approved."""
        decision = await risk_engine.evaluate_signal(valid_signal)

        assert decision.approved is True
        assert decision.approved_size > 0
        assert decision.rejection_reason is None

    @pytest.mark.asyncio
    async def test_reject_expired_signal(
        self, risk_engine: RiskEngine
    ) -> None:
        """Test that an expired signal is rejected."""
        expired_signal = Signal(
            signal_id="test-123",
            strategy_name="test",
            market_ticker="TEST",
            direction="yes",
            target_probability=0.70,
            confidence=0.80,
            edge=0.20,
            max_position=50,
            metadata={"market_price_cents": 50},
            created_at=datetime.now(timezone.utc) - timedelta(hours=2),
            expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )

        decision = await risk_engine.evaluate_signal(expired_signal)

        assert decision.approved is False
        assert "expired" in decision.rejection_reason.lower()

    @pytest.mark.asyncio
    async def test_reject_when_circuit_breaker_triggered(
        self, risk_engine: RiskEngine, valid_signal: Signal
    ) -> None:
        """Test that signals are rejected when circuit breaker is triggered."""
        # Trigger the daily loss circuit breaker
        risk_engine._circuit_breakers.daily_loss_triggered = True
        risk_engine._circuit_breakers.daily_loss_amount = 150.0

        decision = await risk_engine.evaluate_signal(valid_signal)

        assert decision.approved is False
        assert "circuit breaker" in decision.rejection_reason.lower()

    @pytest.mark.asyncio
    async def test_reject_negative_ev(
        self, risk_engine: RiskEngine
    ) -> None:
        """Test that negative EV signals are rejected."""
        negative_ev_signal = Signal.create(
            strategy_name="test",
            market_ticker="TEST",
            direction="yes",
            target_probability=0.40,  # Below market price
            market_probability=0.50,
            confidence=0.80,
            max_position=50,
            metadata={"market_price_cents": 50},
        )

        decision = await risk_engine.evaluate_signal(negative_ev_signal)

        assert decision.approved is False
        assert "negative" in decision.rejection_reason.lower() or "ev" in decision.rejection_reason.lower()

    @pytest.mark.asyncio
    async def test_position_limit_enforcement(
        self, risk_engine: RiskEngine, valid_signal: Signal, mock_broker: PaperBroker
    ) -> None:
        """Test that position limits are enforced."""
        # Simulate existing position at max
        mock_broker._positions["TEST-MARKET:yes"] = MagicMock()
        mock_broker._positions["TEST-MARKET:yes"].quantity = 100
        mock_broker._positions["TEST-MARKET:yes"].side = Side.YES
        mock_broker._positions["TEST-MARKET:yes"].to_position = lambda: Position(
            market_ticker="TEST-MARKET",
            side=Side.YES,
            quantity=100,
            average_price=50.0,
            market_exposure=50.0,
        )

        decision = await risk_engine.evaluate_signal(valid_signal)

        assert decision.approved is False
        assert "position" in decision.rejection_reason.lower()


class TestCircuitBreakers:
    """Tests for circuit breaker functionality."""

    def test_daily_loss_trigger(self, risk_engine: RiskEngine) -> None:
        """Test daily loss circuit breaker trigger."""
        # Update daily loss
        risk_engine.update_daily_loss(50.0)
        assert risk_engine._circuit_breakers.daily_loss_triggered is False

        risk_engine.update_daily_loss(60.0)  # Total = 110, exceeds 100
        assert risk_engine._circuit_breakers.daily_loss_triggered is True

    def test_consecutive_loss_trigger(self, risk_engine: RiskEngine) -> None:
        """Test consecutive loss circuit breaker trigger."""
        # Record losses
        for _ in range(4):
            risk_engine.record_trade_result(won=False)
        assert risk_engine._circuit_breakers.consecutive_loss_triggered is False

        risk_engine.record_trade_result(won=False)  # 5th loss
        assert risk_engine._circuit_breakers.consecutive_loss_triggered is True

    def test_consecutive_loss_reset_on_win(self, risk_engine: RiskEngine) -> None:
        """Test that consecutive losses reset on a win."""
        # Record some losses
        for _ in range(3):
            risk_engine.record_trade_result(won=False)
        assert risk_engine._circuit_breakers.consecutive_losses == 3

        # Win resets counter
        risk_engine.record_trade_result(won=True)
        assert risk_engine._circuit_breakers.consecutive_losses == 0

    def test_drawdown_trigger(self, risk_engine: RiskEngine, mock_broker: PaperBroker) -> None:
        """Test drawdown circuit breaker trigger."""
        # Set initial peak
        risk_engine._peak_balance = 1000.0
        risk_engine._initial_balance = 1000.0

        # Simulate loss bringing balance to 750 (25% drawdown)
        mock_broker._balance = 750.0
        risk_engine.update_drawdown()

        assert risk_engine._circuit_breakers.drawdown_triggered is True
        assert risk_engine._circuit_breakers.current_drawdown >= 0.20

    def test_reset_daily_breakers(self, risk_engine: RiskEngine) -> None:
        """Test resetting daily circuit breakers."""
        risk_engine._circuit_breakers.daily_loss_triggered = True
        risk_engine._circuit_breakers.daily_loss_amount = 150.0

        risk_engine.reset_daily_breakers()

        assert risk_engine._circuit_breakers.daily_loss_triggered is False
        assert risk_engine._circuit_breakers.daily_loss_amount == 0.0


class TestCircuitBreakerState:
    """Tests for CircuitBreakerState dataclass."""

    def test_is_triggered_daily_loss(self) -> None:
        """Test is_triggered with daily loss."""
        state = CircuitBreakerState(daily_loss_triggered=True)
        assert state.is_triggered() is True

    def test_is_triggered_drawdown(self) -> None:
        """Test is_triggered with drawdown."""
        state = CircuitBreakerState(drawdown_triggered=True)
        assert state.is_triggered() is True

    def test_is_triggered_consecutive(self) -> None:
        """Test is_triggered with consecutive losses."""
        state = CircuitBreakerState(consecutive_loss_triggered=True)
        assert state.is_triggered() is True

    def test_is_triggered_cooldown(self) -> None:
        """Test is_triggered with active cooldown."""
        future_time = datetime.now(timezone.utc) + timedelta(minutes=30)
        state = CircuitBreakerState(cooldown_until=future_time)
        assert state.is_triggered() is True

    def test_is_triggered_cooldown_expired(self) -> None:
        """Test is_triggered with expired cooldown."""
        past_time = datetime.now(timezone.utc) - timedelta(minutes=30)
        state = CircuitBreakerState(cooldown_until=past_time)
        assert state.is_triggered() is False

    def test_to_dict(self) -> None:
        """Test to_dict serialization."""
        state = CircuitBreakerState(
            daily_loss_triggered=True,
            daily_loss_amount=50.0,
            consecutive_losses=3,
        )
        d = state.to_dict()

        assert d["daily_loss_triggered"] is True
        assert d["daily_loss_amount"] == 50.0
        assert d["consecutive_losses"] == 3


class TestPositionCheck:
    """Tests for PositionCheck dataclass."""

    def test_approved_position(self) -> None:
        """Test approved position check."""
        check = PositionCheck(
            approved=True,
            current_position=10,
            max_allowed=100,
        )
        assert check.approved is True
        assert check.reason is None

    def test_rejected_position(self) -> None:
        """Test rejected position check."""
        check = PositionCheck(
            approved=False,
            current_position=100,
            max_allowed=100,
            reason="At maximum position",
        )
        assert check.approved is False
        assert check.reason == "At maximum position"

    def test_to_dict(self) -> None:
        """Test to_dict serialization."""
        check = PositionCheck(
            approved=True,
            current_position=50,
            max_allowed=100,
        )
        d = check.to_dict()

        assert d["approved"] is True
        assert d["current_position"] == 50
        assert d["max_allowed"] == 100


class TestRiskDecision:
    """Tests for RiskDecision dataclass."""

    def test_approved_decision_to_dict(self, valid_signal: Signal) -> None:
        """Test to_dict for approved decision."""
        from src.risk.kelly import KellyResult

        decision = RiskDecision(
            decision_id="dec-123",
            signal_id=valid_signal.signal_id,
            approved=True,
            approved_size=10,
            rejection_reason=None,
            circuit_breaker_status=CircuitBreakerState(),
            position_check=PositionCheck(approved=True, current_position=0, max_allowed=100),
            kelly_result=KellyResult(
                full_kelly_fraction=0.20,
                position_fraction=0.05,
                position_dollars=50.0,
                position_contracts=10,
                edge=0.10,
                expected_value=0.20,
                is_positive_ev=True,
            ),
            mode_caps={"max_position_dollars": 1000.0},
            trading_mode=TradingMode.PAPER,
        )

        d = decision.to_dict()

        assert d["approved"] is True
        assert d["approved_size"] == 10
        assert d["rejection_reason"] is None
        assert d["trading_mode"] == "paper"


class TestModeSpecificLimits:
    """Tests for mode-specific position limits."""

    @pytest.mark.asyncio
    async def test_probation_mode_limits(
        self, mock_broker: PaperBroker, valid_signal: Signal
    ) -> None:
        """Test that LIVE_PROBATION has reduced limits."""
        mode_manager = MagicMock()
        mode_manager.current_mode = TradingMode.LIVE_PROBATION
        mode_manager.config = MagicMock()
        mode_manager.config.max_position_dollars = 500.0
        mode_manager.config.max_daily_loss_dollars = 100.0

        engine = RiskEngine(
            broker=mock_broker,
            db=None,
            mode_manager=mode_manager,
        )

        decision = await engine.evaluate_signal(valid_signal)

        # Check that mode caps are applied
        assert decision.mode_caps["max_position_dollars"] <= 500.0

    @pytest.mark.asyncio
    async def test_paper_mode_no_limits(
        self, mock_broker: PaperBroker, valid_signal: Signal
    ) -> None:
        """Test that PAPER mode has no practical limits."""
        mode_manager = MagicMock()
        mode_manager.current_mode = TradingMode.PAPER
        mode_manager.config = MagicMock()
        mode_manager.config.max_position_dollars = 10000.0
        mode_manager.config.max_daily_loss_dollars = 10000.0

        engine = RiskEngine(
            broker=mock_broker,
            db=None,
            mode_manager=mode_manager,
        )

        decision = await engine.evaluate_signal(valid_signal)

        assert decision.mode_caps["max_position_dollars"] == 10000.0


class TestRiskEngineStatus:
    """Tests for risk engine status reporting."""

    def test_get_status(self, risk_engine: RiskEngine) -> None:
        """Test get_status returns expected fields."""
        status = risk_engine.get_status()

        assert "circuit_breakers" in status
        assert "limits" in status
        assert "balance" in status
        assert "mode" in status
        assert "broker_is_paper" in status

        assert status["mode"] == "paper"
        assert status["broker_is_paper"] is True
        assert status["limits"]["max_position_contracts"] == 100
        assert status["limits"]["kelly_fraction"] == 0.25
