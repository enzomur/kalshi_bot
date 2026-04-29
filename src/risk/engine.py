"""Risk Engine - THE GATEKEEPER to execution.

The Risk Engine is the ONLY component that can authorize trade execution.
Strategies emit Signals, but only the Risk Engine can convert them to Orders.

Responsibilities:
1. Check circuit breakers (daily loss, drawdown)
2. Check position limits (per-market, category, total)
3. Calculate Kelly-optimal position size
4. Apply mode-specific caps (PROBATION limits)
5. Log all decisions to the ledger
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from src.core.types import Signal, Side, TradingMode
from src.core.mode import ModeManager, get_mode_manager
from src.core.exceptions import CircuitBreakerError, RiskError
from src.execution.base import BaseBroker, ExecutionResult
from src.execution.paper_broker import PaperBroker
from src.ledger.database import Database
from src.risk.kelly import calculate_kelly_size, KellyResult
from src.observability.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CircuitBreakerState:
    """State of circuit breakers."""

    daily_loss_triggered: bool = False
    drawdown_triggered: bool = False
    consecutive_loss_triggered: bool = False

    daily_loss_amount: float = 0.0
    current_drawdown: float = 0.0
    consecutive_losses: int = 0

    cooldown_until: datetime | None = None

    def is_triggered(self) -> bool:
        """Check if any circuit breaker is triggered."""
        if self.cooldown_until and datetime.now(timezone.utc) < self.cooldown_until:
            return True
        return (
            self.daily_loss_triggered
            or self.drawdown_triggered
            or self.consecutive_loss_triggered
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "daily_loss_triggered": self.daily_loss_triggered,
            "drawdown_triggered": self.drawdown_triggered,
            "consecutive_loss_triggered": self.consecutive_loss_triggered,
            "daily_loss_amount": self.daily_loss_amount,
            "current_drawdown": self.current_drawdown,
            "consecutive_losses": self.consecutive_losses,
            "cooldown_until": (
                self.cooldown_until.isoformat() if self.cooldown_until else None
            ),
        }


@dataclass
class PositionCheck:
    """Result of position limit checks."""

    approved: bool
    current_position: int
    max_allowed: int
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "approved": self.approved,
            "current_position": self.current_position,
            "max_allowed": self.max_allowed,
            "reason": self.reason,
        }


@dataclass
class RiskDecision:
    """Result of risk evaluation for a signal."""

    decision_id: str
    signal_id: str
    approved: bool
    approved_size: int
    rejection_reason: str | None

    circuit_breaker_status: CircuitBreakerState
    position_check: PositionCheck
    kelly_result: KellyResult
    mode_caps: dict[str, Any]

    trading_mode: TradingMode
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "signal_id": self.signal_id,
            "approved": self.approved,
            "approved_size": self.approved_size,
            "rejection_reason": self.rejection_reason,
            "circuit_breaker_status": self.circuit_breaker_status.to_dict(),
            "position_check": self.position_check.to_dict(),
            "kelly_result": {
                "full_kelly_fraction": self.kelly_result.full_kelly_fraction,
                "position_fraction": self.kelly_result.position_fraction,
                "position_contracts": self.kelly_result.position_contracts,
                "edge": self.kelly_result.edge,
                "expected_value": self.kelly_result.expected_value,
                "is_positive_ev": self.kelly_result.is_positive_ev,
            },
            "mode_caps": self.mode_caps,
            "trading_mode": self.trading_mode.value,
            "created_at": self.created_at.isoformat(),
        }


class RiskEngine:
    """The Gatekeeper - evaluates signals and controls execution.

    ENFORCEMENT: Only RiskEngine may call broker.execute().
    Strategies emit Signals -> RiskEngine evaluates -> Broker executes.
    """

    # Default limits
    DEFAULT_MAX_POSITION_CONTRACTS = 100
    DEFAULT_MAX_DAILY_LOSS = 100.0  # $100
    DEFAULT_MAX_DRAWDOWN = 0.20  # 20%
    DEFAULT_MAX_CONSECUTIVE_LOSSES = 5
    DEFAULT_KELLY_FRACTION = 0.25

    # Mode-specific limits
    PROBATION_MAX_POSITION_DOLLARS = 500.0
    PROBATION_MAX_DAILY_LOSS = 100.0

    def __init__(
        self,
        broker: BaseBroker,
        db: Database | None = None,
        mode_manager: ModeManager | None = None,
        kelly_fraction: float = DEFAULT_KELLY_FRACTION,
        max_position_contracts: int = DEFAULT_MAX_POSITION_CONTRACTS,
        max_daily_loss: float = DEFAULT_MAX_DAILY_LOSS,
        max_drawdown: float = DEFAULT_MAX_DRAWDOWN,
        max_consecutive_losses: int = DEFAULT_MAX_CONSECUTIVE_LOSSES,
    ) -> None:
        """
        Initialize Risk Engine.

        Args:
            broker: Execution broker (paper or live)
            db: Database for persistence
            mode_manager: Mode manager for mode-specific limits
            kelly_fraction: Fraction of Kelly to use
            max_position_contracts: Maximum contracts per position
            max_daily_loss: Maximum daily loss before circuit breaker
            max_drawdown: Maximum drawdown before circuit breaker
            max_consecutive_losses: Max consecutive losses before circuit breaker
        """
        self._broker = broker
        self._db = db
        self._mode_manager = mode_manager or get_mode_manager()

        self._kelly_fraction = kelly_fraction
        self._max_position_contracts = max_position_contracts
        self._max_daily_loss = max_daily_loss
        self._max_drawdown = max_drawdown
        self._max_consecutive_losses = max_consecutive_losses

        self._circuit_breakers = CircuitBreakerState()
        self._peak_balance: float | None = None
        self._initial_balance: float | None = None

    async def evaluate_signal(self, signal: Signal) -> RiskDecision:
        """
        Evaluate a trading signal and decide whether to approve execution.

        This is THE GATEKEEPER function. All trading decisions flow through here.

        Args:
            signal: The trading signal to evaluate

        Returns:
            RiskDecision with approval status and sizing
        """
        decision_id = str(uuid.uuid4())
        mode = self._mode_manager.current_mode
        mode_config = self._mode_manager.config

        logger.info(
            f"Evaluating signal {signal.signal_id[:8]} from {signal.strategy_name}: "
            f"{signal.direction} {signal.market_ticker} "
            f"(edge={signal.edge:.2%}, confidence={signal.confidence:.2%})"
        )

        # 1. Check if signal is expired
        if signal.is_expired:
            return self._reject(
                decision_id,
                signal,
                "Signal expired",
                mode,
            )

        # 2. Check circuit breakers
        if self._circuit_breakers.is_triggered():
            return self._reject(
                decision_id,
                signal,
                f"Circuit breaker triggered: {self._get_breaker_reason()}",
                mode,
            )

        # 3. Check position limits
        position_check = await self._check_position_limits(signal)
        if not position_check.approved:
            return self._reject(
                decision_id,
                signal,
                f"Position limit exceeded: {position_check.reason}",
                mode,
                position_check=position_check,
            )

        # 4. Calculate Kelly-optimal size
        balance = self._broker.balance
        if self._initial_balance is None:
            self._initial_balance = balance
        if self._peak_balance is None or balance > self._peak_balance:
            self._peak_balance = balance

        # Get market price from signal metadata or use a default
        market_price = signal.metadata.get("market_price_cents", 50)

        kelly_result = calculate_kelly_size(
            model_probability=signal.target_probability,
            market_price_cents=market_price,
            direction=signal.direction,
            bankroll=balance,
            kelly_fraction=self._kelly_fraction * signal.confidence,  # Scale by confidence
            max_position_contracts=min(
                self._max_position_contracts,
                signal.max_position,
                position_check.max_allowed - position_check.current_position,
            ),
        )

        if not kelly_result.is_positive_ev:
            return self._reject(
                decision_id,
                signal,
                f"Negative expected value (EV={kelly_result.expected_value:.4f})",
                mode,
                position_check=position_check,
                kelly_result=kelly_result,
            )

        # 5. Apply mode-specific caps
        mode_caps = self._get_mode_caps(mode, mode_config)
        approved_size = kelly_result.position_contracts

        # Cap by mode limits
        if mode == TradingMode.LIVE_PROBATION:
            max_dollars = mode_caps.get("max_position_dollars", self.PROBATION_MAX_POSITION_DOLLARS)
            max_contracts_by_dollars = int(max_dollars / (market_price / 100))
            approved_size = min(approved_size, max_contracts_by_dollars)

        # Ensure at least 1 contract if approved
        if approved_size <= 0:
            return self._reject(
                decision_id,
                signal,
                "Position size rounds to zero",
                mode,
                position_check=position_check,
                kelly_result=kelly_result,
                mode_caps=mode_caps,
            )

        # 6. Approved - create decision
        decision = RiskDecision(
            decision_id=decision_id,
            signal_id=signal.signal_id,
            approved=True,
            approved_size=approved_size,
            rejection_reason=None,
            circuit_breaker_status=self._circuit_breakers,
            position_check=position_check,
            kelly_result=kelly_result,
            mode_caps=mode_caps,
            trading_mode=mode,
        )

        # Log to database
        await self._log_decision(decision, signal)

        logger.info(
            f"APPROVED signal {signal.signal_id[:8]}: "
            f"{approved_size} contracts (Kelly={kelly_result.full_kelly_fraction:.2%})"
        )

        return decision

    async def execute_decision(self, decision: RiskDecision, signal: Signal) -> ExecutionResult:
        """
        Execute an approved risk decision.

        Args:
            decision: Approved RiskDecision
            signal: Original signal

        Returns:
            ExecutionResult from the broker
        """
        if not decision.approved:
            raise RiskError(
                "Cannot execute rejected decision",
                signal_id=signal.signal_id,
                reason=decision.rejection_reason,
            )

        market_price = signal.metadata.get("market_price_cents", 50)
        side = Side.YES if signal.direction == "yes" else Side.NO

        result = await self._broker.execute(
            market_ticker=signal.market_ticker,
            side=side,
            quantity=decision.approved_size,
            price=market_price,
            decision_id=decision.decision_id,
        )

        # Update circuit breakers based on result
        if result.success:
            # Will update P&L tracking when position settles
            pass
        else:
            # Failed execution might indicate issues
            logger.warning(f"Execution failed: {result.error_message}")

        return result

    def _reject(
        self,
        decision_id: str,
        signal: Signal,
        reason: str,
        mode: TradingMode,
        position_check: PositionCheck | None = None,
        kelly_result: KellyResult | None = None,
        mode_caps: dict[str, Any] | None = None,
    ) -> RiskDecision:
        """Create a rejection decision."""
        if position_check is None:
            position_check = PositionCheck(
                approved=False,
                current_position=0,
                max_allowed=0,
                reason=reason,
            )

        if kelly_result is None:
            kelly_result = KellyResult(
                full_kelly_fraction=0.0,
                position_fraction=0.0,
                position_dollars=0.0,
                position_contracts=0,
                edge=signal.edge,
                expected_value=0.0,
                is_positive_ev=False,
            )

        decision = RiskDecision(
            decision_id=decision_id,
            signal_id=signal.signal_id,
            approved=False,
            approved_size=0,
            rejection_reason=reason,
            circuit_breaker_status=self._circuit_breakers,
            position_check=position_check,
            kelly_result=kelly_result,
            mode_caps=mode_caps or {},
            trading_mode=mode,
        )

        logger.info(f"REJECTED signal {signal.signal_id[:8]}: {reason}")

        return decision

    async def _check_position_limits(self, signal: Signal) -> PositionCheck:
        """Check if signal would exceed position limits."""
        side = Side.YES if signal.direction == "yes" else Side.NO
        position = await self._broker.get_position(signal.market_ticker, side)

        current_quantity = position.quantity if position else 0

        # Check if we're already at max
        if current_quantity >= self._max_position_contracts:
            return PositionCheck(
                approved=False,
                current_position=current_quantity,
                max_allowed=self._max_position_contracts,
                reason=f"Already at max position ({current_quantity} contracts)",
            )

        # Check if adding would exceed max
        max_additional = self._max_position_contracts - current_quantity

        return PositionCheck(
            approved=True,
            current_position=current_quantity,
            max_allowed=self._max_position_contracts,
        )

    def _get_mode_caps(
        self, mode: TradingMode, mode_config: Any
    ) -> dict[str, Any]:
        """Get mode-specific position caps."""
        if mode == TradingMode.PAPER:
            return {
                "max_position_dollars": 10000.0,  # Effectively unlimited
                "max_daily_loss": 10000.0,
            }
        elif mode == TradingMode.LIVE_PROBATION:
            return {
                "max_position_dollars": min(
                    mode_config.max_position_dollars,
                    self.PROBATION_MAX_POSITION_DOLLARS,
                ),
                "max_daily_loss": min(
                    mode_config.max_daily_loss_dollars,
                    self.PROBATION_MAX_DAILY_LOSS,
                ),
            }
        else:  # LIVE_FULL
            return {
                "max_position_dollars": mode_config.max_position_dollars,
                "max_daily_loss": mode_config.max_daily_loss_dollars,
            }

    def _get_breaker_reason(self) -> str:
        """Get human-readable reason for circuit breaker trigger."""
        reasons = []
        if self._circuit_breakers.daily_loss_triggered:
            reasons.append(
                f"daily loss (${self._circuit_breakers.daily_loss_amount:.2f})"
            )
        if self._circuit_breakers.drawdown_triggered:
            reasons.append(
                f"drawdown ({self._circuit_breakers.current_drawdown:.1%})"
            )
        if self._circuit_breakers.consecutive_loss_triggered:
            reasons.append(
                f"consecutive losses ({self._circuit_breakers.consecutive_losses})"
            )
        if self._circuit_breakers.cooldown_until:
            reasons.append(
                f"cooldown until {self._circuit_breakers.cooldown_until.isoformat()}"
            )
        return ", ".join(reasons) if reasons else "unknown"

    async def _log_decision(self, decision: RiskDecision, signal: Signal) -> None:
        """Log decision to database."""
        if self._db is None:
            return

        try:
            await self._db.save_signal(signal)
            await self._db.save_risk_decision(
                decision_id=decision.decision_id,
                signal_id=signal.signal_id,
                approved=decision.approved,
                approved_size=decision.approved_size,
                rejection_reason=decision.rejection_reason,
                circuit_breaker_status=decision.circuit_breaker_status.to_dict(),
                position_check=decision.position_check.to_dict(),
                kelly_calculation={
                    "full_kelly": decision.kelly_result.full_kelly_fraction,
                    "position_fraction": decision.kelly_result.position_fraction,
                    "position_contracts": decision.kelly_result.position_contracts,
                    "edge": decision.kelly_result.edge,
                    "ev": decision.kelly_result.expected_value,
                },
                mode_caps=decision.mode_caps,
                trading_mode=decision.trading_mode.value,
            )
        except Exception as e:
            logger.warning(f"Failed to log decision: {e}")

    def update_daily_loss(self, loss_amount: float) -> None:
        """Update daily loss tracker and check circuit breaker."""
        self._circuit_breakers.daily_loss_amount += loss_amount

        if self._circuit_breakers.daily_loss_amount >= self._max_daily_loss:
            self._circuit_breakers.daily_loss_triggered = True
            logger.warning(
                f"Daily loss circuit breaker triggered: "
                f"${self._circuit_breakers.daily_loss_amount:.2f}"
            )

    def record_trade_result(self, won: bool) -> None:
        """Record a trade result for consecutive loss tracking."""
        if won:
            self._circuit_breakers.consecutive_losses = 0
        else:
            self._circuit_breakers.consecutive_losses += 1

            if self._circuit_breakers.consecutive_losses >= self._max_consecutive_losses:
                self._circuit_breakers.consecutive_loss_triggered = True
                logger.warning(
                    f"Consecutive loss circuit breaker triggered: "
                    f"{self._circuit_breakers.consecutive_losses} losses"
                )

    def update_drawdown(self) -> None:
        """Update drawdown calculation and check circuit breaker."""
        if self._peak_balance is None or self._peak_balance <= 0:
            return

        current_balance = self._broker.balance
        if current_balance > self._peak_balance:
            self._peak_balance = current_balance
            return

        drawdown = (self._peak_balance - current_balance) / self._peak_balance
        self._circuit_breakers.current_drawdown = drawdown

        if drawdown >= self._max_drawdown:
            self._circuit_breakers.drawdown_triggered = True
            logger.warning(
                f"Drawdown circuit breaker triggered: {drawdown:.1%}"
            )

    def reset_daily_breakers(self) -> None:
        """Reset daily circuit breakers (call at start of each trading day)."""
        self._circuit_breakers.daily_loss_triggered = False
        self._circuit_breakers.daily_loss_amount = 0.0
        logger.info("Daily circuit breakers reset")

    def get_status(self) -> dict[str, Any]:
        """Get risk engine status."""
        return {
            "circuit_breakers": self._circuit_breakers.to_dict(),
            "limits": {
                "max_position_contracts": self._max_position_contracts,
                "max_daily_loss": self._max_daily_loss,
                "max_drawdown": self._max_drawdown,
                "max_consecutive_losses": self._max_consecutive_losses,
                "kelly_fraction": self._kelly_fraction,
            },
            "balance": {
                "current": self._broker.balance,
                "initial": self._initial_balance,
                "peak": self._peak_balance,
            },
            "mode": self._mode_manager.current_mode.value,
            "broker_is_paper": self._broker.is_paper,
        }
