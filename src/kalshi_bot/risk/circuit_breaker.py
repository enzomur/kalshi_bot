"""Circuit breakers for risk management."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from kalshi_bot.config.settings import Settings
from kalshi_bot.core.exceptions import CircuitBreakerError
from kalshi_bot.persistence.models import CircuitBreakerRepository
from kalshi_bot.utils.logging import get_logger
from kalshi_bot.utils.notifications import send_notification

logger = get_logger(__name__)


class BreakerType(str, Enum):
    """Types of circuit breakers."""

    DRAWDOWN = "drawdown"
    CONSECUTIVE_FAILURES = "consecutive_failures"
    SUCCESS_RATE = "success_rate"
    DAILY_LOSS = "daily_loss"
    ERROR_RATE = "error_rate"
    PORTFOLIO_STOP = "portfolio_stop"  # Hard stop at 50% loss - requires manual reset


class CircuitBreakerState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Halted, no trading
    HALF_OPEN = "half_open"  # Testing if it's safe to resume


@dataclass
class BreakerStatus:
    """Status of a single circuit breaker."""

    breaker_type: BreakerType
    state: CircuitBreakerState
    current_value: float
    threshold: float
    triggered_at: datetime | None = None
    cooldown_until: datetime | None = None
    trigger_count: int = 0


@dataclass
class TradeResult:
    """Result of a trade for tracking."""

    success: bool
    profit_loss: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


class CircuitBreaker:
    """
    Implements multiple circuit breakers for risk management.

    Circuit breakers halt trading when risk thresholds are exceeded:

    1. Drawdown Breaker: Triggers when portfolio drops X% from peak
    2. Consecutive Failures: Triggers after N failed trades in a row
    3. Success Rate: Triggers when win rate drops below threshold
    4. Daily Loss: Triggers when daily loss exceeds limit
    5. Error Rate: Triggers on high API/system error rate

    Each breaker can be in one of three states:
    - CLOSED: Normal operation
    - OPEN: Trading halted
    - HALF_OPEN: Testing if safe to resume
    """

    def __init__(
        self,
        settings: Settings,
        breaker_repo: CircuitBreakerRepository | None = None,
    ) -> None:
        """
        Initialize circuit breakers.

        Args:
            settings: Application settings
            breaker_repo: Repository for persisting breaker events
        """
        self.settings = settings
        self.breaker_repo = breaker_repo

        self._max_drawdown = settings.risk.max_drawdown
        self._max_consecutive_failures = settings.risk.max_consecutive_failures
        self._min_success_rate = settings.risk.min_success_rate
        self._success_rate_window = settings.risk.success_rate_window
        self._max_daily_loss = settings.risk.max_daily_loss
        self._cooldown_seconds = getattr(
            settings.risk, 'circuit_breaker_cooldown', 1800
        )  # Default 30 minutes

        # New settings for enhanced risk controls
        self._reset_success_count = getattr(
            settings.risk, 'circuit_breaker_reset_count', 3
        )  # Require 3 consecutive successes to reset
        self._portfolio_stop_threshold = getattr(
            settings.risk, 'portfolio_stop_threshold', 0.50
        )  # 50% loss = hard stop

        self._breaker_states: dict[BreakerType, CircuitBreakerState] = {
            bt: CircuitBreakerState.CLOSED for bt in BreakerType
        }
        self._triggered_at: dict[BreakerType, datetime | None] = {
            bt: None for bt in BreakerType
        }
        self._cooldown_until: dict[BreakerType, datetime | None] = {
            bt: None for bt in BreakerType
        }
        self._trigger_counts: dict[BreakerType, int] = {bt: 0 for bt in BreakerType}

        # Track consecutive successes for half-open breaker reset
        self._half_open_successes: dict[BreakerType, int] = {
            bt: 0 for bt in BreakerType
        }

        self._peak_value: float = 0.0
        self._initial_value: float = 0.0  # Track initial portfolio value
        self._consecutive_failures: int = 0
        self._trade_history: deque[TradeResult] = deque(
            maxlen=self._success_rate_window
        )
        self._daily_pnl: float = 0.0
        self._daily_reset_date: str = datetime.utcnow().strftime("%Y-%m-%d")
        self._error_count: int = 0
        self._request_count: int = 0

        self._lock = asyncio.Lock()

    async def check_all(self) -> None:
        """
        Check all circuit breakers and raise if any are triggered.

        Raises:
            CircuitBreakerError: If any breaker is in OPEN state
        """
        async with self._lock:
            now = datetime.utcnow()

            for breaker_type in BreakerType:
                state = self._breaker_states[breaker_type]
                cooldown = self._cooldown_until.get(breaker_type)

                if state == CircuitBreakerState.OPEN:
                    # PORTFOLIO_STOP has no cooldown - always stays open until manual reset
                    if breaker_type == BreakerType.PORTFOLIO_STOP:
                        raise CircuitBreakerError(
                            breaker_type.value,
                            f"PORTFOLIO HARD STOP is OPEN - requires manual reset",
                            cooldown_remaining=-1,  # -1 indicates no auto-recovery
                        )
                    elif cooldown and now >= cooldown:
                        self._breaker_states[breaker_type] = CircuitBreakerState.HALF_OPEN
                        self._half_open_successes[breaker_type] = 0  # Reset counter
                        logger.info(f"{breaker_type.value} breaker entering half-open state")
                    else:
                        remaining = (cooldown - now).total_seconds() if cooldown else 0
                        raise CircuitBreakerError(
                            breaker_type.value,
                            f"Circuit breaker {breaker_type.value} is OPEN",
                            cooldown_remaining=remaining,
                        )

    async def record_trade(self, result: TradeResult) -> None:
        """
        Record a trade result and update breaker states.

        Args:
            result: The trade result to record
        """
        async with self._lock:
            self._trade_history.append(result)

            if result.success:
                self._consecutive_failures = 0
                await self._check_half_open_success(BreakerType.CONSECUTIVE_FAILURES)
            else:
                self._consecutive_failures += 1

            self._daily_pnl += result.profit_loss

            await self._check_consecutive_failures()
            await self._check_success_rate()
            await self._check_daily_loss()

    async def record_portfolio_value(self, current_value: float) -> None:
        """
        Record portfolio value and check drawdown.

        Args:
            current_value: Current portfolio value
        """
        async with self._lock:
            if current_value > self._peak_value:
                self._peak_value = current_value

            await self._check_drawdown(current_value)
            await self._check_portfolio_stop(current_value)

    async def record_error(self) -> None:
        """Record an API/system error."""
        async with self._lock:
            self._error_count += 1
            self._request_count += 1
            await self._check_error_rate()

    async def record_success(self) -> None:
        """Record a successful API request."""
        async with self._lock:
            self._request_count += 1

    async def reset_daily(self) -> None:
        """Reset daily counters (call at midnight)."""
        async with self._lock:
            today = datetime.utcnow().strftime("%Y-%m-%d")
            if today != self._daily_reset_date:
                self._daily_pnl = 0.0
                self._daily_reset_date = today
                self._error_count = 0
                self._request_count = 0

                if self._breaker_states[BreakerType.DAILY_LOSS] == CircuitBreakerState.OPEN:
                    await self._reset_breaker(BreakerType.DAILY_LOSS)

                logger.info("Daily circuit breaker counters reset")

    async def _check_drawdown(self, current_value: float) -> None:
        """Check drawdown breaker."""
        if self._peak_value <= 0:
            return

        drawdown = (self._peak_value - current_value) / self._peak_value

        if drawdown >= self._max_drawdown:
            await self._trigger_breaker(
                BreakerType.DRAWDOWN,
                f"Drawdown {drawdown:.2%} exceeds maximum {self._max_drawdown:.2%}",
                drawdown,
                self._max_drawdown,
            )

    async def _check_consecutive_failures(self) -> None:
        """Check consecutive failures breaker."""
        if self._consecutive_failures >= self._max_consecutive_failures:
            await self._trigger_breaker(
                BreakerType.CONSECUTIVE_FAILURES,
                f"{self._consecutive_failures} consecutive failures",
                float(self._consecutive_failures),
                float(self._max_consecutive_failures),
            )

    async def _check_success_rate(self) -> None:
        """Check success rate breaker."""
        if len(self._trade_history) < self._success_rate_window:
            return

        successes = sum(1 for t in self._trade_history if t.success)
        success_rate = successes / len(self._trade_history)

        if success_rate < self._min_success_rate:
            await self._trigger_breaker(
                BreakerType.SUCCESS_RATE,
                f"Success rate {success_rate:.2%} below minimum {self._min_success_rate:.2%}",
                success_rate,
                self._min_success_rate,
            )

    async def _check_daily_loss(self) -> None:
        """Check daily loss breaker."""
        if self._daily_pnl < -self._max_daily_loss:
            await self._trigger_breaker(
                BreakerType.DAILY_LOSS,
                f"Daily loss ${abs(self._daily_pnl):.2f} exceeds maximum ${self._max_daily_loss:.2f}",
                abs(self._daily_pnl),
                self._max_daily_loss,
            )

    async def _check_error_rate(self) -> None:
        """Check error rate breaker."""
        if self._request_count < 10:
            return

        error_rate = self._error_count / self._request_count
        max_error_rate = 0.20

        if error_rate >= max_error_rate:
            await self._trigger_breaker(
                BreakerType.ERROR_RATE,
                f"Error rate {error_rate:.2%} exceeds maximum {max_error_rate:.2%}",
                error_rate,
                max_error_rate,
            )

    async def _check_portfolio_stop(self, current_value: float) -> None:
        """
        Check portfolio hard stop - triggers at 50% total loss.

        This breaker requires MANUAL reset - no automatic cooldown.
        """
        if self._initial_value <= 0:
            return

        loss_pct = (self._initial_value - current_value) / self._initial_value

        if loss_pct >= self._portfolio_stop_threshold:
            await self._trigger_portfolio_stop(
                f"Portfolio loss {loss_pct:.2%} exceeds hard stop threshold "
                f"{self._portfolio_stop_threshold:.2%}",
                loss_pct,
                self._portfolio_stop_threshold,
            )

    async def _trigger_portfolio_stop(
        self,
        reason: str,
        trigger_value: float,
        threshold_value: float,
    ) -> None:
        """
        Trigger the portfolio hard stop breaker.

        Unlike other breakers, this has NO automatic cooldown.
        It requires manual reset.
        """
        if self._breaker_states[BreakerType.PORTFOLIO_STOP] == CircuitBreakerState.OPEN:
            return

        now = datetime.utcnow()

        # NO cooldown - this stays open until manually reset
        self._breaker_states[BreakerType.PORTFOLIO_STOP] = CircuitBreakerState.OPEN
        self._triggered_at[BreakerType.PORTFOLIO_STOP] = now
        self._cooldown_until[BreakerType.PORTFOLIO_STOP] = None  # No auto-cooldown
        self._trigger_counts[BreakerType.PORTFOLIO_STOP] += 1

        logger.critical(
            f"PORTFOLIO HARD STOP TRIGGERED: {reason}. "
            "Manual reset required to resume trading."
        )

        send_notification(
            "CRITICAL: Portfolio Hard Stop",
            f"Trading halted - {reason}. Manual intervention required.",
        )

        if self.breaker_repo:
            try:
                await self.breaker_repo.log_trigger(
                    BreakerType.PORTFOLIO_STOP.value,
                    reason,
                    trigger_value,
                    threshold_value,
                    None,  # No cooldown
                )
            except Exception as e:
                logger.error(f"Failed to log portfolio stop trigger: {e}")

    async def _trigger_breaker(
        self,
        breaker_type: BreakerType,
        reason: str,
        trigger_value: float,
        threshold_value: float,
    ) -> None:
        """Trigger a circuit breaker."""
        if self._breaker_states[breaker_type] == CircuitBreakerState.OPEN:
            return

        now = datetime.utcnow()
        cooldown_until = now + timedelta(seconds=self._cooldown_seconds)

        self._breaker_states[breaker_type] = CircuitBreakerState.OPEN
        self._triggered_at[breaker_type] = now
        self._cooldown_until[breaker_type] = cooldown_until
        self._trigger_counts[breaker_type] += 1

        logger.warning(
            f"Circuit breaker TRIGGERED: {breaker_type.value} - {reason}"
        )

        send_notification(
            "Circuit Breaker Triggered",
            f"{breaker_type.value.upper()}: {reason}",
        )

        if self.breaker_repo:
            try:
                await self.breaker_repo.log_trigger(
                    breaker_type.value,
                    reason,
                    trigger_value,
                    threshold_value,
                    cooldown_until,
                )
            except Exception as e:
                logger.error(f"Failed to log breaker trigger: {e}")

    async def _reset_breaker(self, breaker_type: BreakerType) -> None:
        """Reset a circuit breaker to closed state."""
        self._breaker_states[breaker_type] = CircuitBreakerState.CLOSED
        self._cooldown_until[breaker_type] = None

        logger.info(f"Circuit breaker RESET: {breaker_type.value}")

        if self.breaker_repo:
            try:
                await self.breaker_repo.log_reset(breaker_type.value)
            except Exception as e:
                logger.error(f"Failed to log breaker reset: {e}")

    async def _check_half_open_success(self, breaker_type: BreakerType) -> None:
        """
        Check if a half-open breaker should close.

        Requires consecutive successes (default 3) to fully reset.
        """
        if self._breaker_states[breaker_type] == CircuitBreakerState.HALF_OPEN:
            self._half_open_successes[breaker_type] += 1

            if self._half_open_successes[breaker_type] >= self._reset_success_count:
                await self._reset_breaker(breaker_type)
                self._half_open_successes[breaker_type] = 0
                logger.info(
                    f"Breaker {breaker_type.value} reset after "
                    f"{self._reset_success_count} consecutive successes"
                )
            else:
                logger.info(
                    f"Breaker {breaker_type.value} half-open: "
                    f"{self._half_open_successes[breaker_type]}/{self._reset_success_count} "
                    f"successes toward reset"
                )

    async def manual_reset(self, breaker_type: BreakerType | None = None) -> None:
        """
        Manually reset circuit breaker(s).

        Args:
            breaker_type: Specific breaker to reset, or None for all
        """
        async with self._lock:
            if breaker_type:
                await self._reset_breaker(breaker_type)
            else:
                for bt in BreakerType:
                    if self._breaker_states[bt] != CircuitBreakerState.CLOSED:
                        await self._reset_breaker(bt)

    def set_peak_value(self, value: float) -> None:
        """Set peak portfolio value (for initialization)."""
        self._peak_value = value

    def set_initial_value(self, value: float) -> None:
        """Set initial portfolio value (for hard stop calculation)."""
        self._initial_value = value
        if self._peak_value <= 0:
            self._peak_value = value

    def get_status(self, breaker_type: BreakerType) -> BreakerStatus:
        """Get status of a specific breaker."""
        now = datetime.utcnow()

        if breaker_type == BreakerType.DRAWDOWN:
            current_value = (
                (self._peak_value - self._peak_value) / self._peak_value
                if self._peak_value > 0
                else 0.0
            )
            threshold = self._max_drawdown
        elif breaker_type == BreakerType.CONSECUTIVE_FAILURES:
            current_value = float(self._consecutive_failures)
            threshold = float(self._max_consecutive_failures)
        elif breaker_type == BreakerType.SUCCESS_RATE:
            if len(self._trade_history) > 0:
                successes = sum(1 for t in self._trade_history if t.success)
                current_value = successes / len(self._trade_history)
            else:
                current_value = 1.0
            threshold = self._min_success_rate
        elif breaker_type == BreakerType.DAILY_LOSS:
            current_value = abs(min(0, self._daily_pnl))
            threshold = self._max_daily_loss
        elif breaker_type == BreakerType.ERROR_RATE:
            if self._request_count > 0:
                current_value = self._error_count / self._request_count
            else:
                current_value = 0.0
            threshold = 0.20
        elif breaker_type == BreakerType.PORTFOLIO_STOP:
            if self._initial_value > 0:
                current_value = max(0, (self._initial_value - self._peak_value) / self._initial_value)
            else:
                current_value = 0.0
            threshold = self._portfolio_stop_threshold
        else:
            current_value = 0.0
            threshold = 0.0

        return BreakerStatus(
            breaker_type=breaker_type,
            state=self._breaker_states[breaker_type],
            current_value=current_value,
            threshold=threshold,
            triggered_at=self._triggered_at[breaker_type],
            cooldown_until=self._cooldown_until[breaker_type],
            trigger_count=self._trigger_counts[breaker_type],
        )

    def get_all_status(self) -> dict[str, Any]:
        """Get status of all breakers."""
        statuses = {}
        for bt in BreakerType:
            status = self.get_status(bt)
            statuses[bt.value] = {
                "state": status.state.value,
                "current_value": status.current_value,
                "threshold": status.threshold,
                "triggered_at": status.triggered_at.isoformat() if status.triggered_at else None,
                "cooldown_until": status.cooldown_until.isoformat() if status.cooldown_until else None,
                "trigger_count": status.trigger_count,
            }

        return {
            "breakers": statuses,
            "any_triggered": any(
                s.state != CircuitBreakerState.CLOSED
                for s in [self.get_status(bt) for bt in BreakerType]
            ),
            "peak_value": self._peak_value,
            "initial_value": self._initial_value,
            "daily_pnl": self._daily_pnl,
            "consecutive_failures": self._consecutive_failures,
            "trade_history_size": len(self._trade_history),
            "cooldown_seconds": self._cooldown_seconds,
            "reset_success_count": self._reset_success_count,
            "portfolio_stop_threshold": self._portfolio_stop_threshold,
        }
