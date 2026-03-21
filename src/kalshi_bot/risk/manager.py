"""Risk manager coordinating all risk controls."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from kalshi_bot.config.settings import Settings
from kalshi_bot.core.exceptions import (
    CircuitBreakerError,
    InsufficientBalanceError,
    PositionLimitError,
)
from kalshi_bot.core.types import ArbitrageOpportunity, Position
from kalshi_bot.persistence.models import AuditRepository
from kalshi_bot.risk.circuit_breaker import CircuitBreaker, TradeResult
from kalshi_bot.risk.limits import PositionLimits
from kalshi_bot.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RiskCheckResult:
    """Result of risk evaluation."""

    approved: bool
    reason: str | None = None
    max_quantity: int | None = None
    adjustments: dict[str, Any] | None = None


class RiskManager:
    """
    Central risk management coordinator.

    Integrates:
    - Circuit breakers for automatic halt
    - Position limits for exposure control
    - Pre-trade risk checks
    - Post-trade analysis

    All trades must pass through the risk manager before execution.
    """

    def __init__(
        self,
        settings: Settings,
        circuit_breaker: CircuitBreaker,
        position_limits: PositionLimits,
        audit_repo: AuditRepository | None = None,
    ) -> None:
        """
        Initialize risk manager.

        Args:
            settings: Application settings
            circuit_breaker: Circuit breaker instance
            position_limits: Position limits instance
            audit_repo: Audit repository for logging
        """
        self.settings = settings
        self.circuit_breaker = circuit_breaker
        self.position_limits = position_limits
        self.audit_repo = audit_repo

        self._min_balance = 2.0
        self._emergency_stop = False
        self._lock = asyncio.Lock()

    async def pre_trade_check(
        self,
        opportunity: ArbitrageOpportunity,
        positions: list[Position],
        available_balance: float,
        portfolio_value: float,
    ) -> RiskCheckResult:
        """
        Perform comprehensive pre-trade risk check.

        Args:
            opportunity: Opportunity to evaluate
            positions: Current positions
            available_balance: Available cash balance
            portfolio_value: Total portfolio value

        Returns:
            RiskCheckResult indicating if trade is approved
        """
        async with self._lock:
            if self._emergency_stop:
                return RiskCheckResult(
                    approved=False,
                    reason="Emergency stop active",
                )

            try:
                await self.circuit_breaker.check_all()
            except CircuitBreakerError as e:
                await self._log_risk_event(
                    "trade_blocked_circuit_breaker",
                    {"opportunity_id": opportunity.opportunity_id, "breaker": e.breaker_type},
                    "warning",
                )
                return RiskCheckResult(
                    approved=False,
                    reason=f"Circuit breaker active: {e.breaker_type}",
                )

            if available_balance < self._min_balance:
                return RiskCheckResult(
                    approved=False,
                    reason=f"Balance ${available_balance:.2f} below minimum ${self._min_balance:.2f}",
                )

            if opportunity.total_cost > available_balance:
                max_affordable = int(
                    available_balance / (opportunity.total_cost / opportunity.max_quantity)
                )
                if max_affordable < 1:
                    return RiskCheckResult(
                        approved=False,
                        reason=f"Insufficient balance: need ${opportunity.total_cost:.2f}, have ${available_balance:.2f}",
                    )
                return RiskCheckResult(
                    approved=True,
                    max_quantity=max_affordable,
                    adjustments={"reduced_for_balance": True},
                )

            primary_market = opportunity.markets[0]
            cost_per_contract = opportunity.total_cost / opportunity.max_quantity

            limit_checks = self.position_limits.check_all(
                primary_market,
                opportunity.max_quantity,
                opportunity.total_cost,
                positions,
                portfolio_value,
            )

            failed_checks = [c for c in limit_checks if not c.passed]
            if failed_checks:
                max_allowed = self.position_limits.get_max_allowed_quantity(
                    primary_market,
                    cost_per_contract,
                    positions,
                    portfolio_value,
                )

                if max_allowed < 1:
                    return RiskCheckResult(
                        approved=False,
                        reason=failed_checks[0].message,
                    )

                return RiskCheckResult(
                    approved=True,
                    max_quantity=max_allowed,
                    adjustments={"limit_type": failed_checks[0].limit_type},
                )

            if opportunity.confidence < 0.5:
                return RiskCheckResult(
                    approved=False,
                    reason=f"Confidence {opportunity.confidence:.2%} below minimum 50%",
                )

            min_edge = self.settings.trading.min_edge
            if opportunity.roi < min_edge:
                return RiskCheckResult(
                    approved=False,
                    reason=f"ROI {opportunity.roi:.4%} below minimum {min_edge:.4%}",
                )

            await self._log_risk_event(
                "trade_approved",
                {
                    "opportunity_id": opportunity.opportunity_id,
                    "max_quantity": opportunity.max_quantity,
                    "total_cost": opportunity.total_cost,
                },
                "info",
            )

            return RiskCheckResult(
                approved=True,
                max_quantity=opportunity.max_quantity,
            )

    async def post_trade_update(
        self,
        success: bool,
        profit_loss: float,
        opportunity_id: str,
        error_message: str | None = None,
    ) -> None:
        """
        Update risk state after trade execution.

        Args:
            success: Whether trade succeeded
            profit_loss: Realized P&L
            opportunity_id: ID of the opportunity
            error_message: Error message if failed
        """
        result = TradeResult(
            success=success,
            profit_loss=profit_loss,
        )
        await self.circuit_breaker.record_trade(result)

        await self._log_risk_event(
            "trade_completed" if success else "trade_failed",
            {
                "opportunity_id": opportunity_id,
                "success": success,
                "profit_loss": profit_loss,
                "error": error_message,
            },
            "info" if success else "warning",
        )

    async def update_portfolio_value(self, value: float) -> None:
        """
        Update portfolio value for drawdown tracking.

        Args:
            value: Current portfolio value
        """
        await self.circuit_breaker.record_portfolio_value(value)

    async def record_api_error(self) -> None:
        """Record an API error."""
        await self.circuit_breaker.record_error()

    async def record_api_success(self) -> None:
        """Record a successful API call."""
        await self.circuit_breaker.record_success()

    async def emergency_stop(self, reason: str) -> None:
        """
        Activate emergency stop.

        Args:
            reason: Reason for emergency stop
        """
        async with self._lock:
            self._emergency_stop = True
            logger.critical(f"EMERGENCY STOP ACTIVATED: {reason}")

            await self._log_risk_event(
                "emergency_stop",
                {"reason": reason},
                "critical",
            )

    async def clear_emergency_stop(self) -> None:
        """Clear emergency stop."""
        async with self._lock:
            self._emergency_stop = False
            logger.info("Emergency stop cleared")

            await self._log_risk_event(
                "emergency_stop_cleared",
                {},
                "info",
            )

    async def reset_daily(self) -> None:
        """Reset daily risk counters."""
        await self.circuit_breaker.reset_daily()

    async def _log_risk_event(
        self,
        event_type: str,
        data: dict[str, Any],
        severity: str,
    ) -> None:
        """Log a risk event."""
        if self.audit_repo:
            try:
                await self.audit_repo.log(
                    event_type,
                    data,
                    severity,
                    component="risk_manager",
                )
            except Exception as e:
                logger.error(f"Failed to log risk event: {e}")

    def get_status(self) -> dict[str, Any]:
        """Get comprehensive risk status."""
        return {
            "emergency_stop": self._emergency_stop,
            "circuit_breakers": self.circuit_breaker.get_all_status(),
            "position_limits": self.position_limits.get_status(),
            "min_balance": self._min_balance,
        }

    def can_trade(self) -> bool:
        """Quick check if trading is allowed."""
        if self._emergency_stop:
            return False

        breaker_status = self.circuit_breaker.get_all_status()
        if breaker_status.get("any_triggered", False):
            return False

        return True
