"""Strategy disabler for automatic risk management."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from kalshi_bot.ml.self_correction.monitor import PerformanceMonitor
from kalshi_bot.utils.logging import get_logger

if TYPE_CHECKING:
    from kalshi_bot.persistence.database import Database

logger = get_logger(__name__)


@dataclass
class DisableCondition:
    """A condition that can trigger strategy disable."""

    name: str
    threshold: float
    current_value: float
    triggered: bool
    description: str


@dataclass
class StrategyState:
    """Current state of strategy."""

    is_enabled: bool
    disabled_reason: str | None
    disabled_at: datetime | None
    conditions_checked: list[DisableCondition]
    manual_override: bool


class StrategyDisabler:
    """
    Automatically disables ML strategy when performance degrades.

    Disable triggers:
    - Accuracy < 45% over 50 predictions
    - 5+ consecutive losses
    - Daily loss > $50
    - Drawdown > 20%

    Requires manual re-enable via dashboard.
    """

    # Disable thresholds
    ACCURACY_DISABLE_THRESHOLD = 0.45
    CONSECUTIVE_LOSS_THRESHOLD = 5
    DAILY_LOSS_THRESHOLD = 50.0  # $50
    DRAWDOWN_THRESHOLD = 0.20  # 20%

    # Minimum predictions before auto-disable logic applies
    MIN_PREDICTIONS = 30

    def __init__(
        self,
        db: Database,
        monitor: PerformanceMonitor,
        strategy_name: str = "ml_edge",
    ) -> None:
        """
        Initialize strategy disabler.

        Args:
            db: Database connection
            monitor: Performance monitor instance
            strategy_name: Name of strategy to manage
        """
        self._db = db
        self._monitor = monitor
        self._strategy_name = strategy_name

        self._is_enabled = True
        self._disabled_reason: str | None = None
        self._disabled_at: datetime | None = None
        self._manual_override = False

    async def check_disable_conditions(self) -> StrategyState:
        """
        Check all disable conditions and update strategy state.

        Returns:
            Current StrategyState
        """
        # If manually disabled, don't auto-enable
        if self._manual_override and not self._is_enabled:
            return self._get_current_state([])

        # Get current metrics
        metrics = await self._monitor.update_metrics()

        conditions: list[DisableCondition] = []

        # Check accuracy
        if metrics.total_predictions >= self.MIN_PREDICTIONS:
            accuracy = metrics.accuracy or 0.5
            conditions.append(DisableCondition(
                name="accuracy",
                threshold=self.ACCURACY_DISABLE_THRESHOLD,
                current_value=accuracy,
                triggered=accuracy < self.ACCURACY_DISABLE_THRESHOLD,
                description=f"Accuracy {accuracy:.1%} < {self.ACCURACY_DISABLE_THRESHOLD:.0%} threshold",
            ))

        # Check consecutive losses
        conditions.append(DisableCondition(
            name="consecutive_losses",
            threshold=float(self.CONSECUTIVE_LOSS_THRESHOLD),
            current_value=float(metrics.consecutive_losses),
            triggered=metrics.consecutive_losses >= self.CONSECUTIVE_LOSS_THRESHOLD,
            description=f"{metrics.consecutive_losses} consecutive losses >= {self.CONSECUTIVE_LOSS_THRESHOLD} threshold",
        ))

        # Check daily loss
        daily_loss = await self._get_daily_loss()
        conditions.append(DisableCondition(
            name="daily_loss",
            threshold=self.DAILY_LOSS_THRESHOLD,
            current_value=abs(daily_loss) if daily_loss < 0 else 0,
            triggered=daily_loss < -self.DAILY_LOSS_THRESHOLD,
            description=f"Daily loss ${abs(daily_loss):.2f} > ${self.DAILY_LOSS_THRESHOLD:.2f} threshold",
        ))

        # Check drawdown
        conditions.append(DisableCondition(
            name="drawdown",
            threshold=self.DRAWDOWN_THRESHOLD,
            current_value=metrics.current_drawdown,
            triggered=metrics.current_drawdown > self.DRAWDOWN_THRESHOLD,
            description=f"Drawdown {metrics.current_drawdown:.1%} > {self.DRAWDOWN_THRESHOLD:.0%} threshold",
        ))

        # Check if any condition is triggered
        triggered_conditions = [c for c in conditions if c.triggered]

        if triggered_conditions and self._is_enabled:
            # Disable strategy
            reason = "; ".join(c.description for c in triggered_conditions)
            await self._disable_strategy(reason)
            logger.warning(f"ML strategy auto-disabled: {reason}")

        return self._get_current_state(conditions)

    async def _get_daily_loss(self) -> float:
        """Get today's P&L."""
        result = await self._db.fetch_one(
            """
            SELECT COALESCE(SUM(net_pnl), 0) as daily_pnl
            FROM strategy_performance
            WHERE strategy_name = ?
              AND DATE(window_end) = DATE('now')
            """,
            (self._strategy_name,),
        )
        return result["daily_pnl"] if result else 0.0

    async def _disable_strategy(self, reason: str) -> None:
        """Disable the strategy."""
        self._is_enabled = False
        self._disabled_reason = reason
        self._disabled_at = datetime.utcnow()

        # Update database
        await self._db.execute(
            """
            UPDATE strategy_performance
            SET is_enabled = FALSE,
                disabled_reason = ?,
                disabled_at = CURRENT_TIMESTAMP
            WHERE strategy_name = ?
              AND id = (
                  SELECT id FROM strategy_performance
                  WHERE strategy_name = ?
                  ORDER BY created_at DESC
                  LIMIT 1
              )
            """,
            (reason, self._strategy_name, self._strategy_name),
        )

    async def enable_strategy(self) -> bool:
        """
        Manually re-enable the strategy.

        Returns:
            True if successfully enabled
        """
        self._is_enabled = True
        self._disabled_reason = None
        self._disabled_at = None
        self._manual_override = False

        # Update database
        await self._db.execute(
            """
            UPDATE strategy_performance
            SET is_enabled = TRUE,
                disabled_reason = NULL,
                disabled_at = NULL
            WHERE strategy_name = ?
              AND id = (
                  SELECT id FROM strategy_performance
                  WHERE strategy_name = ?
                  ORDER BY created_at DESC
                  LIMIT 1
              )
            """,
            (self._strategy_name, self._strategy_name),
        )

        logger.info("ML strategy manually re-enabled")
        return True

    async def disable_strategy_manual(self, reason: str = "Manual disable") -> None:
        """Manually disable the strategy."""
        self._manual_override = True
        await self._disable_strategy(reason)
        logger.info(f"ML strategy manually disabled: {reason}")

    def _get_current_state(self, conditions: list[DisableCondition]) -> StrategyState:
        """Get current strategy state."""
        return StrategyState(
            is_enabled=self._is_enabled,
            disabled_reason=self._disabled_reason,
            disabled_at=self._disabled_at,
            conditions_checked=conditions,
            manual_override=self._manual_override,
        )

    @property
    def is_enabled(self) -> bool:
        """Check if strategy is enabled."""
        return self._is_enabled

    async def load_state(self) -> None:
        """Load strategy state from database."""
        result = await self._db.fetch_one(
            """
            SELECT is_enabled, disabled_reason, disabled_at
            FROM strategy_performance
            WHERE strategy_name = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (self._strategy_name,),
        )

        if result:
            self._is_enabled = bool(result["is_enabled"])
            self._disabled_reason = result["disabled_reason"]
            if result["disabled_at"]:
                try:
                    self._disabled_at = datetime.fromisoformat(
                        result["disabled_at"].replace("Z", "+00:00")
                    )
                except (ValueError, AttributeError):
                    self._disabled_at = None

    def get_status(self) -> dict:
        """Get disabler status."""
        return {
            "is_enabled": self._is_enabled,
            "disabled_reason": self._disabled_reason,
            "disabled_at": self._disabled_at.isoformat() if self._disabled_at else None,
            "manual_override": self._manual_override,
            "thresholds": {
                "accuracy": self.ACCURACY_DISABLE_THRESHOLD,
                "consecutive_losses": self.CONSECUTIVE_LOSS_THRESHOLD,
                "daily_loss": self.DAILY_LOSS_THRESHOLD,
                "drawdown": self.DRAWDOWN_THRESHOLD,
            },
            "min_predictions": self.MIN_PREDICTIONS,
        }
