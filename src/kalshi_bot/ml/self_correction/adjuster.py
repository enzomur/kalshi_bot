"""Position size adjustment based on performance."""

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
class PositionAdjustment:
    """Recommended position size adjustment."""

    kelly_multiplier: float  # 0.0 to 1.0
    reason: str
    accuracy: float | None
    win_rate: float | None
    should_trade: bool


class PositionAdjuster:
    """
    Adjusts position sizes based on recent performance.

    Uses accuracy to scale Kelly fraction:
    - > 60% accuracy: Full Kelly (1.0x)
    - 55-60% accuracy: 0.75x Kelly
    - 50-55% accuracy: 0.5x Kelly
    - 45-50% accuracy: 0.25x Kelly
    - < 45% accuracy: Stop trading (0.0x)
    """

    # Accuracy thresholds and corresponding Kelly multipliers
    ACCURACY_TIERS = [
        (0.60, 1.00, "Strong performance - full position sizes"),
        (0.55, 0.75, "Good performance - reduced position sizes"),
        (0.50, 0.50, "Marginal performance - half position sizes"),
        (0.45, 0.25, "Poor performance - minimal position sizes"),
        (0.00, 0.00, "Very poor performance - trading paused"),
    ]

    # Minimum predictions required before adjusting
    MIN_PREDICTIONS_FOR_ADJUSTMENT = 20

    def __init__(
        self,
        db: Database,
        monitor: PerformanceMonitor,
    ) -> None:
        """
        Initialize position adjuster.

        Args:
            db: Database connection
            monitor: Performance monitor instance
        """
        self._db = db
        self._monitor = monitor

        self._current_multiplier = 1.0
        self._last_adjustment: datetime | None = None

    async def calculate_adjustment(self) -> PositionAdjustment:
        """
        Calculate recommended position size adjustment.

        Returns:
            PositionAdjustment with recommended multiplier
        """
        # Get current metrics
        metrics = await self._monitor.update_metrics()

        # If we don't have enough predictions, use default
        if metrics.total_predictions < self.MIN_PREDICTIONS_FOR_ADJUSTMENT:
            return PositionAdjustment(
                kelly_multiplier=1.0,
                reason=f"Insufficient data ({metrics.total_predictions} predictions < {self.MIN_PREDICTIONS_FOR_ADJUSTMENT} required)",
                accuracy=metrics.accuracy,
                win_rate=metrics.win_rate,
                should_trade=True,
            )

        # Determine multiplier based on accuracy
        accuracy = metrics.accuracy or 0.5
        multiplier = 0.0
        reason = ""

        for threshold, mult, desc in self.ACCURACY_TIERS:
            if accuracy >= threshold:
                multiplier = mult
                reason = f"Accuracy {accuracy:.1%}: {desc}"
                break

        # Additional adjustment for consecutive losses
        if metrics.consecutive_losses >= 5:
            multiplier = min(multiplier, 0.25)
            reason += f" (reduced due to {metrics.consecutive_losses} consecutive losses)"

        # Additional adjustment for drawdown
        if metrics.current_drawdown >= 0.15:
            multiplier = min(multiplier, 0.5)
            reason += f" (reduced due to {metrics.current_drawdown:.1%} drawdown)"

        self._current_multiplier = multiplier
        self._last_adjustment = datetime.utcnow()

        # Save adjustment to database
        await self._save_adjustment(multiplier, reason, accuracy, metrics.win_rate)

        return PositionAdjustment(
            kelly_multiplier=multiplier,
            reason=reason,
            accuracy=accuracy,
            win_rate=metrics.win_rate,
            should_trade=multiplier > 0,
        )

    async def get_current_multiplier(self) -> float:
        """Get current Kelly multiplier."""
        # Recalculate if stale (> 1 hour)
        if self._last_adjustment is None or \
           (datetime.utcnow() - self._last_adjustment).total_seconds() > 3600:
            adjustment = await self.calculate_adjustment()
            return adjustment.kelly_multiplier

        return self._current_multiplier

    async def _save_adjustment(
        self,
        multiplier: float,
        reason: str,
        accuracy: float | None,
        win_rate: float | None,
    ) -> None:
        """Save adjustment to strategy_performance."""
        await self._db.execute(
            """
            UPDATE strategy_performance
            SET kelly_multiplier = ?, updated_at = CURRENT_TIMESTAMP
            WHERE strategy_name = 'ml_edge'
              AND id = (
                  SELECT id FROM strategy_performance
                  WHERE strategy_name = 'ml_edge'
                  ORDER BY created_at DESC
                  LIMIT 1
              )
            """,
            (multiplier,),
        )

    async def apply_to_quantity(self, base_quantity: int) -> int:
        """
        Apply adjustment to a base quantity.

        Args:
            base_quantity: Quantity before adjustment

        Returns:
            Adjusted quantity
        """
        multiplier = await self.get_current_multiplier()
        adjusted = int(base_quantity * multiplier)

        if adjusted != base_quantity:
            logger.debug(
                f"Position adjusted: {base_quantity} -> {adjusted} "
                f"(multiplier: {multiplier:.2f})"
            )

        return max(0, adjusted)

    def get_status(self) -> dict:
        """Get adjuster status."""
        return {
            "current_multiplier": self._current_multiplier,
            "last_adjustment": self._last_adjustment.isoformat() if self._last_adjustment else None,
            "accuracy_tiers": [
                {"threshold": t, "multiplier": m, "description": d}
                for t, m, d in self.ACCURACY_TIERS
            ],
            "min_predictions": self.MIN_PREDICTIONS_FOR_ADJUSTMENT,
        }
