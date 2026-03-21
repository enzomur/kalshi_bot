"""Performance monitoring for ML strategies."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from kalshi_bot.utils.logging import get_logger

if TYPE_CHECKING:
    from kalshi_bot.persistence.database import Database

logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Current performance metrics snapshot."""

    # Prediction accuracy
    total_predictions: int
    correct_predictions: int
    accuracy: float | None

    # Trading performance
    total_trades: int
    winning_trades: int
    win_rate: float | None

    # P&L
    total_pnl: float
    total_fees: float
    net_pnl: float

    # Risk
    max_drawdown: float
    current_drawdown: float

    # Streaks
    consecutive_wins: int
    consecutive_losses: int

    # Rolling window info
    window_start: datetime
    window_end: datetime

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "total_predictions": self.total_predictions,
            "correct_predictions": self.correct_predictions,
            "accuracy": self.accuracy,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "win_rate": self.win_rate,
            "total_pnl": self.total_pnl,
            "total_fees": self.total_fees,
            "net_pnl": self.net_pnl,
            "max_drawdown": self.max_drawdown,
            "current_drawdown": self.current_drawdown,
            "consecutive_wins": self.consecutive_wins,
            "consecutive_losses": self.consecutive_losses,
            "window_start": self.window_start.isoformat(),
            "window_end": self.window_end.isoformat(),
        }


class PerformanceMonitor:
    """
    Monitors ML strategy performance over rolling windows.

    Tracks:
    - Prediction accuracy (rolling N predictions)
    - Realized P&L
    - Win rate
    - Drawdown
    - Consecutive win/loss streaks
    """

    # Rolling window size for metrics
    ROLLING_PREDICTIONS = 50
    ROLLING_TRADES = 20

    def __init__(
        self,
        db: Database,
        strategy_name: str = "ml_edge",
    ) -> None:
        """
        Initialize performance monitor.

        Args:
            db: Database connection
            strategy_name: Name of strategy to monitor
        """
        self._db = db
        self._strategy_name = strategy_name

        self._peak_value: float = 0.0
        self._current_value: float = 0.0
        self._last_update: datetime | None = None

    async def update_metrics(
        self,
        current_portfolio_value: float | None = None,
    ) -> PerformanceMetrics:
        """
        Update and return current performance metrics.

        Args:
            current_portfolio_value: Current total portfolio value

        Returns:
            Current PerformanceMetrics
        """
        now = datetime.utcnow()

        # Update drawdown tracking
        if current_portfolio_value is not None:
            self._current_value = current_portfolio_value
            if current_portfolio_value > self._peak_value:
                self._peak_value = current_portfolio_value

        # Calculate drawdown
        if self._peak_value > 0:
            current_drawdown = (self._peak_value - self._current_value) / self._peak_value
        else:
            current_drawdown = 0.0

        # Get prediction accuracy
        prediction_stats = await self._get_prediction_stats()

        # Get trading stats
        trading_stats = await self._get_trading_stats()

        # Get streak info
        streaks = await self._get_streaks()

        # Get P&L
        pnl_stats = await self._get_pnl_stats()

        # Get max historical drawdown
        max_drawdown = await self._get_max_drawdown()

        metrics = PerformanceMetrics(
            total_predictions=prediction_stats["total"],
            correct_predictions=prediction_stats["correct"],
            accuracy=prediction_stats["accuracy"],
            total_trades=trading_stats["total"],
            winning_trades=trading_stats["winning"],
            win_rate=trading_stats["win_rate"],
            total_pnl=pnl_stats["total_pnl"],
            total_fees=pnl_stats["total_fees"],
            net_pnl=pnl_stats["net_pnl"],
            max_drawdown=max(max_drawdown, current_drawdown),
            current_drawdown=current_drawdown,
            consecutive_wins=streaks["consecutive_wins"],
            consecutive_losses=streaks["consecutive_losses"],
            window_start=now - timedelta(days=7),
            window_end=now,
        )

        # Save metrics to database
        await self._save_metrics(metrics)

        self._last_update = now
        return metrics

    async def _get_prediction_stats(self) -> dict:
        """Get prediction accuracy stats."""
        result = await self._db.fetch_one(
            """
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) as correct
            FROM (
                SELECT was_correct FROM ml_predictions
                WHERE actual_outcome IS NOT NULL
                ORDER BY predicted_at DESC
                LIMIT ?
            )
            """,
            (self.ROLLING_PREDICTIONS,),
        )

        if result and result["total"] > 0:
            return {
                "total": result["total"],
                "correct": result["correct"] or 0,
                "accuracy": (result["correct"] or 0) / result["total"],
            }

        return {"total": 0, "correct": 0, "accuracy": None}

    async def _get_trading_stats(self) -> dict:
        """Get trading win rate stats from opportunities."""
        # Look at executed ML opportunities
        result = await self._db.fetch_one(
            """
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN execution_result LIKE '%"success": true%' THEN 1 ELSE 0 END) as winning
            FROM opportunities
            WHERE arbitrage_type = 'single_market'
              AND execution_result IS NOT NULL
              AND detected_at >= datetime('now', '-7 days')
            ORDER BY detected_at DESC
            LIMIT ?
            """,
            (self.ROLLING_TRADES,),
        )

        if result and result["total"] > 0:
            return {
                "total": result["total"],
                "winning": result["winning"] or 0,
                "win_rate": (result["winning"] or 0) / result["total"],
            }

        return {"total": 0, "winning": 0, "win_rate": None}

    async def _get_streaks(self) -> dict:
        """Get consecutive win/loss streaks."""
        # Get recent predictions in order
        predictions = await self._db.fetch_all(
            """
            SELECT was_correct FROM ml_predictions
            WHERE actual_outcome IS NOT NULL
            ORDER BY settled_at DESC
            LIMIT 20
            """
        )

        consecutive_wins = 0
        consecutive_losses = 0

        for pred in predictions:
            if pred["was_correct"]:
                if consecutive_losses > 0:
                    break
                consecutive_wins += 1
            else:
                if consecutive_wins > 0:
                    break
                consecutive_losses += 1

        return {
            "consecutive_wins": consecutive_wins,
            "consecutive_losses": consecutive_losses,
        }

    async def _get_pnl_stats(self) -> dict:
        """Get P&L statistics."""
        result = await self._db.fetch_one(
            """
            SELECT
                COALESCE(SUM(
                    CASE
                        WHEN execution_result LIKE '%"success": true%'
                        THEN CAST(json_extract(execution_result, '$.profit') AS REAL)
                        ELSE 0
                    END
                ), 0) as total_pnl,
                COALESCE(SUM(
                    CASE
                        WHEN execution_result IS NOT NULL
                        THEN CAST(json_extract(execution_result, '$.fees') AS REAL)
                        ELSE 0
                    END
                ), 0) as total_fees
            FROM opportunities
            WHERE detected_at >= datetime('now', '-7 days')
            """
        )

        total_pnl = result["total_pnl"] or 0 if result else 0
        total_fees = result["total_fees"] or 0 if result else 0

        return {
            "total_pnl": total_pnl,
            "total_fees": total_fees,
            "net_pnl": total_pnl - total_fees,
        }

    async def _get_max_drawdown(self) -> float:
        """Get maximum historical drawdown."""
        result = await self._db.fetch_one(
            """
            SELECT MAX(drawdown) as max_dd
            FROM portfolio_snapshots
            WHERE snapshot_at >= datetime('now', '-30 days')
            """
        )

        return result["max_dd"] or 0.0 if result else 0.0

    async def _save_metrics(self, metrics: PerformanceMetrics) -> None:
        """Save metrics snapshot to database."""
        await self._db.execute(
            """
            INSERT INTO strategy_performance (
                strategy_name, window_start, window_end,
                total_predictions, correct_predictions, accuracy,
                total_trades, winning_trades, win_rate,
                total_pnl, total_fees, net_pnl,
                max_drawdown, consecutive_wins, consecutive_losses
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self._strategy_name,
                metrics.window_start.isoformat(),
                metrics.window_end.isoformat(),
                metrics.total_predictions,
                metrics.correct_predictions,
                metrics.accuracy,
                metrics.total_trades,
                metrics.winning_trades,
                metrics.win_rate,
                metrics.total_pnl,
                metrics.total_fees,
                metrics.net_pnl,
                metrics.max_drawdown,
                metrics.consecutive_wins,
                metrics.consecutive_losses,
            ),
        )

    def set_peak_value(self, value: float) -> None:
        """Set initial peak value."""
        self._peak_value = value
        self._current_value = value

    def get_status(self) -> dict:
        """Get monitor status."""
        return {
            "strategy_name": self._strategy_name,
            "peak_value": self._peak_value,
            "current_value": self._current_value,
            "last_update": self._last_update.isoformat() if self._last_update else None,
            "rolling_windows": {
                "predictions": self.ROLLING_PREDICTIONS,
                "trades": self.ROLLING_TRADES,
            },
        }
