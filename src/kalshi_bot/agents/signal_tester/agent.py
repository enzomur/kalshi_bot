"""Signal Tester Agent - discovers and validates trading signals."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from kalshi_bot.agents.base import BaseAgent
from kalshi_bot.agents.signal_tester.backtest_runner import SignalBacktestRunner
from kalshi_bot.agents.signal_tester.signal_generator import SignalGenerator
from kalshi_bot.agents.signal_tester.signal_ranker import (
    SignalRanker,
    SignalRecommendation,
)
from kalshi_bot.utils.logging import get_logger

if TYPE_CHECKING:
    from kalshi_bot.persistence.database import Database

logger = get_logger(__name__)


class SignalTesterAgent(BaseAgent):
    """
    Agent that tests candidate trading signals.

    Runs periodically to:
    1. Generate candidate signals from signal library
    2. Backtest each signal against historical data
    3. Rank signals by predictive power
    4. Store approved signals for use by EdgePredictor
    """

    def __init__(
        self,
        db: "Database",
        backtest_days: int = 90,
        required_win_rate: float = 0.55,
        update_interval_hours: int = 24,
        enabled: bool = True,
    ) -> None:
        """
        Initialize the Signal Tester Agent.

        Args:
            db: Database connection
            backtest_days: Number of days for backtesting
            required_win_rate: Minimum win rate to approve signals
            update_interval_hours: How often to run tests
            enabled: Whether the agent is enabled
        """
        super().__init__(
            db=db,
            name="signal_tester",
            update_interval_seconds=update_interval_hours * 3600,
            enabled=enabled,
        )

        self._signal_generator = SignalGenerator()
        self._backtest_runner = SignalBacktestRunner(
            db=db,
            backtest_days=backtest_days,
        )
        self._ranker = SignalRanker(
            required_win_rate=required_win_rate,
        )

        # Track approved signals
        self._approved_signals: list[str] = []

    async def _run_cycle(self) -> None:
        """Execute one cycle of signal testing."""
        logger.info("Signal tester: starting backtest cycle")

        # Get all candidate signals
        signals = self._signal_generator.get_all_signals()
        logger.info(f"Signal tester: testing {len(signals)} candidates")

        # Run backtests
        results = await self._backtest_runner.run_all_backtests(signals)

        # Rank signals
        ranked = self._ranker.rank_signals(results)

        # Store results in database
        await self._store_results(ranked)

        # Update approved signals list
        approved = self._ranker.get_approved_signals(ranked)
        self._approved_signals = [s.signal_id for s in approved]

        # Update metrics
        self._status.metrics = {
            "signals_tested": len(signals),
            "signals_approved": len(approved),
            "signals_rejected": sum(
                1 for s in ranked
                if s.recommendation == SignalRecommendation.REJECT
            ),
            "signals_review": sum(
                1 for s in ranked
                if s.recommendation == SignalRecommendation.REVIEW
            ),
            "top_signal": approved[0].signal_id if approved else None,
            "top_win_rate": approved[0].win_rate if approved else None,
        }

        logger.info(
            f"Signal tester: {len(approved)} approved, "
            f"{len(ranked) - len(approved)} not approved"
        )

    async def _store_results(self, ranked: list) -> None:
        """Store backtest results in database."""
        for signal in ranked:
            # Update signal status
            await self._db.execute(
                """
                INSERT OR REPLACE INTO signal_candidates
                (signal_id, name, feature_formula, status, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    signal.signal_id,
                    signal.signal_id,  # Use ID as name for now
                    "",  # Formula stored elsewhere
                    signal.recommendation.value,
                    datetime.utcnow().isoformat(),
                ),
            )

            # Store backtest result
            await self._db.execute(
                """
                INSERT INTO signal_backtest_results
                (signal_id, sample_size, win_rate, information_coefficient,
                 sharpe_ratio, p_value, recommended_action, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    signal.signal_id,
                    signal.sample_size,
                    signal.win_rate,
                    signal.information_coefficient,
                    signal.sharpe_ratio,
                    signal.p_value,
                    signal.recommendation.value,
                    signal.notes,
                ),
            )

    def get_approved_signals(self) -> list[str]:
        """Get list of approved signal IDs."""
        return self._approved_signals.copy()

    async def get_signal_status(self, signal_id: str) -> dict[str, Any] | None:
        """Get status of a specific signal."""
        row = await self._db.fetch_one(
            """
            SELECT * FROM signal_backtest_results
            WHERE signal_id = ?
            ORDER BY backtest_run_at DESC
            LIMIT 1
            """,
            (signal_id,),
        )
        return dict(row) if row else None

    def get_status(self) -> dict[str, Any]:
        """Get agent status with signal-specific metrics."""
        status = super().get_status()
        status["approved_signals"] = self._approved_signals
        return status
