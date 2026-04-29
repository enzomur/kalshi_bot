"""CLI and web dashboard for trading performance monitoring.

Displays:
- Calibration curves (predicted vs actual probabilities)
- Brier scores by strategy
- P&L attribution
- Risk metrics (Sharpe, drawdown, etc.)
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from src.metrics import (
    BrierCalculator,
    PerformanceTracker,
    CalibrationCurve,
    PerformanceSummary,
)
from src.observability.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DashboardState:
    """Current state for dashboard display."""

    brier_calculator: BrierCalculator | None = None
    performance_tracker: PerformanceTracker | None = None
    calibration_curve: CalibrationCurve | None = None
    last_update: datetime | None = None
    resolved_trades: int = 0


class CLIDashboard:
    """Command-line dashboard for trading metrics.

    Provides formatted output for terminal display of:
    - Performance summary
    - Calibration status
    - Strategy breakdown
    - Risk metrics
    """

    # Display configuration
    WIDTH = 70
    UPDATE_INTERVAL_TRADES = 25  # Show dashboard every N resolved trades

    def __init__(
        self,
        brier_calculator: BrierCalculator | None = None,
        performance_tracker: PerformanceTracker | None = None,
        calibration_curve: CalibrationCurve | None = None,
    ) -> None:
        """
        Initialize CLI dashboard.

        Args:
            brier_calculator: Brier score tracker
            performance_tracker: Performance metrics tracker
            calibration_curve: Calibration analysis
        """
        self._brier = brier_calculator
        self._performance = performance_tracker
        self._calibration = calibration_curve
        self._last_display_count = 0

    def should_display(self, resolved_count: int) -> bool:
        """Check if dashboard should be displayed based on resolved trade count."""
        if resolved_count == 0:
            return False

        if resolved_count >= self._last_display_count + self.UPDATE_INTERVAL_TRADES:
            self._last_display_count = resolved_count
            return True

        return False

    def display(self) -> str:
        """Generate and return full dashboard output."""
        lines = []

        lines.append(self._header("KALSHI TRADING BOT DASHBOARD"))
        lines.append(f"Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        lines.append("")

        # Performance Summary
        if self._performance:
            lines.extend(self._performance_section())
            lines.append("")

        # Brier Score
        if self._brier:
            lines.extend(self._brier_section())
            lines.append("")

        # Calibration
        if self._calibration:
            lines.extend(self._calibration_section())
            lines.append("")

        # Strategy Breakdown
        if self._performance:
            lines.extend(self._strategy_section())

        lines.append(self._header("END DASHBOARD"))

        output = "\n".join(lines)
        return output

    def print_dashboard(self) -> None:
        """Print dashboard to stdout."""
        print(self.display())

    def _header(self, title: str) -> str:
        """Create a section header."""
        padding = (self.WIDTH - len(title) - 2) // 2
        return "=" * padding + f" {title} " + "=" * padding

    def _subheader(self, title: str) -> str:
        """Create a subsection header."""
        return f"--- {title} ---"

    def _performance_section(self) -> list[str]:
        """Generate performance summary section."""
        lines = [self._subheader("PERFORMANCE SUMMARY")]

        if not self._performance:
            lines.append("No performance data available")
            return lines

        summary = self._performance.get_summary()

        # Key metrics table
        lines.append(f"{'Metric':<25} {'Value':>15}")
        lines.append("-" * 42)
        lines.append(f"{'Total Trades':<25} {summary.total_trades:>15}")
        lines.append(f"{'Win Rate':<25} {summary.win_rate:>14.1%}")
        lines.append(f"{'Total P&L':<25} ${summary.total_pnl:>13.2f}")
        lines.append(f"{'ROI':<25} {summary.roi:>14.1%}")

        if summary.sharpe_ratio is not None:
            lines.append(f"{'Sharpe Ratio':<25} {summary.sharpe_ratio:>15.2f}")
        else:
            lines.append(f"{'Sharpe Ratio':<25} {'N/A':>15}")

        lines.append(f"{'Max Drawdown':<25} {summary.max_drawdown:>14.1%}")
        lines.append(f"{'Current Drawdown':<25} {summary.current_drawdown:>14.1%}")
        lines.append("")

        # Balance info
        lines.append(f"Initial Balance: ${summary.initial_balance:.2f}")
        lines.append(f"Current Balance: ${summary.current_balance:.2f}")
        lines.append(f"Peak Balance:    ${summary.peak_balance:.2f}")

        return lines

    def _brier_section(self) -> list[str]:
        """Generate Brier score section."""
        lines = [self._subheader("CALIBRATION SCORE (BRIER)")]

        if not self._brier:
            lines.append("No Brier data available")
            return lines

        status = self._brier.get_status()
        result = self._brier.calculate_brier(min_predictions=10)

        lines.append(f"Predictions: {status['total_predictions']} total, {status['resolved']} resolved")

        if result:
            # Brier score with interpretation
            brier = result.brier_score
            if brier < 0.15:
                interpretation = "Excellent"
                indicator = "[+++]"
            elif brier < 0.22:
                interpretation = "Good (Target Met)"
                indicator = "[++ ]"
            elif brier < 0.25:
                interpretation = "Fair"
                indicator = "[+  ]"
            else:
                interpretation = "Poor"
                indicator = "[   ]"

            lines.append("")
            lines.append(f"Brier Score: {brier:.4f}  {indicator}  {interpretation}")
            lines.append(f"Target: < 0.22")
            lines.append("")
            lines.append(f"Win Rate: {result.win_rate:.1%}")
            lines.append(f"Mean Predicted: {result.mean_probability:.1%}")

            # Per-strategy breakdown
            if result.strategy_scores:
                lines.append("")
                lines.append("By Strategy:")
                for strategy, score in sorted(result.strategy_scores.items()):
                    lines.append(f"  {strategy:<20} {score:.4f}")
        else:
            lines.append("Need 10+ resolved predictions for Brier score")

        return lines

    def _calibration_section(self) -> list[str]:
        """Generate calibration curve section."""
        lines = [self._subheader("CALIBRATION CURVE")]

        if not self._calibration:
            lines.append("No calibration data available")
            return lines

        analysis = self._calibration.analyze()

        if analysis is None:
            lines.append("Need more data for calibration analysis")
            return lines

        # Summary metrics
        lines.append(f"Max Deviation:  {analysis.max_deviation:.1%}")
        lines.append(f"Mean Abs Dev:   {analysis.mean_absolute_deviation:.1%}")
        lines.append(f"Well Calibrated: {'YES' if analysis.is_well_calibrated else 'NO'}")

        if analysis.overall_overconfident:
            lines.append("Tendency: OVERCONFIDENT (predictions too high)")
        elif analysis.overall_underconfident:
            lines.append("Tendency: UNDERCONFIDENT (predictions too low)")
        else:
            lines.append("Tendency: Balanced")

        lines.append("")

        # Bucket table
        lines.append(f"{'Bucket':<12} {'N':>6} {'Predicted':>10} {'Actual':>10} {'Dev':>8}")
        lines.append("-" * 50)

        for bucket in analysis.buckets:
            range_str = f"{bucket.bucket_start:.0%}-{bucket.bucket_end:.0%}"
            dev_str = f"{bucket.deviation:+.1%}"
            if abs(bucket.deviation) > 0.05:
                dev_str += " !"

            lines.append(
                f"{range_str:<12} {bucket.n_predictions:>6} "
                f"{bucket.mean_predicted:>9.1%} {bucket.actual_rate:>9.1%} "
                f"{dev_str:>8}"
            )

        return lines

    def _strategy_section(self) -> list[str]:
        """Generate strategy breakdown section."""
        lines = [self._subheader("STRATEGY BREAKDOWN")]

        if not self._performance:
            lines.append("No strategy data available")
            return lines

        summary = self._performance.get_summary()

        if not summary.strategy_metrics:
            lines.append("No per-strategy metrics yet")
            return lines

        lines.append(
            f"{'Strategy':<15} {'Trades':>8} {'Wins':>6} {'Win%':>8} {'P&L':>12}"
        )
        lines.append("-" * 55)

        for strategy, metrics in sorted(summary.strategy_metrics.items()):
            lines.append(
                f"{strategy:<15} {metrics['trades']:>8} {metrics['wins']:>6} "
                f"{metrics['win_rate']:>7.1%} ${metrics['pnl']:>10.2f}"
            )

        return lines


class DashboardManager:
    """Manages dashboard state and updates.

    Tracks when to display dashboard and coordinates metrics sources.
    """

    def __init__(
        self,
        brier_calculator: BrierCalculator | None = None,
        performance_tracker: PerformanceTracker | None = None,
        calibration_curve: CalibrationCurve | None = None,
        auto_display: bool = True,
        display_interval: int = 25,
    ) -> None:
        """
        Initialize dashboard manager.

        Args:
            brier_calculator: Brier score tracker
            performance_tracker: Performance metrics
            calibration_curve: Calibration analysis
            auto_display: Whether to auto-display on interval
            display_interval: Trades between auto-displays
        """
        self._brier = brier_calculator
        self._performance = performance_tracker
        self._calibration = calibration_curve
        self._auto_display = auto_display
        self._display_interval = display_interval

        self._cli_dashboard = CLIDashboard(
            brier_calculator=brier_calculator,
            performance_tracker=performance_tracker,
            calibration_curve=calibration_curve,
        )

        self._last_resolved_count = 0

    def on_trade_resolved(self, won: bool, predicted_prob: float) -> None:
        """
        Called when a trade is resolved.

        Updates calibration tracking and potentially displays dashboard.

        Args:
            won: Whether the trade was won
            predicted_prob: The predicted probability for this trade
        """
        # Add to calibration curve
        if self._calibration:
            self._calibration.add_prediction(predicted_prob, won)

        # Check if should display
        current_count = 0
        if self._brier:
            current_count = len(self._brier.get_resolved_predictions())

        if self._auto_display:
            if current_count >= self._last_resolved_count + self._display_interval:
                self._last_resolved_count = current_count
                self.display()

    def display(self) -> None:
        """Display the dashboard."""
        print("\n")
        self._cli_dashboard.print_dashboard()
        print("\n")

    def get_dashboard_text(self) -> str:
        """Get dashboard as text without printing."""
        return self._cli_dashboard.display()

    def get_metrics_summary(self) -> dict[str, Any]:
        """Get summary metrics as dictionary."""
        result: dict[str, Any] = {}

        if self._performance:
            summary = self._performance.get_summary()
            result["performance"] = {
                "trades": summary.total_trades,
                "win_rate": summary.win_rate,
                "pnl": summary.total_pnl,
                "roi": summary.roi,
                "sharpe": summary.sharpe_ratio,
                "max_drawdown": summary.max_drawdown,
            }

        if self._brier:
            brier_result = self._brier.calculate_brier(min_predictions=10)
            if brier_result:
                result["brier"] = {
                    "score": brier_result.brier_score,
                    "n_predictions": brier_result.n_predictions,
                    "is_calibrated": brier_result.is_calibrated,
                }

        if self._calibration:
            cal_result = self._calibration.analyze()
            if cal_result:
                result["calibration"] = {
                    "max_deviation": cal_result.max_deviation,
                    "mean_abs_deviation": cal_result.mean_absolute_deviation,
                    "is_well_calibrated": cal_result.is_well_calibrated,
                }

        return result


def print_startup_banner(mode: str, balance: float, strategies: list[str]) -> None:
    """Print startup banner with bot info."""
    width = 60
    print("=" * width)
    print("  KALSHI TRADING BOT v2.0")
    print("=" * width)
    print(f"  Mode:       {mode.upper()}")
    print(f"  Balance:    ${balance:.2f}")
    print(f"  Strategies: {', '.join(strategies) if strategies else 'None'}")
    print("=" * width)
    print()


def print_shutdown_summary(
    performance: PerformanceTracker | None,
    brier: BrierCalculator | None,
) -> None:
    """Print summary at shutdown."""
    print("\n")
    print("=" * 60)
    print("  SHUTDOWN SUMMARY")
    print("=" * 60)

    if performance:
        summary = performance.get_summary()
        print(f"  Total Trades:  {summary.total_trades}")
        print(f"  Win Rate:      {summary.win_rate:.1%}")
        print(f"  Total P&L:     ${summary.total_pnl:.2f}")
        print(f"  ROI:           {summary.roi:.1%}")

    if brier:
        result = brier.calculate_brier(min_predictions=5)
        if result:
            print(f"  Brier Score:   {result.brier_score:.4f}")
            status = "PASS" if result.is_calibrated else "FAIL"
            print(f"  Calibration:   {status}")

    print("=" * 60)
    print()
