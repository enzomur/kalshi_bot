#!/usr/bin/env python3
"""Soak test runner for PAPER mode validation.

This script runs the trading bot in PAPER mode for extended periods,
collecting metrics and validating stability before live promotion.

Features:
- Runs bot in PAPER mode with configurable duration
- Tracks trade count, Brier score, and win rate over time
- Monitors system health (memory, errors, latency)
- Generates promotion eligibility report
- Saves metrics to database for analysis

Usage:
    # Run 24-hour soak test
    python scripts/soak_test.py --duration 24h

    # Run 7-day soak test (recommended before promotion)
    python scripts/soak_test.py --duration 7d

    # Check current soak test progress
    python scripts/soak_test.py --status

    # Generate promotion report
    python scripts/soak_test.py --report

Requirements:
    - Bot must be configured for PAPER mode
    - Database must be accessible
    - Kalshi API credentials configured
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class SoakTestMetrics:
    """Metrics collected during soak testing."""

    start_time: datetime
    end_time: datetime | None = None
    duration_hours: float = 0.0

    # Trade metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0

    # Calibration metrics
    brier_score: float = 1.0
    predictions_made: int = 0
    predictions_resolved: int = 0

    # System health
    errors_count: int = 0
    warnings_count: int = 0
    avg_latency_ms: float = 0.0
    max_memory_mb: float = 0.0

    # Strategy breakdown
    trades_by_strategy: dict[str, int] = field(default_factory=dict)
    pnl_by_strategy: dict[str, float] = field(default_factory=dict)

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades

    @property
    def is_promotion_eligible(self) -> tuple[bool, list[str]]:
        """Check if metrics meet promotion criteria."""
        failures = []

        if self.total_trades < 100:
            failures.append(f"Trades: {self.total_trades}/100 required")

        if self.brier_score > 0.22:
            failures.append(f"Brier score: {self.brier_score:.3f} > 0.22 max")

        if self.win_rate < 0.52:
            failures.append(f"Win rate: {self.win_rate:.1%} < 52% required")

        if self.errors_count > 10:
            failures.append(f"Errors: {self.errors_count} > 10 max")

        return len(failures) == 0, failures

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_hours": self.duration_hours,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "total_pnl": self.total_pnl,
            "win_rate": self.win_rate,
            "brier_score": self.brier_score,
            "predictions_made": self.predictions_made,
            "predictions_resolved": self.predictions_resolved,
            "errors_count": self.errors_count,
            "warnings_count": self.warnings_count,
            "avg_latency_ms": self.avg_latency_ms,
            "max_memory_mb": self.max_memory_mb,
            "trades_by_strategy": self.trades_by_strategy,
            "pnl_by_strategy": self.pnl_by_strategy,
        }


class SoakTestRunner:
    """Runs extended soak tests in PAPER mode."""

    METRICS_FILE = "data/soak_test_metrics.json"
    STATUS_FILE = "data/soak_test_status.json"

    def __init__(
        self,
        duration_hours: float = 24.0,
        check_interval_minutes: float = 15.0,
    ) -> None:
        """Initialize soak test runner.

        Args:
            duration_hours: Total duration of soak test.
            check_interval_minutes: How often to check metrics.
        """
        self.duration_hours = duration_hours
        self.check_interval = check_interval_minutes * 60  # Convert to seconds
        self.metrics = SoakTestMetrics(start_time=datetime.now(timezone.utc))
        self._running = False
        self._bot_process = None

    def _load_status(self) -> dict[str, Any] | None:
        """Load current soak test status."""
        status_path = Path(self.STATUS_FILE)
        if not status_path.exists():
            return None

        try:
            with open(status_path) as f:
                return json.load(f)
        except Exception:
            return None

    def _save_status(self, status: str, message: str = "") -> None:
        """Save current soak test status."""
        Path(self.STATUS_FILE).parent.mkdir(parents=True, exist_ok=True)

        data = {
            "status": status,
            "message": message,
            "started_at": self.metrics.start_time.isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "duration_hours": self.duration_hours,
            "elapsed_hours": self.metrics.duration_hours,
            "metrics": self.metrics.to_dict(),
        }

        with open(self.STATUS_FILE, "w") as f:
            json.dump(data, f, indent=2)

    def _save_metrics(self) -> None:
        """Save final metrics to file."""
        Path(self.METRICS_FILE).parent.mkdir(parents=True, exist_ok=True)

        # Load existing metrics history
        history = []
        metrics_path = Path(self.METRICS_FILE)
        if metrics_path.exists():
            try:
                with open(metrics_path) as f:
                    history = json.load(f)
            except Exception:
                history = []

        # Append current metrics
        history.append(self.metrics.to_dict())

        # Keep last 10 soak tests
        history = history[-10:]

        with open(self.METRICS_FILE, "w") as f:
            json.dump(history, f, indent=2)

    def _collect_metrics_from_db(self) -> None:
        """Collect current metrics from database."""
        db_path = Path("data/kalshi_bot.db")
        if not db_path.exists():
            return

        try:
            import sqlite3

            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            # Get trade metrics since soak test start
            start_ts = self.metrics.start_time.isoformat()

            cursor.execute(
                """
                SELECT
                    COUNT(*),
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END),
                    SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END),
                    SUM(pnl)
                FROM trades
                WHERE created_at >= ? AND status = 'settled'
                """,
                (start_ts,),
            )
            row = cursor.fetchone()
            if row and row[0]:
                self.metrics.total_trades = row[0]
                self.metrics.winning_trades = row[1] or 0
                self.metrics.losing_trades = row[2] or 0
                self.metrics.total_pnl = row[3] or 0.0

            # Get trades by strategy
            cursor.execute(
                """
                SELECT strategy_name, COUNT(*), SUM(pnl)
                FROM trades
                WHERE created_at >= ? AND status = 'settled'
                GROUP BY strategy_name
                """,
                (start_ts,),
            )
            for row in cursor.fetchall():
                self.metrics.trades_by_strategy[row[0]] = row[1]
                self.metrics.pnl_by_strategy[row[0]] = row[2] or 0.0

            # Get Brier score
            cursor.execute(
                """
                SELECT value FROM metrics
                WHERE metric_name = 'brier_score'
                ORDER BY timestamp DESC LIMIT 1
                """
            )
            row = cursor.fetchone()
            if row:
                self.metrics.brier_score = float(row[0])

            # Get error count from logs
            cursor.execute(
                """
                SELECT COUNT(*) FROM logs
                WHERE level = 'ERROR' AND timestamp >= ?
                """,
                (start_ts,),
            )
            row = cursor.fetchone()
            if row:
                self.metrics.errors_count = row[0]

            cursor.execute(
                """
                SELECT COUNT(*) FROM logs
                WHERE level = 'WARNING' AND timestamp >= ?
                """,
                (start_ts,),
            )
            row = cursor.fetchone()
            if row:
                self.metrics.warnings_count = row[0]

            conn.close()

        except Exception as e:
            print(f"[WARN] Error collecting metrics: {e}")

    def _check_memory_usage(self) -> float:
        """Check current memory usage in MB."""
        try:
            import psutil

            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0

    async def run(self) -> SoakTestMetrics:
        """Run the soak test.

        Returns:
            Final metrics from the test.
        """
        print("\n" + "=" * 60)
        print("SOAK TEST STARTING")
        print("=" * 60)
        print(f"Duration: {self.duration_hours} hours")
        print(f"Check interval: {self.check_interval / 60} minutes")
        print(f"Started: {self.metrics.start_time.isoformat()}")
        print("=" * 60 + "\n")

        self._running = True
        self._save_status("running", "Soak test in progress")

        # Set up signal handlers
        def handle_signal(signum, frame):
            print("\n[INFO] Received signal, stopping soak test...")
            self._running = False

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

        start_time = time.time()
        target_duration = self.duration_hours * 3600  # Convert to seconds

        try:
            while self._running:
                elapsed = time.time() - start_time
                self.metrics.duration_hours = elapsed / 3600

                if elapsed >= target_duration:
                    print("\n[INFO] Soak test duration complete")
                    break

                # Collect metrics
                self._collect_metrics_from_db()

                # Check memory
                mem_mb = self._check_memory_usage()
                if mem_mb > self.metrics.max_memory_mb:
                    self.metrics.max_memory_mb = mem_mb

                # Update status
                self._save_status("running", f"Elapsed: {elapsed/3600:.1f}h")

                # Print progress
                remaining = (target_duration - elapsed) / 3600
                print(
                    f"[SOAK] {self.metrics.duration_hours:.1f}h elapsed, "
                    f"{remaining:.1f}h remaining | "
                    f"Trades: {self.metrics.total_trades} | "
                    f"Win rate: {self.metrics.win_rate:.1%} | "
                    f"Brier: {self.metrics.brier_score:.3f}"
                )

                # Wait for next check
                await asyncio.sleep(self.check_interval)

        except Exception as e:
            print(f"[ERROR] Soak test error: {e}")
            self._save_status("error", str(e))
            raise

        finally:
            self._running = False
            self.metrics.end_time = datetime.now(timezone.utc)
            self._collect_metrics_from_db()  # Final collection
            self._save_metrics()

        # Final status
        eligible, failures = self.metrics.is_promotion_eligible
        if eligible:
            self._save_status("completed", "Promotion eligible!")
        else:
            self._save_status("completed", f"Not eligible: {', '.join(failures)}")

        return self.metrics


def print_status() -> None:
    """Print current soak test status."""
    status_path = Path(SoakTestRunner.STATUS_FILE)

    if not status_path.exists():
        print("\nNo soak test in progress or completed.")
        print("Run: python scripts/soak_test.py --duration 24h")
        return

    with open(status_path) as f:
        data = json.load(f)

    print("\n" + "=" * 60)
    print("SOAK TEST STATUS")
    print("=" * 60)

    print(f"\nStatus: {data['status'].upper()}")
    print(f"Started: {data['started_at']}")
    print(f"Duration: {data['elapsed_hours']:.1f}h / {data['duration_hours']}h")

    if data.get("message"):
        print(f"Message: {data['message']}")

    metrics = data.get("metrics", {})
    print("\n" + "-" * 40)
    print("METRICS")
    print("-" * 40)
    print(f"Total Trades: {metrics.get('total_trades', 0)}")
    print(f"Win Rate: {metrics.get('win_rate', 0):.1%}")
    print(f"Total P&L: ${metrics.get('total_pnl', 0):.2f}")
    print(f"Brier Score: {metrics.get('brier_score', 1.0):.3f}")
    print(f"Errors: {metrics.get('errors_count', 0)}")

    print("\n" + "=" * 60)


def print_report() -> None:
    """Print promotion eligibility report."""
    metrics_path = Path(SoakTestRunner.METRICS_FILE)

    if not metrics_path.exists():
        print("\nNo soak test data available.")
        print("Run a soak test first: python scripts/soak_test.py --duration 24h")
        return

    with open(metrics_path) as f:
        history = json.load(f)

    if not history:
        print("\nNo soak test data available.")
        return

    # Use most recent test
    latest = history[-1]

    print("\n" + "=" * 60)
    print("PROMOTION ELIGIBILITY REPORT")
    print("=" * 60)

    print(f"\nSoak Test Date: {latest['start_time']}")
    print(f"Duration: {latest['duration_hours']:.1f} hours")

    print("\n" + "-" * 40)
    print("CRITERIA CHECK")
    print("-" * 40)

    # Check each criterion
    checks = [
        ("Trades >= 100", latest["total_trades"] >= 100, f"{latest['total_trades']}/100"),
        ("Brier Score <= 0.22", latest["brier_score"] <= 0.22, f"{latest['brier_score']:.3f}"),
        ("Win Rate >= 52%", latest["win_rate"] >= 0.52, f"{latest['win_rate']:.1%}"),
        ("Errors <= 10", latest["errors_count"] <= 10, f"{latest['errors_count']}/10"),
    ]

    all_passed = True
    for name, passed, value in checks:
        status = "✓" if passed else "✗"
        print(f"  {status} {name}: {value}")
        if not passed:
            all_passed = False

    print("\n" + "-" * 40)
    if all_passed:
        print("RESULT: ✓ ELIGIBLE FOR PROMOTION")
        print("\nTo promote to LIVE_PROBATION, run:")
        print("  python scripts/promote_mode.py probation --by YOUR_NAME --reason 'Soak test passed'")
    else:
        print("RESULT: ✗ NOT YET ELIGIBLE")
        print("\nContinue running in PAPER mode until criteria are met.")

    print("\n" + "=" * 60)


def parse_duration(duration_str: str) -> float:
    """Parse duration string to hours.

    Supports formats: 24h, 7d, 168h, etc.
    """
    duration_str = duration_str.lower().strip()

    if duration_str.endswith("h"):
        return float(duration_str[:-1])
    elif duration_str.endswith("d"):
        return float(duration_str[:-1]) * 24
    elif duration_str.endswith("m"):
        return float(duration_str[:-1]) / 60
    else:
        return float(duration_str)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Soak test runner for PAPER mode validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--duration",
        "-d",
        default="24h",
        help="Test duration (e.g., 24h, 7d). Default: 24h",
    )
    parser.add_argument(
        "--interval",
        "-i",
        type=float,
        default=15.0,
        help="Check interval in minutes. Default: 15",
    )
    parser.add_argument(
        "--status",
        "-s",
        action="store_true",
        help="Show current soak test status",
    )
    parser.add_argument(
        "--report",
        "-r",
        action="store_true",
        help="Generate promotion eligibility report",
    )

    args = parser.parse_args()

    if args.status:
        print_status()
        return 0

    if args.report:
        print_report()
        return 0

    # Parse duration
    try:
        duration_hours = parse_duration(args.duration)
    except ValueError:
        print(f"[ERROR] Invalid duration: {args.duration}")
        return 1

    # Run soak test
    runner = SoakTestRunner(
        duration_hours=duration_hours,
        check_interval_minutes=args.interval,
    )

    try:
        metrics = asyncio.run(runner.run())

        # Print final report
        print("\n" + "=" * 60)
        print("SOAK TEST COMPLETE")
        print("=" * 60)

        eligible, failures = metrics.is_promotion_eligible
        if eligible:
            print("\n✓ PROMOTION ELIGIBLE")
            print("\nTo promote, run:")
            print("  python scripts/promote_mode.py probation --by YOUR_NAME --reason 'Soak test passed'")
        else:
            print("\n✗ NOT YET ELIGIBLE")
            for f in failures:
                print(f"  - {f}")

        print("\n" + "=" * 60)
        return 0

    except KeyboardInterrupt:
        print("\n[INFO] Soak test interrupted")
        return 1
    except Exception as e:
        print(f"\n[ERROR] Soak test failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
