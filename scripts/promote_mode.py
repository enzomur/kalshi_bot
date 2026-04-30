#!/usr/bin/env python3
"""Script to promote trading mode with cryptographic signatures.

This script generates valid signed configurations for LIVE trading modes.
It performs safety checks before allowing mode promotion.

Usage:
    # Check current status and eligibility
    python scripts/promote_mode.py status

    # Promote to LIVE_PROBATION (requires 100+ trades, Brier < 0.22)
    python scripts/promote_mode.py probation --by "username" --reason "Testing live"

    # Promote to LIVE_FULL (requires 7+ days probation, continued performance)
    python scripts/promote_mode.py full --by "username" --reason "Ready for full live"

    # Downgrade to PAPER (always allowed)
    python scripts/promote_mode.py paper --by "username" --reason "Pausing live trading"

Promotion Criteria:
    PAPER → LIVE_PROBATION:
        - 100+ paper trades completed
        - Brier score < 0.22
        - Win rate > 52%
        - Max drawdown < 15%
        - secrets/kalshi.pem exists

    LIVE_PROBATION → LIVE_FULL:
        - 7+ days in probation mode
        - 50+ additional trades
        - Brier score < 0.20
        - Win rate > 54%
        - Max drawdown < 10%
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from src.core.mode import ModeManager
from src.core.types import TradingMode


# Promotion criteria thresholds
PROBATION_MIN_TRADES = 100
PROBATION_MAX_BRIER = 0.22
PROBATION_MIN_WIN_RATE = 0.52
PROBATION_MAX_DRAWDOWN = 0.15

FULL_MIN_TRADES = 50  # Additional trades in probation
FULL_MAX_BRIER = 0.20
FULL_MIN_WIN_RATE = 0.54
FULL_MAX_DRAWDOWN = 0.10
FULL_MIN_DAYS = 7


@dataclass
class PerformanceStats:
    """Performance statistics for criteria checking."""

    total_trades: int = 0
    winning_trades: int = 0
    brier_score: float = 1.0
    max_drawdown: float = 0.0
    days_in_mode: int = 0
    current_mode: TradingMode = TradingMode.PAPER
    mode_activated_at: str | None = None

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades


def load_performance_stats() -> PerformanceStats:
    """Load performance statistics from database."""
    stats = PerformanceStats()

    # Get current mode info
    manager = ModeManager()
    config = manager.load_config()
    stats.current_mode = config.mode
    stats.mode_activated_at = config.activated_at

    if config.activated_at:
        try:
            activated = datetime.fromisoformat(
                config.activated_at.replace("Z", "+00:00")
            )
            stats.days_in_mode = (datetime.now(timezone.utc) - activated).days
        except (ValueError, TypeError):
            stats.days_in_mode = 0

    # Load from database
    db_path = Path("data/kalshi_bot.db")
    if not db_path.exists():
        return stats

    try:
        import sqlite3

        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Get trade count and win rate
        cursor.execute(
            """
            SELECT COUNT(*), SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END)
            FROM trades WHERE status = 'settled'
            """
        )
        row = cursor.fetchone()
        if row and row[0]:
            stats.total_trades = row[0]
            stats.winning_trades = row[1] or 0

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
            stats.brier_score = float(row[0])

        # Get max drawdown
        cursor.execute(
            """
            SELECT value FROM metrics
            WHERE metric_name = 'max_drawdown'
            ORDER BY timestamp DESC LIMIT 1
            """
        )
        row = cursor.fetchone()
        if row:
            stats.max_drawdown = float(row[0])

        conn.close()
    except Exception as e:
        print(f"[WARN] Error loading stats: {e}")

    return stats


def print_status() -> None:
    """Print current status and eligibility."""
    stats = load_performance_stats()

    print("\n" + "=" * 60)
    print("TRADING MODE STATUS")
    print("=" * 60)

    print(f"\nCurrent Mode: {stats.current_mode.value.upper()}")
    if stats.mode_activated_at:
        print(f"Activated: {stats.mode_activated_at}")
        print(f"Days in Mode: {stats.days_in_mode}")

    print("\n" + "-" * 40)
    print("PERFORMANCE METRICS")
    print("-" * 40)
    print(f"Total Trades: {stats.total_trades}")
    print(f"Win Rate: {stats.win_rate:.1%}")
    print(f"Brier Score: {stats.brier_score:.3f}")
    print(f"Max Drawdown: {stats.max_drawdown:.1%}")

    print("\n" + "-" * 40)
    print("PROMOTION ELIGIBILITY")
    print("-" * 40)

    # Check probation eligibility
    if stats.current_mode == TradingMode.PAPER:
        passed, failures = check_probation_criteria(stats)
        status = "✓ ELIGIBLE" if passed else "✗ NOT ELIGIBLE"
        print(f"\nLIVE_PROBATION: {status}")
        if failures:
            for f in failures:
                print(f"  - {f}")

    # Check full eligibility
    if stats.current_mode in (TradingMode.PAPER, TradingMode.LIVE_PROBATION):
        passed, failures = check_full_criteria(stats)
        status = "✓ ELIGIBLE" if passed else "✗ NOT ELIGIBLE"
        print(f"\nLIVE_FULL: {status}")
        if failures:
            for f in failures:
                print(f"  - {f}")

    print("\n" + "=" * 60)


def check_probation_criteria(
    stats: PerformanceStats | None = None,
) -> tuple[bool, list[str]]:
    """Check if criteria for LIVE_PROBATION are met.

    Criteria:
    - 100+ paper trades completed
    - Brier score < 0.22
    - Win rate > 52%
    - Max drawdown < 15%
    - secrets/kalshi.pem exists

    Returns:
        Tuple of (passed, list of reasons if failed)
    """
    if stats is None:
        stats = load_performance_stats()

    failures: list[str] = []

    # Check for secrets
    if not Path("secrets/kalshi.pem").exists():
        failures.append("Missing secrets/kalshi.pem - API key not configured")

    # Check trade count
    if stats.total_trades < PROBATION_MIN_TRADES:
        failures.append(
            f"Trades: {stats.total_trades}/{PROBATION_MIN_TRADES} required"
        )

    # Check Brier score
    if stats.brier_score > PROBATION_MAX_BRIER:
        failures.append(
            f"Brier score: {stats.brier_score:.3f} > {PROBATION_MAX_BRIER} max"
        )

    # Check win rate
    if stats.win_rate < PROBATION_MIN_WIN_RATE:
        failures.append(
            f"Win rate: {stats.win_rate:.1%} < {PROBATION_MIN_WIN_RATE:.0%} required"
        )

    # Check drawdown
    if stats.max_drawdown > PROBATION_MAX_DRAWDOWN:
        failures.append(
            f"Max drawdown: {stats.max_drawdown:.1%} > {PROBATION_MAX_DRAWDOWN:.0%} limit"
        )

    return len(failures) == 0, failures


def check_full_criteria(
    stats: PerformanceStats | None = None,
) -> tuple[bool, list[str]]:
    """Check if criteria for LIVE_FULL are met.

    Criteria:
    - Currently in LIVE_PROBATION
    - 7+ days in probation
    - 50+ additional trades
    - Brier score < 0.20
    - Win rate > 54%
    - Max drawdown < 10%

    Returns:
        Tuple of (passed, list of reasons if failed)
    """
    if stats is None:
        stats = load_performance_stats()

    failures: list[str] = []

    # Check current mode
    if stats.current_mode != TradingMode.LIVE_PROBATION:
        failures.append(
            f"Must be in LIVE_PROBATION first (currently: {stats.current_mode.value})"
        )

    # Check days in probation
    if stats.days_in_mode < FULL_MIN_DAYS:
        failures.append(
            f"Days in probation: {stats.days_in_mode}/{FULL_MIN_DAYS} required"
        )

    # Check trade count (additional trades in probation)
    if stats.total_trades < FULL_MIN_TRADES:
        failures.append(
            f"Trades in probation: {stats.total_trades}/{FULL_MIN_TRADES} required"
        )

    # Check Brier score
    if stats.brier_score > FULL_MAX_BRIER:
        failures.append(
            f"Brier score: {stats.brier_score:.3f} > {FULL_MAX_BRIER} max"
        )

    # Check win rate
    if stats.win_rate < FULL_MIN_WIN_RATE:
        failures.append(
            f"Win rate: {stats.win_rate:.1%} < {FULL_MIN_WIN_RATE:.0%} required"
        )

    # Check drawdown
    if stats.max_drawdown > FULL_MAX_DRAWDOWN:
        failures.append(
            f"Max drawdown: {stats.max_drawdown:.1%} > {FULL_MAX_DRAWDOWN:.0%} limit"
        )

    return len(failures) == 0, failures


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Promote trading mode with cryptographic signatures"
    )
    parser.add_argument(
        "mode",
        choices=["paper", "probation", "full", "status"],
        help="Target mode or 'status' to check eligibility",
    )
    parser.add_argument(
        "--by",
        help="Username or identifier of who is promoting (required for promotion)",
    )
    parser.add_argument(
        "--reason",
        help="Reason for mode change (required for promotion)",
    )
    parser.add_argument(
        "--max-position",
        type=float,
        help="Maximum position in dollars (optional override)",
    )
    parser.add_argument(
        "--max-daily-loss",
        type=float,
        help="Maximum daily loss in dollars (optional override)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip criteria checks (dangerous!)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print config without writing",
    )

    args = parser.parse_args()

    # Handle status command
    if args.mode == "status":
        print_status()
        return 0

    # Require --by and --reason for promotion
    if not args.by or not args.reason:
        print("ERROR: --by and --reason are required for mode promotion")
        return 1

    # Map mode argument to TradingMode
    mode_map = {
        "paper": TradingMode.PAPER,
        "probation": TradingMode.LIVE_PROBATION,
        "full": TradingMode.LIVE_FULL,
    }
    target_mode = mode_map[args.mode]

    print(f"Mode promotion request: {target_mode.value}")
    print(f"  By: {args.by}")
    print(f"  Reason: {args.reason}")
    print()

    # Run criteria checks for LIVE modes
    if target_mode == TradingMode.LIVE_PROBATION and not args.force:
        passed, failures = check_probation_criteria()
        if not passed:
            print("FAILED: Criteria for LIVE_PROBATION not met:")
            for f in failures:
                print(f"  - {f}")
            print()
            print("Use --force to bypass checks (not recommended)")
            return 1

    elif target_mode == TradingMode.LIVE_FULL and not args.force:
        passed, failures = check_full_criteria()
        if not passed:
            print("FAILED: Criteria for LIVE_FULL not met:")
            for f in failures:
                print(f"  - {f}")
            print()
            print("Use --force to bypass checks (not recommended)")
            return 1

    # Generate signed configuration
    manager = ModeManager()
    config = manager.create_signed_config(
        mode=target_mode,
        activated_by=args.by,
        reason=args.reason,
        max_position_dollars=args.max_position,
        max_daily_loss_dollars=args.max_daily_loss,
    )

    print("Generated configuration:")
    print("-" * 40)
    print(yaml.dump(config, default_flow_style=False))
    print("-" * 40)

    if args.dry_run:
        print("DRY RUN: Configuration not written")
        return 0

    # Write configuration
    config_path = Path("config/mode.yaml")
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"Configuration written to {config_path}")

    # Verify the configuration loads correctly
    manager = ModeManager()
    loaded = manager.load_config()

    if loaded.mode != target_mode:
        print(f"ERROR: Mode verification failed! Got {loaded.mode.value}")
        return 1

    print(f"SUCCESS: Mode promoted to {target_mode.value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
