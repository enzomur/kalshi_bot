#!/usr/bin/env python3
"""Script to promote trading mode with cryptographic signatures.

This script generates valid signed configurations for LIVE trading modes.
It performs safety checks before allowing mode promotion.

Usage:
    # Promote to LIVE_PROBATION (requires passing criteria)
    python scripts/promote_mode.py probation --by "username" --reason "Testing live"

    # Promote to LIVE_FULL (requires proven track record in probation)
    python scripts/promote_mode.py full --by "username" --reason "Ready for full live"

    # Downgrade to PAPER (always allowed)
    python scripts/promote_mode.py paper --by "username" --reason "Pausing live trading"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from src.core.mode import ModeManager
from src.core.types import TradingMode


def check_probation_criteria() -> tuple[bool, list[str]]:
    """Check if criteria for LIVE_PROBATION are met.

    Criteria:
    1. Bot has run successfully in PAPER mode
    2. Positive simulated P&L over test period
    3. No critical errors in recent runs

    Returns:
        Tuple of (passed, list of reasons if failed)
    """
    failures: list[str] = []

    # Check for secrets
    if not Path("secrets/kalshi.pem").exists():
        failures.append("Missing secrets/kalshi.pem - API key not configured")

    if not Path("secrets/.env").exists():
        failures.append("Missing secrets/.env - Environment not configured")

    # TODO: Add more criteria checks from database
    # - Check paper trading history
    # - Check simulated P&L
    # - Check for errors

    return len(failures) == 0, failures


def check_full_criteria() -> tuple[bool, list[str]]:
    """Check if criteria for LIVE_FULL are met.

    Criteria:
    1. Successfully ran in LIVE_PROBATION for minimum period
    2. Positive actual P&L in probation
    3. No risk limit violations
    4. Proven strategy performance

    Returns:
        Tuple of (passed, list of reasons if failed)
    """
    failures: list[str] = []

    # Check current mode
    manager = ModeManager()
    config = manager.load_config()

    if config.mode != TradingMode.LIVE_PROBATION:
        failures.append(
            f"Must be in LIVE_PROBATION mode first (currently: {config.mode.value})"
        )

    # TODO: Add more criteria checks from database
    # - Check probation trading history
    # - Check actual P&L
    # - Check risk metrics
    # - Check minimum time in probation

    return len(failures) == 0, failures


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Promote trading mode with cryptographic signatures"
    )
    parser.add_argument(
        "mode",
        choices=["paper", "probation", "full"],
        help="Target mode: paper, probation (live_probation), or full (live_full)",
    )
    parser.add_argument(
        "--by",
        required=True,
        help="Username or identifier of who is promoting",
    )
    parser.add_argument(
        "--reason",
        required=True,
        help="Reason for mode change",
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
