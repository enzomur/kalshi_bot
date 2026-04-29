"""Async SQLite database with WAL mode for the ledger.

This module provides the persistence layer for:
- Signals and their outcomes
- Risk decisions
- Orders and fills
- Positions
- Mode transitions
- Performance tracking
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite

from src.core.types import Signal, Order, Fill, Position, Side
from src.observability.logging import get_logger

logger = get_logger(__name__)


class Database:
    """Async SQLite database manager with WAL mode."""

    def __init__(self, db_path: str = "data/ledger.db") -> None:
        """
        Initialize database manager.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._connection: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize database and run migrations."""
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

        self._connection = await aiosqlite.connect(self.db_path)

        # Enable WAL mode and optimizations
        await self._connection.execute("PRAGMA journal_mode=WAL")
        await self._connection.execute("PRAGMA synchronous=NORMAL")
        await self._connection.execute("PRAGMA cache_size=10000")
        await self._connection.execute("PRAGMA foreign_keys=ON")

        await self._run_migrations()

        logger.info(f"Database initialized: {self.db_path}")

    async def _run_migrations(self) -> None:
        """Run database migrations."""
        migrations_dir = Path(__file__).parent / "migrations"

        if not migrations_dir.exists():
            logger.warning(f"Migrations directory not found: {migrations_dir}")
            return

        migration_files = sorted(migrations_dir.glob("*.sql"))

        for migration_file in migration_files:
            await self._run_migration(migration_file)

    async def _run_migration(self, migration_file: Path) -> None:
        """Run a single migration file."""
        if self._connection is None:
            raise RuntimeError("Database not initialized")

        migration_name = migration_file.stem

        # Check if migration was already run
        try:
            cursor = await self._connection.execute(
                "SELECT value FROM bot_state WHERE key = ?",
                (f"migration:{migration_name}",),
            )
            row = await cursor.fetchone()

            if row is not None:
                return
        except aiosqlite.OperationalError:
            # bot_state table doesn't exist yet, migration needs to run
            pass

        logger.info(f"Running migration: {migration_name}")

        sql = migration_file.read_text()

        statements = [s.strip() for s in sql.split(";") if s.strip()]

        for statement in statements:
            if statement:
                try:
                    await self._connection.execute(statement)
                except aiosqlite.OperationalError as e:
                    if "already exists" not in str(e):
                        raise

        try:
            await self._connection.execute(
                "INSERT OR REPLACE INTO bot_state (key, value) VALUES (?, ?)",
                (f"migration:{migration_name}", "completed"),
            )
        except aiosqlite.OperationalError:
            pass

        await self._connection.commit()
        logger.info(f"Migration completed: {migration_name}")

    async def close(self) -> None:
        """Close database connection."""
        if self._connection is not None:
            await self._connection.close()
            self._connection = None
            logger.info("Database connection closed")

    async def execute(
        self, query: str, params: tuple[Any, ...] | None = None
    ) -> aiosqlite.Cursor:
        """Execute a query."""
        if self._connection is None:
            raise RuntimeError("Database not initialized")

        async with self._lock:
            cursor = await self._connection.execute(query, params or ())
            await self._connection.commit()
            return cursor

    async def execute_many(
        self, query: str, params_list: list[tuple[Any, ...]]
    ) -> None:
        """Execute a query with multiple parameter sets."""
        if self._connection is None:
            raise RuntimeError("Database not initialized")

        async with self._lock:
            await self._connection.executemany(query, params_list)
            await self._connection.commit()

    async def fetch_one(
        self, query: str, params: tuple[Any, ...] | None = None
    ) -> dict[str, Any] | None:
        """Fetch a single row as dictionary."""
        if self._connection is None:
            raise RuntimeError("Database not initialized")

        self._connection.row_factory = aiosqlite.Row
        cursor = await self._connection.execute(query, params or ())
        row = await cursor.fetchone()

        if row is None:
            return None

        return dict(row)

    async def fetch_all(
        self, query: str, params: tuple[Any, ...] | None = None
    ) -> list[dict[str, Any]]:
        """Fetch all rows as dictionaries."""
        if self._connection is None:
            raise RuntimeError("Database not initialized")

        self._connection.row_factory = aiosqlite.Row
        cursor = await self._connection.execute(query, params or ())
        rows = await cursor.fetchall()

        return [dict(row) for row in rows]

    # Signal operations

    async def save_signal(self, signal: Signal) -> None:
        """Save a signal to the database."""
        await self.execute(
            """
            INSERT INTO signals (
                signal_id, strategy_name, market_ticker, direction,
                target_probability, confidence, edge, max_position,
                metadata, created_at, expires_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                signal.signal_id,
                signal.strategy_name,
                signal.market_ticker,
                signal.direction,
                signal.target_probability,
                signal.confidence,
                signal.edge,
                signal.max_position,
                json.dumps(signal.metadata),
                signal.created_at.isoformat(),
                signal.expires_at.isoformat() if signal.expires_at else None,
            ),
        )

    async def update_signal_outcome(
        self,
        signal_id: str,
        actual_outcome: str,
        was_correct: bool,
    ) -> None:
        """Update signal with settlement outcome."""
        await self.execute(
            """
            UPDATE signals
            SET actual_outcome = ?,
                settled_at = ?,
                was_correct = ?
            WHERE signal_id = ?
            """,
            (
                actual_outcome,
                datetime.now(timezone.utc).isoformat(),
                1 if was_correct else 0,
                signal_id,
            ),
        )

    async def get_unsettled_signals(self) -> list[dict[str, Any]]:
        """Get all signals that haven't been settled yet."""
        return await self.fetch_all(
            """
            SELECT * FROM signals
            WHERE settled_at IS NULL
            ORDER BY created_at DESC
            """
        )

    # Risk decision operations

    async def save_risk_decision(
        self,
        decision_id: str,
        signal_id: str,
        approved: bool,
        approved_size: int,
        rejection_reason: str | None,
        circuit_breaker_status: dict[str, Any],
        position_check: dict[str, Any],
        kelly_calculation: dict[str, Any],
        mode_caps: dict[str, Any],
        trading_mode: str,
    ) -> None:
        """Save a risk decision to the database."""
        await self.execute(
            """
            INSERT INTO risk_decisions (
                decision_id, signal_id, approved, approved_size,
                rejection_reason, circuit_breaker_status, position_check,
                kelly_calculation, mode_caps, trading_mode
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                decision_id,
                signal_id,
                1 if approved else 0,
                approved_size,
                rejection_reason,
                json.dumps(circuit_breaker_status),
                json.dumps(position_check),
                json.dumps(kelly_calculation),
                json.dumps(mode_caps),
                trading_mode,
            ),
        )

    # Order operations

    async def save_order(self, order: Order, decision_id: str | None, is_paper: bool) -> None:
        """Save an order to the database."""
        await self.execute(
            """
            INSERT INTO orders (
                order_id, decision_id, market_ticker, side, order_type,
                price, quantity, status, filled_quantity, remaining_quantity,
                is_paper
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                order.order_id,
                decision_id,
                order.market_ticker,
                order.side.value,
                order.order_type.value,
                order.price,
                order.quantity,
                order.status.value,
                order.filled_quantity,
                order.remaining_quantity,
                1 if is_paper else 0,
            ),
        )

    async def update_order_status(
        self,
        order_id: str,
        status: str,
        filled_quantity: int,
        remaining_quantity: int,
    ) -> None:
        """Update order status."""
        await self.execute(
            """
            UPDATE orders
            SET status = ?, filled_quantity = ?, remaining_quantity = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE order_id = ?
            """,
            (status, filled_quantity, remaining_quantity, order_id),
        )

    # Fill operations

    async def save_fill(self, fill: Fill, is_paper: bool) -> None:
        """Save a fill to the database."""
        await self.execute(
            """
            INSERT INTO fills (
                fill_id, order_id, market_ticker, side, price,
                quantity, fee, is_paper, executed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                fill.fill_id,
                fill.order_id,
                fill.market_ticker,
                fill.side.value,
                fill.price,
                fill.quantity,
                fill.fee,
                1 if is_paper else 0,
                fill.executed_at.isoformat(),
            ),
        )

    # Position operations

    async def update_position(
        self,
        market_ticker: str,
        side: Side,
        quantity: int,
        average_price: float,
        market_exposure: float,
        is_paper: bool,
    ) -> None:
        """Update or insert a position."""
        await self.execute(
            """
            INSERT INTO positions (
                market_ticker, side, quantity, average_price,
                market_exposure, is_paper
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (market_ticker, side, is_paper) DO UPDATE SET
                quantity = excluded.quantity,
                average_price = excluded.average_price,
                market_exposure = excluded.market_exposure,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                market_ticker,
                side.value,
                quantity,
                average_price,
                market_exposure,
                1 if is_paper else 0,
            ),
        )

    async def get_positions(self, is_paper: bool = True) -> list[dict[str, Any]]:
        """Get all positions."""
        return await self.fetch_all(
            """
            SELECT * FROM positions
            WHERE is_paper = ? AND quantity > 0
            ORDER BY market_ticker
            """,
            (1 if is_paper else 0,),
        )

    async def get_position(
        self, market_ticker: str, side: Side, is_paper: bool = True
    ) -> dict[str, Any] | None:
        """Get a specific position."""
        return await self.fetch_one(
            """
            SELECT * FROM positions
            WHERE market_ticker = ? AND side = ? AND is_paper = ?
            """,
            (market_ticker, side.value, 1 if is_paper else 0),
        )

    # Mode transition operations

    async def log_mode_transition(
        self,
        from_mode: str,
        to_mode: str,
        activated_by: str,
        reason: str | None,
        signature_valid: bool,
    ) -> None:
        """Log a mode transition."""
        await self.execute(
            """
            INSERT INTO mode_transitions (
                from_mode, to_mode, activated_by, reason, signature_valid
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                from_mode,
                to_mode,
                activated_by,
                reason,
                1 if signature_valid else 0,
            ),
        )

    # Daily P&L operations

    async def update_daily_pnl(
        self,
        realized_pnl_delta: float,
        trade_result: str | None,  # "win", "loss", or None
        is_paper: bool = True,
    ) -> None:
        """Update daily P&L tracking."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # Ensure record exists
        await self.execute(
            """
            INSERT OR IGNORE INTO daily_pnl (date, is_paper)
            VALUES (?, ?)
            """,
            (today, 1 if is_paper else 0),
        )

        # Update P&L
        if trade_result == "win":
            await self.execute(
                """
                UPDATE daily_pnl
                SET realized_pnl = realized_pnl + ?,
                    total_trades = total_trades + 1,
                    winning_trades = winning_trades + 1,
                    updated_at = CURRENT_TIMESTAMP
                WHERE date = ? AND is_paper = ?
                """,
                (realized_pnl_delta, today, 1 if is_paper else 0),
            )
        elif trade_result == "loss":
            await self.execute(
                """
                UPDATE daily_pnl
                SET realized_pnl = realized_pnl + ?,
                    total_trades = total_trades + 1,
                    losing_trades = losing_trades + 1,
                    updated_at = CURRENT_TIMESTAMP
                WHERE date = ? AND is_paper = ?
                """,
                (realized_pnl_delta, today, 1 if is_paper else 0),
            )
        else:
            await self.execute(
                """
                UPDATE daily_pnl
                SET realized_pnl = realized_pnl + ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE date = ? AND is_paper = ?
                """,
                (realized_pnl_delta, today, 1 if is_paper else 0),
            )

    async def get_daily_pnl(self, is_paper: bool = True) -> dict[str, Any] | None:
        """Get today's P&L."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return await self.fetch_one(
            """
            SELECT * FROM daily_pnl
            WHERE date = ? AND is_paper = ?
            """,
            (today, 1 if is_paper else 0),
        )

    # Strategy performance operations

    async def update_strategy_performance(
        self,
        strategy_name: str,
        signal_approved: bool = False,
        signal_executed: bool = False,
        signal_settled: bool = False,
        signal_correct: bool = False,
        pnl_delta: float = 0.0,
    ) -> None:
        """Update strategy performance metrics."""
        # Ensure record exists
        await self.execute(
            """
            INSERT OR IGNORE INTO strategy_performance (strategy_name)
            VALUES (?)
            """,
            (strategy_name,),
        )

        updates = ["total_signals = total_signals + 1", "last_signal_at = CURRENT_TIMESTAMP"]

        if signal_approved:
            updates.append("approved_signals = approved_signals + 1")
        if signal_executed:
            updates.append("executed_signals = executed_signals + 1")
        if signal_settled:
            updates.append("settled_signals = settled_signals + 1")
        if signal_correct:
            updates.append("correct_signals = correct_signals + 1")
        if pnl_delta != 0:
            updates.append(f"total_pnl = total_pnl + {pnl_delta}")

        # Update accuracy
        updates.append(
            "accuracy = CASE WHEN settled_signals > 0 "
            "THEN CAST(correct_signals AS REAL) / settled_signals "
            "ELSE NULL END"
        )

        updates.append("updated_at = CURRENT_TIMESTAMP")

        await self.execute(
            f"""
            UPDATE strategy_performance
            SET {", ".join(updates)}
            WHERE strategy_name = ?
            """,
            (strategy_name,),
        )

    async def get_strategy_performance(
        self, strategy_name: str
    ) -> dict[str, Any] | None:
        """Get strategy performance metrics."""
        return await self.fetch_one(
            """
            SELECT * FROM strategy_performance
            WHERE strategy_name = ?
            """,
            (strategy_name,),
        )

    async def get_all_strategy_performance(self) -> list[dict[str, Any]]:
        """Get all strategy performance metrics."""
        return await self.fetch_all(
            """
            SELECT * FROM strategy_performance
            ORDER BY total_pnl DESC
            """
        )

    # State operations

    async def get_state(self, key: str) -> str | None:
        """Get a value from bot_state."""
        row = await self.fetch_one(
            "SELECT value FROM bot_state WHERE key = ?",
            (key,),
        )
        return row["value"] if row else None

    async def set_state(self, key: str, value: str) -> None:
        """Set a value in bot_state."""
        await self.execute(
            """
            INSERT OR REPLACE INTO bot_state (key, value, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            """,
            (key, value),
        )

    @property
    def is_connected(self) -> bool:
        """Check if database is connected."""
        return self._connection is not None
