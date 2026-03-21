"""Async SQLite database with WAL mode."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import aiosqlite

from kalshi_bot.utils.logging import get_logger

logger = get_logger(__name__)


class Database:
    """Async SQLite database manager with WAL mode."""

    def __init__(self, db_path: str) -> None:
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

        # Check if migration was already run (handle case where bot_state doesn't exist yet)
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
        """
        Execute a query.

        Args:
            query: SQL query
            params: Query parameters

        Returns:
            Cursor with results
        """
        if self._connection is None:
            raise RuntimeError("Database not initialized")

        async with self._lock:
            cursor = await self._connection.execute(query, params or ())
            await self._connection.commit()
            return cursor

    async def execute_many(
        self, query: str, params_list: list[tuple[Any, ...]]
    ) -> None:
        """
        Execute a query with multiple parameter sets.

        Args:
            query: SQL query
            params_list: List of parameter tuples
        """
        if self._connection is None:
            raise RuntimeError("Database not initialized")

        async with self._lock:
            await self._connection.executemany(query, params_list)
            await self._connection.commit()

    async def fetch_one(
        self, query: str, params: tuple[Any, ...] | None = None
    ) -> dict[str, Any] | None:
        """
        Fetch a single row as dictionary.

        Args:
            query: SQL query
            params: Query parameters

        Returns:
            Row as dictionary or None
        """
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
        """
        Fetch all rows as dictionaries.

        Args:
            query: SQL query
            params: Query parameters

        Returns:
            List of rows as dictionaries
        """
        if self._connection is None:
            raise RuntimeError("Database not initialized")

        self._connection.row_factory = aiosqlite.Row
        cursor = await self._connection.execute(query, params or ())
        rows = await cursor.fetchall()

        return [dict(row) for row in rows]

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
            "INSERT OR REPLACE INTO bot_state (key, value, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP)",
            (key, value),
        )

    async def transaction(self) -> aiosqlite.Connection:
        """
        Get connection for manual transaction management.

        Usage:
            async with db.transaction() as conn:
                await conn.execute(...)
                await conn.execute(...)
                await conn.commit()
        """
        if self._connection is None:
            raise RuntimeError("Database not initialized")
        return self._connection

    @property
    def is_connected(self) -> bool:
        """Check if database is connected."""
        return self._connection is not None
