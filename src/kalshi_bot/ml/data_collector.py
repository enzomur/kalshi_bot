"""Market snapshot collector for ML training data."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import TYPE_CHECKING

from kalshi_bot.utils.logging import get_logger

if TYPE_CHECKING:
    from kalshi_bot.api.client import KalshiAPIClient
    from kalshi_bot.persistence.database import Database

logger = get_logger(__name__)


class MarketSnapshotCollector:
    """
    Collects market price snapshots at regular intervals for ML training.

    Polls all active markets every interval (default 5 minutes) and stores
    snapshots in the database for later use in feature engineering and
    model training.
    """

    def __init__(
        self,
        db: Database,
        api_client: KalshiAPIClient,
        interval_seconds: int = 300,  # 5 minutes
        min_volume: int = 0,
        min_open_interest: int = 0,
    ) -> None:
        """
        Initialize the snapshot collector.

        Args:
            db: Database connection
            api_client: Kalshi API client
            interval_seconds: Seconds between snapshot collections
            min_volume: Minimum volume filter for markets
            min_open_interest: Minimum open interest filter
        """
        self._db = db
        self._api_client = api_client
        self._interval_seconds = interval_seconds
        self._min_volume = min_volume
        self._min_open_interest = min_open_interest

        self._running = False
        self._last_collection: datetime | None = None
        self._total_snapshots = 0
        self._collection_errors = 0

    async def collect_snapshots(self) -> int:
        """
        Collect snapshots for all active markets.

        Returns:
            Number of snapshots collected
        """
        try:
            # Fetch all open markets
            markets = await self._api_client.get_all_markets(
                status="open",
                min_volume=self._min_volume,
                min_open_interest=self._min_open_interest,
                require_orderbook=False,
                max_pages=100,
            )

            if not markets:
                logger.debug("No markets found for snapshot collection")
                return 0

            # Prepare batch insert
            snapshot_time = datetime.utcnow()
            snapshots = []

            for market in markets:
                # Calculate spread if we have bid/ask
                spread = None
                if market.yes_bid is not None and market.yes_ask is not None:
                    spread = market.yes_ask - market.yes_bid

                snapshots.append((
                    market.ticker,
                    market.event_ticker,
                    market.yes_bid,
                    market.yes_ask,
                    market.no_bid,
                    market.no_ask,
                    market.last_price,
                    market.volume,
                    market.open_interest,
                    spread,
                    market.status,
                    market.close_time.isoformat() if market.close_time else None,
                    market.expiration_time.isoformat() if market.expiration_time else None,
                    snapshot_time.isoformat(),
                ))

            # Batch insert snapshots
            await self._db.execute_many(
                """
                INSERT INTO market_snapshots (
                    ticker, event_ticker,
                    yes_bid, yes_ask, no_bid, no_ask, last_price,
                    volume, open_interest, spread,
                    status, close_time, expiration_time,
                    snapshot_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                snapshots,
            )

            self._last_collection = snapshot_time
            self._total_snapshots += len(snapshots)

            logger.info(
                f"Collected {len(snapshots)} market snapshots "
                f"(total: {self._total_snapshots})"
            )

            return len(snapshots)

        except Exception as e:
            self._collection_errors += 1
            logger.error(f"Snapshot collection error: {e}")
            return 0

    async def start_collection_loop(self, shutdown_event: asyncio.Event) -> None:
        """
        Start the continuous snapshot collection loop.

        Args:
            shutdown_event: Event to signal shutdown
        """
        self._running = True
        logger.info(
            f"Starting snapshot collection loop (interval: {self._interval_seconds}s)"
        )

        while not shutdown_event.is_set():
            await self.collect_snapshots()

            try:
                await asyncio.wait_for(
                    shutdown_event.wait(),
                    timeout=self._interval_seconds,
                )
                break
            except asyncio.TimeoutError:
                pass

        self._running = False
        logger.info("Snapshot collection loop stopped")

    async def get_snapshot_count(self, ticker: str | None = None) -> int:
        """Get total snapshot count, optionally for a specific ticker."""
        if ticker:
            result = await self._db.fetch_one(
                "SELECT COUNT(*) as count FROM market_snapshots WHERE ticker = ?",
                (ticker,),
            )
        else:
            result = await self._db.fetch_one(
                "SELECT COUNT(*) as count FROM market_snapshots"
            )
        return result["count"] if result else 0

    async def get_snapshots_for_market(
        self,
        ticker: str,
        hours_back: int = 24,
        limit: int | None = None,
    ) -> list[dict]:
        """
        Get recent snapshots for a specific market.

        Args:
            ticker: Market ticker
            hours_back: Hours of history to fetch
            limit: Maximum number of snapshots

        Returns:
            List of snapshot dictionaries
        """
        query = """
            SELECT * FROM market_snapshots
            WHERE ticker = ?
              AND snapshot_at >= datetime('now', ?)
            ORDER BY snapshot_at DESC
        """
        params: tuple = (ticker, f"-{hours_back} hours")

        if limit:
            query += " LIMIT ?"
            params = (ticker, f"-{hours_back} hours", limit)

        return await self._db.fetch_all(query, params)

    async def get_unique_tickers(self) -> list[str]:
        """Get all unique tickers with snapshots."""
        results = await self._db.fetch_all(
            "SELECT DISTINCT ticker FROM market_snapshots ORDER BY ticker"
        )
        return [r["ticker"] for r in results]

    async def cleanup_old_snapshots(self, days_to_keep: int = 90) -> int:
        """
        Delete snapshots older than specified days.

        Args:
            days_to_keep: Number of days of history to retain

        Returns:
            Number of deleted rows
        """
        cursor = await self._db.execute(
            """
            DELETE FROM market_snapshots
            WHERE snapshot_at < datetime('now', ?)
            """,
            (f"-{days_to_keep} days",),
        )
        deleted = cursor.rowcount
        if deleted > 0:
            logger.info(f"Cleaned up {deleted} old snapshots")
        return deleted

    def get_status(self) -> dict:
        """Get collector status."""
        return {
            "running": self._running,
            "interval_seconds": self._interval_seconds,
            "last_collection": self._last_collection.isoformat() if self._last_collection else None,
            "total_snapshots": self._total_snapshots,
            "collection_errors": self._collection_errors,
            "filters": {
                "min_volume": self._min_volume,
                "min_open_interest": self._min_open_interest,
            },
        }
