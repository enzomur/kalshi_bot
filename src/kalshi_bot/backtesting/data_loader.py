"""Historical data loader for backtesting."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from kalshi_bot.core.types import MarketData, OrderBook, OrderBookLevel
from kalshi_bot.utils.logging import get_logger

if TYPE_CHECKING:
    from kalshi_bot.persistence.database import Database

logger = get_logger(__name__)


class HistoricalDataLoader:
    """
    Loads historical market snapshots and settlements from SQLite.

    Provides efficient data access for backtesting:
    - Gets distinct timestamps in date range
    - Loads snapshots for each timestamp
    - Tracks settlements between time steps
    - Converts snapshots to MarketData and OrderBook objects
    """

    def __init__(self, db: Database) -> None:
        """
        Initialize the data loader.

        Args:
            db: Database connection
        """
        self._db = db

    async def get_timestamps_in_range(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> list[datetime]:
        """
        Get all distinct snapshot timestamps in the date range.

        Args:
            start_date: Start of date range
            end_date: End of date range

        Returns:
            List of timestamps sorted chronologically
        """
        rows = await self._db.fetch_all(
            """
            SELECT DISTINCT snapshot_at
            FROM market_snapshots
            WHERE snapshot_at >= ? AND snapshot_at <= ?
            ORDER BY snapshot_at ASC
            """,
            (start_date.isoformat(), end_date.isoformat()),
        )
        return [datetime.fromisoformat(row["snapshot_at"]) for row in rows]

    async def get_date_range(self) -> tuple[datetime | None, datetime | None]:
        """
        Get the earliest and latest snapshot dates available.

        Returns:
            Tuple of (earliest_date, latest_date), or (None, None) if no data
        """
        row = await self._db.fetch_one(
            """
            SELECT MIN(snapshot_at) as earliest, MAX(snapshot_at) as latest
            FROM market_snapshots
            """
        )
        if row and row["earliest"] and row["latest"]:
            return (
                datetime.fromisoformat(row["earliest"]),
                datetime.fromisoformat(row["latest"]),
            )
        return None, None

    async def load_snapshots_at_timestamp(
        self,
        timestamp: datetime,
    ) -> list[dict]:
        """
        Load all market snapshots at a specific timestamp.

        Args:
            timestamp: Exact snapshot timestamp

        Returns:
            List of snapshot dictionaries
        """
        return await self._db.fetch_all(
            """
            SELECT *
            FROM market_snapshots
            WHERE snapshot_at = ?
            """,
            (timestamp.isoformat(),),
        )

    async def get_settlements_in_range(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> list[dict]:
        """
        Get settlements that were confirmed between two timestamps.

        Args:
            start_time: Start of time range
            end_time: End of time range

        Returns:
            List of settlement dictionaries
        """
        return await self._db.fetch_all(
            """
            SELECT *
            FROM market_settlements
            WHERE confirmed_at > ? AND confirmed_at <= ?
            ORDER BY confirmed_at ASC
            """,
            (start_time.isoformat(), end_time.isoformat()),
        )

    async def get_all_settlements_before(
        self,
        end_time: datetime,
    ) -> dict[str, dict]:
        """
        Get all settlements that occurred before a timestamp.

        Returns a dictionary keyed by ticker for fast lookup.

        Args:
            end_time: Get settlements before this time

        Returns:
            Dictionary mapping ticker to settlement info
        """
        rows = await self._db.fetch_all(
            """
            SELECT *
            FROM market_settlements
            WHERE confirmed_at <= ?
            """,
            (end_time.isoformat(),),
        )
        return {row["ticker"]: row for row in rows}

    async def get_settlement_for_ticker(self, ticker: str) -> dict | None:
        """
        Get settlement info for a specific ticker.

        Args:
            ticker: Market ticker

        Returns:
            Settlement dictionary or None if not settled
        """
        return await self._db.fetch_one(
            """
            SELECT *
            FROM market_settlements
            WHERE ticker = ?
            """,
            (ticker,),
        )

    def snapshot_to_market_data(self, snapshot: dict) -> MarketData:
        """
        Convert a snapshot dictionary to a MarketData object.

        Args:
            snapshot: Snapshot from database

        Returns:
            MarketData object
        """
        close_time = None
        if snapshot.get("close_time"):
            try:
                close_time = datetime.fromisoformat(snapshot["close_time"])
            except (ValueError, TypeError):
                pass

        expiration_time = None
        if snapshot.get("expiration_time"):
            try:
                expiration_time = datetime.fromisoformat(snapshot["expiration_time"])
            except (ValueError, TypeError):
                pass

        return MarketData(
            ticker=snapshot["ticker"],
            event_ticker=snapshot["event_ticker"],
            title=snapshot.get("ticker", ""),  # Use ticker as title if not available
            subtitle=None,
            status=snapshot.get("status", "open"),
            yes_bid=snapshot.get("yes_bid"),
            yes_ask=snapshot.get("yes_ask"),
            no_bid=snapshot.get("no_bid"),
            no_ask=snapshot.get("no_ask"),
            last_price=snapshot.get("last_price"),
            volume=snapshot.get("volume", 0),
            open_interest=snapshot.get("open_interest", 0),
            close_time=close_time,
            expiration_time=expiration_time,
        )

    def snapshot_to_orderbook(
        self,
        snapshot: dict,
        default_quantity: int = 100,
    ) -> OrderBook | None:
        """
        Convert a snapshot to a synthetic OrderBook.

        Creates an orderbook with single levels at bid/ask prices.
        Uses a default quantity since snapshots don't include depth.

        Args:
            snapshot: Snapshot from database
            default_quantity: Quantity to use for each level

        Returns:
            OrderBook object or None if no price data
        """
        if snapshot.get("yes_ask") is None and snapshot.get("no_ask") is None:
            return None

        yes_bids = []
        yes_asks = []
        no_bids = []
        no_asks = []

        if snapshot.get("yes_bid") is not None:
            yes_bids = [OrderBookLevel(price=snapshot["yes_bid"], quantity=default_quantity)]

        if snapshot.get("yes_ask") is not None:
            yes_asks = [OrderBookLevel(price=snapshot["yes_ask"], quantity=default_quantity)]

        if snapshot.get("no_bid") is not None:
            no_bids = [OrderBookLevel(price=snapshot["no_bid"], quantity=default_quantity)]

        if snapshot.get("no_ask") is not None:
            no_asks = [OrderBookLevel(price=snapshot["no_ask"], quantity=default_quantity)]

        return OrderBook(
            market_ticker=snapshot["ticker"],
            yes_bids=yes_bids,
            yes_asks=yes_asks,
            no_bids=no_bids,
            no_asks=no_asks,
        )

    async def load_market_state(
        self,
        timestamp: datetime,
    ) -> tuple[list[MarketData], dict[str, OrderBook]]:
        """
        Load full market state at a timestamp.

        Converts all snapshots to MarketData and OrderBook objects.

        Args:
            timestamp: Snapshot timestamp

        Returns:
            Tuple of (markets, orderbooks)
        """
        snapshots = await self.load_snapshots_at_timestamp(timestamp)

        markets: list[MarketData] = []
        orderbooks: dict[str, OrderBook] = {}

        for snapshot in snapshots:
            market = self.snapshot_to_market_data(snapshot)
            markets.append(market)

            orderbook = self.snapshot_to_orderbook(snapshot)
            if orderbook:
                orderbooks[market.ticker] = orderbook

        return markets, orderbooks

    async def get_snapshot_stats(self) -> dict:
        """
        Get statistics about available snapshot data.

        Returns:
            Dictionary with stats about snapshots
        """
        stats = {}

        # Total snapshots
        row = await self._db.fetch_one(
            "SELECT COUNT(*) as count FROM market_snapshots"
        )
        stats["total_snapshots"] = row["count"] if row else 0

        # Unique tickers
        row = await self._db.fetch_one(
            "SELECT COUNT(DISTINCT ticker) as count FROM market_snapshots"
        )
        stats["unique_tickers"] = row["count"] if row else 0

        # Unique timestamps
        row = await self._db.fetch_one(
            "SELECT COUNT(DISTINCT snapshot_at) as count FROM market_snapshots"
        )
        stats["unique_timestamps"] = row["count"] if row else 0

        # Date range
        earliest, latest = await self.get_date_range()
        stats["earliest_date"] = earliest.isoformat() if earliest else None
        stats["latest_date"] = latest.isoformat() if latest else None

        # Total settlements
        row = await self._db.fetch_one(
            "SELECT COUNT(*) as count FROM market_settlements"
        )
        stats["total_settlements"] = row["count"] if row else 0

        return stats
