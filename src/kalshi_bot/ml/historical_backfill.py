"""Historical data backfiller for bootstrapping ML training."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from kalshi_bot.utils.logging import get_logger

if TYPE_CHECKING:
    from kalshi_bot.api.client import KalshiAPIClient
    from kalshi_bot.persistence.database import Database

logger = get_logger(__name__)


@dataclass
class BackfillResult:
    """Result of a backfill operation."""

    success: bool
    settled_markets_added: int
    snapshots_added: int
    errors: list[str]
    duration_seconds: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "settled_markets_added": self.settled_markets_added,
            "snapshots_added": self.snapshots_added,
            "errors": self.errors,
            "duration_seconds": self.duration_seconds,
        }


class HistoricalBackfiller:
    """
    Fetches historical data from Kalshi to bootstrap ML training.

    This class handles:
    1. Fetching settled markets and their outcomes
    2. Fetching candlestick OHLC data and converting to snapshots
    3. Running a complete backfill process

    Usage:
        backfiller = HistoricalBackfiller(db, api_client)
        result = await backfiller.run_full_backfill(days_back=90)
    """

    def __init__(
        self,
        db: Database,
        api_client: KalshiAPIClient,
    ) -> None:
        """
        Initialize the backfiller.

        Args:
            db: Database connection
            api_client: Kalshi API client
        """
        self._db = db
        self._api_client = api_client

    async def backfill_settled_markets(
        self,
        days_back: int = 90,
        max_pages: int = 500,  # Default limit: 50K markets to avoid timeouts
        min_volume: int = 0,
    ) -> int:
        """
        Fetch settled markets and record their outcomes.

        Args:
            days_back: Days of history to fetch
            max_pages: Maximum API pages to fetch
            min_volume: Minimum volume filter

        Returns:
            Number of new settlements added
        """
        logger.info(f"Backfilling settled markets (last {days_back} days)...")

        # Get cutoff date (timezone-aware to match API datetimes)
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=days_back)

        # Fetch all settled markets
        all_markets = await self._api_client.get_all_settled_markets(
            max_pages=max_pages,
            min_volume=min_volume,
        )

        logger.info(f"Found {len(all_markets)} settled markets from API")

        # Filter by date if we have close_time
        # Handle both timezone-aware and naive datetimes
        def is_in_range(close_time: datetime | None) -> bool:
            if close_time is None:
                return True
            # Make timezone-aware if naive
            if close_time.tzinfo is None:
                close_time = close_time.replace(tzinfo=timezone.utc)
            return close_time >= cutoff_time

        markets_in_range = [m for m in all_markets if is_in_range(m.close_time)]

        logger.info(f"{len(markets_in_range)} markets within {days_back} day range")

        added_count = 0
        skipped_no_outcome = 0

        for market in markets_in_range:
            # Use result field if available, otherwise infer from price
            outcome = market.result  # "yes" or "no" from API
            if not outcome:
                # Fallback to inferring from price
                outcome = self._infer_outcome_from_price(market.last_price)
            if not outcome:
                skipped_no_outcome += 1
                continue

            # Check if already recorded
            existing = await self._db.fetch_one(
                "SELECT ticker FROM market_settlements WHERE ticker = ?",
                (market.ticker,),
            )
            if existing:
                continue

            # Count existing snapshots for this market
            snapshot_result = await self._db.fetch_one(
                "SELECT COUNT(*) as count FROM market_snapshots WHERE ticker = ?",
                (market.ticker,),
            )
            snapshot_count = snapshot_result["count"] if snapshot_result else 0

            # Record the settlement
            now = datetime.utcnow()
            try:
                await self._db.execute(
                    """
                    INSERT INTO market_settlements (
                        ticker, event_ticker, outcome, final_price,
                        first_detected_at, confirmed_at, snapshot_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        market.ticker,
                        market.event_ticker,
                        outcome,
                        market.last_price or 0,
                        (market.close_time or now).isoformat(),
                        now.isoformat(),
                        snapshot_count,
                    ),
                )
                added_count += 1

                if added_count % 50 == 0:
                    logger.info(f"Added {added_count} settlements so far...")

            except Exception as e:
                logger.debug(f"Error adding settlement for {market.ticker}: {e}")

        if skipped_no_outcome > 0:
            logger.info(f"Backfilled {added_count} new settled markets (skipped {skipped_no_outcome} with no outcome)")
        else:
            logger.info(f"Backfilled {added_count} new settled markets")
        return added_count

    async def backfill_candlesticks(
        self,
        tickers: list[str] | None = None,
        days_back: int = 90,
        period_interval: int = 60,
        batch_size: int = 50,
    ) -> int:
        """
        Fetch OHLC candlestick history and convert to snapshots.

        Args:
            tickers: List of tickers to backfill (None = all with settlements)
            days_back: Days of history to fetch
            period_interval: Candlestick interval in minutes (60 = hourly)
            batch_size: Number of tickers to fetch per batch

        Returns:
            Number of snapshots added
        """
        logger.info(f"Backfilling candlestick data (last {days_back} days)...")

        if tickers is None:
            # Get all tickers with settlements
            settlements = await self._db.fetch_all(
                "SELECT ticker, event_ticker FROM market_settlements"
            )
            tickers = [s["ticker"] for s in settlements]

        if not tickers:
            logger.info("No tickers to backfill")
            return 0

        logger.info(f"Backfilling candlesticks for {len(tickers)} markets")

        # Calculate time range
        end_ts = int(datetime.utcnow().timestamp())
        start_ts = int((datetime.utcnow() - timedelta(days=days_back)).timestamp())

        total_snapshots = 0
        errors = 0

        # Process tickers in batches
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]
            logger.info(f"Processing batch {i // batch_size + 1} ({len(batch)} tickers)")

            for ticker in batch:
                try:
                    # Get event_ticker for this market
                    market_info = await self._db.fetch_one(
                        "SELECT event_ticker FROM market_settlements WHERE ticker = ?",
                        (ticker,),
                    )

                    if not market_info:
                        # Try to get from existing snapshots
                        market_info = await self._db.fetch_one(
                            "SELECT event_ticker FROM market_snapshots WHERE ticker = ? LIMIT 1",
                            (ticker,),
                        )

                    event_ticker = market_info["event_ticker"] if market_info else ""

                    # Fetch candlesticks
                    candlesticks = await self._api_client.get_market_candlesticks(
                        ticker=ticker,
                        series_ticker=event_ticker,
                        start_ts=start_ts,
                        end_ts=end_ts,
                        period_interval=period_interval,
                    )

                    if not candlesticks:
                        continue

                    # Convert candlesticks to snapshots
                    added = await self._candlesticks_to_snapshots(
                        ticker, event_ticker, candlesticks
                    )
                    total_snapshots += added

                except Exception as e:
                    logger.debug(f"Error fetching candlesticks for {ticker}: {e}")
                    errors += 1
                    continue

                # Small delay to avoid rate limiting
                await asyncio.sleep(0.1)

            # Log progress
            logger.info(f"Added {total_snapshots} snapshots so far ({errors} errors)")

        logger.info(f"Backfilled {total_snapshots} candlestick snapshots")
        return total_snapshots

    async def backfill_from_historical_api(
        self,
        max_pages: int = 50,
    ) -> int:
        """
        Fetch data from Kalshi's historical/archived API (>3 months old).

        Args:
            max_pages: Maximum pages to fetch

        Returns:
            Number of markets processed
        """
        logger.info("Fetching historical/archived market data...")

        try:
            # Get cutoff info
            cutoff = await self._api_client.get_historical_cutoff()
            logger.info(f"Historical cutoff: {cutoff}")
        except Exception as e:
            logger.warning(f"Could not get historical cutoff: {e}")

        total_markets = 0
        cursor: str | None = None
        page = 0

        while page < max_pages:
            try:
                markets, cursor = await self._api_client.get_historical_markets(
                    limit=100, cursor=cursor
                )
                page += 1

                for market in markets:
                    # Record settlement if we can determine outcome
                    outcome = self._infer_outcome_from_price(market.last_price)
                    if outcome:
                        try:
                            await self._db.execute(
                                """
                                INSERT OR IGNORE INTO market_settlements (
                                    ticker, event_ticker, outcome, final_price,
                                    first_detected_at, confirmed_at, snapshot_count
                                ) VALUES (?, ?, ?, ?, ?, ?, 0)
                                """,
                                (
                                    market.ticker,
                                    market.event_ticker,
                                    outcome,
                                    market.last_price or 0,
                                    (market.close_time or datetime.utcnow()).isoformat(),
                                    datetime.utcnow().isoformat(),
                                ),
                            )
                            total_markets += 1
                        except Exception:
                            pass

                logger.info(f"Processed page {page}: {len(markets)} historical markets")

                if not cursor or len(markets) < 100:
                    break

            except Exception as e:
                logger.warning(f"Error fetching historical markets: {e}")
                break

        logger.info(f"Added {total_markets} historical market settlements")
        return total_markets

    async def run_full_backfill(
        self,
        days_back: int = 90,
        include_candlesticks: bool = True,
        include_historical: bool = False,
    ) -> BackfillResult:
        """
        Run a complete backfill process.

        Args:
            days_back: Days of history to fetch
            include_candlesticks: Whether to fetch candlestick data
            include_historical: Whether to include archived data (>3 months)

        Returns:
            BackfillResult with summary
        """
        start_time = datetime.utcnow()
        errors: list[str] = []
        settled_count = 0
        snapshot_count = 0

        logger.info("=" * 60)
        logger.info("Starting Historical Data Backfill")
        logger.info("=" * 60)
        logger.info(f"Days back: {days_back}")
        logger.info(f"Include candlesticks: {include_candlesticks}")
        logger.info(f"Include historical API: {include_historical}")
        logger.info("=" * 60)

        # Step 1: Backfill settled markets
        try:
            settled_count = await self.backfill_settled_markets(days_back=days_back)
        except Exception as e:
            logger.error(f"Error backfilling settled markets: {e}")
            errors.append(f"Settled markets: {str(e)}")

        # Step 2: Backfill historical/archived data if requested
        if include_historical:
            try:
                historical_count = await self.backfill_from_historical_api()
                settled_count += historical_count
            except Exception as e:
                logger.error(f"Error backfilling historical data: {e}")
                errors.append(f"Historical data: {str(e)}")

        # Step 3: Backfill candlesticks if requested
        if include_candlesticks:
            try:
                snapshot_count = await self.backfill_candlesticks(
                    days_back=days_back,
                    period_interval=60,  # Hourly
                )
            except Exception as e:
                logger.error(f"Error backfilling candlesticks: {e}")
                errors.append(f"Candlesticks: {str(e)}")

        # Calculate duration
        duration = (datetime.utcnow() - start_time).total_seconds()

        # Log summary
        logger.info("=" * 60)
        logger.info("Backfill Complete")
        logger.info("=" * 60)
        logger.info(f"Settled markets added: {settled_count}")
        logger.info(f"Snapshots added: {snapshot_count}")
        logger.info(f"Errors: {len(errors)}")
        logger.info(f"Duration: {duration:.1f} seconds")
        logger.info("=" * 60)

        # Get totals from database
        total_settlements = await self._db.fetch_one(
            "SELECT COUNT(*) as count FROM market_settlements"
        )
        total_snapshots = await self._db.fetch_one(
            "SELECT COUNT(*) as count FROM market_snapshots"
        )

        logger.info(f"Total settlements in DB: {total_settlements['count'] if total_settlements else 0}")
        logger.info(f"Total snapshots in DB: {total_snapshots['count'] if total_snapshots else 0}")

        return BackfillResult(
            success=len(errors) == 0,
            settled_markets_added=settled_count,
            snapshots_added=snapshot_count,
            errors=errors,
            duration_seconds=duration,
        )

    async def _candlesticks_to_snapshots(
        self,
        ticker: str,
        event_ticker: str,
        candlesticks: list,
    ) -> int:
        """
        Convert candlestick data to market snapshots.

        Args:
            ticker: Market ticker
            event_ticker: Event ticker
            candlesticks: List of Candlestick objects

        Returns:
            Number of snapshots added
        """
        if not candlesticks:
            return 0

        snapshots = []
        for candle in candlesticks:
            # Use close price as the snapshot price
            # Calculate spread from high-low if available
            spread = candle.high_price - candle.low_price if candle.high_price and candle.low_price else None

            # Convert timestamp to datetime
            snapshot_time = datetime.fromtimestamp(candle.end_period_ts)

            snapshots.append((
                ticker,
                event_ticker,
                candle.close_price,  # yes_bid approximation
                candle.close_price,  # yes_ask approximation
                100 - candle.close_price if candle.close_price else None,  # no_bid
                100 - candle.close_price if candle.close_price else None,  # no_ask
                candle.close_price,  # last_price
                candle.volume,
                candle.open_interest,
                spread,
                "settled",  # status
                None,  # close_time
                None,  # expiration_time
                snapshot_time.isoformat(),  # snapshot_at
            ))

        if not snapshots:
            return 0

        # Batch insert, ignoring duplicates
        try:
            await self._db.execute_many(
                """
                INSERT OR IGNORE INTO market_snapshots (
                    ticker, event_ticker,
                    yes_bid, yes_ask, no_bid, no_ask, last_price,
                    volume, open_interest, spread,
                    status, close_time, expiration_time,
                    snapshot_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                snapshots,
            )
            return len(snapshots)
        except Exception as e:
            logger.debug(f"Error inserting candlestick snapshots: {e}")
            return 0

    def _infer_outcome_from_price(self, price: int | None) -> str | None:
        """
        Infer market outcome from final price.

        Args:
            price: Final price in cents (0-100)

        Returns:
            'yes', 'no', or None if indeterminate
        """
        if price is None:
            return None

        if price >= 99:
            return "yes"
        if price <= 1:
            return "no"

        # For prices between 1-99, we can't definitively determine outcome
        # Some markets may be in this range at settlement
        if price >= 90:
            return "yes"
        if price <= 10:
            return "no"

        return None

    async def get_backfill_status(self) -> dict[str, Any]:
        """Get current backfill status and statistics."""
        settlements = await self._db.fetch_one(
            "SELECT COUNT(*) as count FROM market_settlements"
        )
        snapshots = await self._db.fetch_one(
            "SELECT COUNT(*) as count FROM market_snapshots"
        )

        unique_tickers = await self._db.fetch_one(
            "SELECT COUNT(DISTINCT ticker) as count FROM market_snapshots"
        )

        settlements_with_data = await self._db.fetch_one(
            """
            SELECT COUNT(*) as count FROM market_settlements
            WHERE snapshot_count >= 10
            """
        )

        outcome_distribution = await self._db.fetch_all(
            "SELECT outcome, COUNT(*) as count FROM market_settlements GROUP BY outcome"
        )

        return {
            "total_settlements": settlements["count"] if settlements else 0,
            "total_snapshots": snapshots["count"] if snapshots else 0,
            "unique_tickers_with_snapshots": unique_tickers["count"] if unique_tickers else 0,
            "settlements_with_sufficient_data": settlements_with_data["count"] if settlements_with_data else 0,
            "outcome_distribution": {r["outcome"]: r["count"] for r in outcome_distribution},
        }
