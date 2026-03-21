"""Outcome tracker for detecting market settlements."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from kalshi_bot.utils.logging import get_logger

if TYPE_CHECKING:
    from kalshi_bot.persistence.database import Database

logger = get_logger(__name__)


@dataclass
class Settlement:
    """Represents a detected market settlement."""

    ticker: str
    event_ticker: str
    outcome: str  # 'yes' or 'no'
    final_price: int
    first_detected_at: datetime
    snapshot_count: int


class OutcomeTracker:
    """
    Detects market settlements by monitoring price convergence.

    Markets are considered settled when:
    - Final price >= 99 cents -> YES won
    - Final price <= 1 cent -> NO won

    This approach infers outcomes from price behavior without needing
    direct access to settlement data from the API.
    """

    # Price thresholds for settlement detection
    YES_WIN_THRESHOLD = 99  # Price >= 99 means YES won
    NO_WIN_THRESHOLD = 1    # Price <= 1 means NO won

    def __init__(self, db: Database) -> None:
        """
        Initialize the outcome tracker.

        Args:
            db: Database connection
        """
        self._db = db
        self._settlements_detected = 0
        self._last_check: datetime | None = None

    async def check_settlements(self) -> list[Settlement]:
        """
        Check for newly settled markets based on price convergence.

        Looks for markets with recent snapshots showing extreme prices
        (>= 99 or <= 1) that haven't been recorded as settled yet.

        Returns:
            List of newly detected settlements
        """
        self._last_check = datetime.utcnow()
        settlements: list[Settlement] = []

        # Find markets with extreme recent prices that aren't yet settled
        # Use the most recent snapshot for each ticker
        query = """
            WITH recent_snapshots AS (
                SELECT
                    ticker,
                    event_ticker,
                    last_price,
                    yes_bid,
                    yes_ask,
                    snapshot_at,
                    ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY snapshot_at DESC) as rn
                FROM market_snapshots
                WHERE snapshot_at >= datetime('now', '-1 hour')
            )
            SELECT
                rs.ticker,
                rs.event_ticker,
                rs.last_price,
                rs.yes_bid,
                rs.yes_ask,
                rs.snapshot_at,
                (SELECT COUNT(*) FROM market_snapshots WHERE ticker = rs.ticker) as snapshot_count
            FROM recent_snapshots rs
            LEFT JOIN market_settlements ms ON rs.ticker = ms.ticker
            WHERE rs.rn = 1
              AND ms.ticker IS NULL
              AND (
                  rs.last_price >= ?
                  OR rs.last_price <= ?
                  OR rs.yes_bid >= ?
                  OR rs.yes_ask <= ?
              )
        """

        rows = await self._db.fetch_all(
            query,
            (
                self.YES_WIN_THRESHOLD,
                self.NO_WIN_THRESHOLD,
                self.YES_WIN_THRESHOLD,
                self.NO_WIN_THRESHOLD,
            ),
        )

        for row in rows:
            settlement = await self._process_potential_settlement(row)
            if settlement:
                settlements.append(settlement)

        if settlements:
            logger.info(f"Detected {len(settlements)} new market settlements")
            self._settlements_detected += len(settlements)

        return settlements

    async def _process_potential_settlement(self, row: dict) -> Settlement | None:
        """
        Process a potential settlement and record it if valid.

        Args:
            row: Database row with market snapshot data

        Returns:
            Settlement if confirmed, None otherwise
        """
        ticker = row["ticker"]
        event_ticker = row["event_ticker"]
        last_price = row["last_price"]
        yes_bid = row["yes_bid"]
        yes_ask = row["yes_ask"]
        snapshot_at = row["snapshot_at"]
        snapshot_count = row["snapshot_count"]

        # Determine outcome from price
        outcome = await self.infer_outcome(ticker, last_price, yes_bid, yes_ask)
        if not outcome:
            return None

        # Determine final price used for inference
        final_price = last_price
        if final_price is None:
            final_price = yes_bid if outcome == "yes" else (100 - yes_ask if yes_ask else 0)

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
                    ticker,
                    event_ticker,
                    outcome,
                    final_price,
                    snapshot_at,
                    now.isoformat(),
                    snapshot_count,
                ),
            )

            logger.info(
                f"Recorded settlement: {ticker} -> {outcome.upper()} "
                f"(final_price={final_price}, snapshots={snapshot_count})"
            )

            return Settlement(
                ticker=ticker,
                event_ticker=event_ticker,
                outcome=outcome,
                final_price=final_price,
                first_detected_at=datetime.fromisoformat(snapshot_at) if isinstance(snapshot_at, str) else snapshot_at,
                snapshot_count=snapshot_count,
            )

        except Exception as e:
            # Likely duplicate - already settled
            logger.debug(f"Settlement already recorded for {ticker}: {e}")
            return None

    async def infer_outcome(
        self,
        ticker: str,
        last_price: int | None = None,
        yes_bid: int | None = None,
        yes_ask: int | None = None,
    ) -> str | None:
        """
        Infer market outcome from price data.

        Args:
            ticker: Market ticker
            last_price: Last traded price
            yes_bid: Current YES bid price
            yes_ask: Current YES ask price

        Returns:
            'yes', 'no', or None if cannot determine
        """
        # Check last_price first
        if last_price is not None:
            if last_price >= self.YES_WIN_THRESHOLD:
                return "yes"
            if last_price <= self.NO_WIN_THRESHOLD:
                return "no"

        # Fall back to bid/ask
        if yes_bid is not None and yes_bid >= self.YES_WIN_THRESHOLD:
            return "yes"
        if yes_ask is not None and yes_ask <= self.NO_WIN_THRESHOLD:
            return "no"

        return None

    async def get_settlement(self, ticker: str) -> dict | None:
        """Get settlement record for a ticker."""
        return await self._db.fetch_one(
            "SELECT * FROM market_settlements WHERE ticker = ?",
            (ticker,),
        )

    async def get_recent_settlements(
        self,
        limit: int = 50,
        outcome: str | None = None,
    ) -> list[dict]:
        """
        Get recent settlements.

        Args:
            limit: Maximum number to return
            outcome: Filter by outcome ('yes' or 'no')

        Returns:
            List of settlement records
        """
        if outcome:
            return await self._db.fetch_all(
                """
                SELECT * FROM market_settlements
                WHERE outcome = ?
                ORDER BY confirmed_at DESC
                LIMIT ?
                """,
                (outcome, limit),
            )
        return await self._db.fetch_all(
            """
            SELECT * FROM market_settlements
            ORDER BY confirmed_at DESC
            LIMIT ?
            """,
            (limit,),
        )

    async def get_settlement_count(self, outcome: str | None = None) -> int:
        """Get total settlement count, optionally filtered by outcome."""
        if outcome:
            result = await self._db.fetch_one(
                "SELECT COUNT(*) as count FROM market_settlements WHERE outcome = ?",
                (outcome,),
            )
        else:
            result = await self._db.fetch_one(
                "SELECT COUNT(*) as count FROM market_settlements"
            )
        return result["count"] if result else 0

    async def get_settlements_with_snapshots(
        self,
        min_snapshots: int = 10,
        limit: int | None = None,
    ) -> list[dict]:
        """
        Get settlements that have sufficient snapshot history for training.

        Args:
            min_snapshots: Minimum number of snapshots required
            limit: Maximum number to return

        Returns:
            List of settlement records with adequate data
        """
        query = """
            SELECT * FROM market_settlements
            WHERE snapshot_count >= ?
            ORDER BY confirmed_at DESC
        """
        params: tuple = (min_snapshots,)

        if limit:
            query += " LIMIT ?"
            params = (min_snapshots, limit)

        return await self._db.fetch_all(query, params)

    async def update_prediction_outcomes(self) -> int:
        """
        Update ml_predictions table with actual outcomes for settled markets.

        Returns:
            Number of predictions updated
        """
        cursor = await self._db.execute(
            """
            UPDATE ml_predictions
            SET
                actual_outcome = (
                    SELECT outcome FROM market_settlements
                    WHERE market_settlements.ticker = ml_predictions.ticker
                ),
                was_correct = (
                    CASE
                        WHEN (
                            SELECT outcome FROM market_settlements
                            WHERE market_settlements.ticker = ml_predictions.ticker
                        ) = 'yes' AND ml_predictions.predicted_prob_yes >= 0.5 THEN 1
                        WHEN (
                            SELECT outcome FROM market_settlements
                            WHERE market_settlements.ticker = ml_predictions.ticker
                        ) = 'no' AND ml_predictions.predicted_prob_yes < 0.5 THEN 1
                        ELSE 0
                    END
                ),
                settled_at = CURRENT_TIMESTAMP
            WHERE actual_outcome IS NULL
              AND ticker IN (SELECT ticker FROM market_settlements)
            """
        )
        updated = cursor.rowcount
        if updated > 0:
            logger.info(f"Updated {updated} prediction outcomes")
        return updated

    def get_status(self) -> dict:
        """Get tracker status."""
        return {
            "settlements_detected": self._settlements_detected,
            "last_check": self._last_check.isoformat() if self._last_check else None,
            "thresholds": {
                "yes_win": self.YES_WIN_THRESHOLD,
                "no_win": self.NO_WIN_THRESHOLD,
            },
        }
