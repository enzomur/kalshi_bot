"""Market tracker for detecting new market listings."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kalshi_bot.persistence.database import Database


@dataclass
class NewMarket:
    """A newly detected market."""

    ticker: str
    event_ticker: str
    title: str
    category: str
    initial_price: int | None
    first_seen_at: datetime
    hours_since_listing: float


class MarketTracker:
    """
    Tracks all markets and detects new listings.

    New markets often have inefficient pricing in first 48 hours
    as price discovery occurs.
    """

    # How long a market is considered "new"
    NEW_MARKET_WINDOW_HOURS = 48.0

    def __init__(
        self,
        db: "Database",
        new_market_window_hours: float = 48.0,
    ) -> None:
        """
        Initialize market tracker.

        Args:
            db: Database connection
            new_market_window_hours: How long to consider a market "new"
        """
        self._db = db
        self._window_hours = new_market_window_hours
        self._known_tickers: set[str] = set()

    async def load_known_markets(self) -> None:
        """Load known markets from database."""
        rows = await self._db.fetch_all("SELECT ticker FROM known_markets")
        self._known_tickers = {row["ticker"] for row in rows}

    async def detect_new_markets(
        self,
        current_markets: list[dict],
    ) -> list[NewMarket]:
        """
        Detect new markets from current market list.

        Args:
            current_markets: List of market dicts from API

        Returns:
            List of NewMarket objects for newly detected markets
        """
        now = datetime.utcnow()
        new_markets = []

        for market in current_markets:
            # Support both MarketData objects and dicts
            ticker = market.ticker if hasattr(market, "ticker") else market.get("ticker", "")
            if not ticker:
                continue

            if ticker not in self._known_tickers:
                event_ticker = market.event_ticker if hasattr(market, "event_ticker") else market.get("event_ticker", "")
                title = market.title if hasattr(market, "title") else market.get("title", "")
                last_price = market.last_price if hasattr(market, "last_price") else market.get("last_price")

                # This is a new market
                new_market = NewMarket(
                    ticker=ticker,
                    event_ticker=event_ticker,
                    title=title,
                    category=self._extract_category(market),
                    initial_price=last_price,
                    first_seen_at=now,
                    hours_since_listing=0.0,
                )
                new_markets.append(new_market)

                # Store in database
                await self._store_new_market(new_market)
                self._known_tickers.add(ticker)

        return new_markets

    async def get_recent_markets(self) -> list[NewMarket]:
        """
        Get markets that are still within the new market window.

        Returns:
            List of NewMarket objects within the window
        """
        now = datetime.utcnow()
        cutoff = now.timestamp() - (self._window_hours * 3600)

        rows = await self._db.fetch_all(
            """
            SELECT ticker, event_ticker, category, initial_price, first_seen_at
            FROM known_markets
            WHERE first_seen_at >= datetime(?, 'unixepoch')
            ORDER BY first_seen_at DESC
            """,
            (cutoff,),
        )

        markets = []
        for row in rows:
            try:
                first_seen = datetime.fromisoformat(row["first_seen_at"])
                hours = (now - first_seen).total_seconds() / 3600
            except (ValueError, TypeError):
                hours = 0.0

            markets.append(
                NewMarket(
                    ticker=row["ticker"],
                    event_ticker=row.get("event_ticker", ""),
                    title="",
                    category=row["category"] or "",
                    initial_price=row["initial_price"],
                    first_seen_at=first_seen if "first_seen" in dir() else now,
                    hours_since_listing=hours,
                )
            )

        return markets

    async def _store_new_market(self, market: NewMarket) -> None:
        """Store a new market in the database."""
        await self._db.execute(
            """
            INSERT OR IGNORE INTO known_markets
            (ticker, event_ticker, category, initial_price, first_seen_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                market.ticker,
                market.event_ticker,
                market.category,
                market.initial_price,
                market.first_seen_at.isoformat(),
            ),
        )

    def _extract_category(self, market) -> str:
        """Extract category from market data."""
        # Support both MarketData objects and dicts
        ticker = market.ticker if hasattr(market, "ticker") else market.get("ticker", "")
        title_raw = market.title if hasattr(market, "title") else market.get("title", "")
        title = (title_raw or "").lower()

        # Common categories
        if "GDP" in ticker or "gdp" in title:
            return "economic_gdp"
        if "CPI" in ticker or "inflation" in title:
            return "economic_cpi"
        if "JOBS" in ticker or "employment" in title or "payroll" in title:
            return "economic_jobs"
        if "TEMP" in ticker or "temperature" in title or "weather" in title:
            return "weather"
        if "PRECIP" in ticker or "rain" in title or "snow" in title:
            return "weather"
        if "ELECT" in ticker or "election" in title or "vote" in title:
            return "politics"
        if "SPORT" in ticker or any(
            s in title for s in ["game", "match", "championship"]
        ):
            return "sports"

        # Default to event ticker prefix
        event = market.event_ticker if hasattr(market, "event_ticker") else market.get("event_ticker", "")
        if event:
            return event.split("-")[0].lower()

        return "other"

    async def update_fair_value(self, ticker: str, fair_value: float) -> None:
        """Update fair value estimate for a market."""
        await self._db.execute(
            """
            UPDATE known_markets
            SET fair_value_estimate = ?
            WHERE ticker = ?
            """,
            (fair_value, ticker),
        )
