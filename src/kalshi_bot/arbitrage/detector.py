"""Arbitrage detector that coordinates multiple strategies."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

from kalshi_bot.api.client import KalshiAPIClient
from kalshi_bot.arbitrage.strategies.base import ArbitrageStrategy
from kalshi_bot.arbitrage.strategies.cross_market import CrossMarketStrategy
from kalshi_bot.arbitrage.strategies.multi_outcome import MultiOutcomeStrategy
from kalshi_bot.arbitrage.strategies.single_market import SingleMarketStrategy
from kalshi_bot.config.settings import Settings
from kalshi_bot.core.types import ArbitrageOpportunity, MarketData, OrderBook, OrderBookLevel
from kalshi_bot.persistence.models import OpportunityRepository
from kalshi_bot.utils.logging import get_logger

logger = get_logger(__name__)


class ArbitrageDetector:
    """
    Coordinates arbitrage detection across multiple strategies.

    Manages:
    - Strategy initialization and configuration
    - Market data fetching
    - Opportunity ranking and filtering
    - Persistence of detected opportunities
    """

    def __init__(
        self,
        settings: Settings,
        api_client: KalshiAPIClient,
        opportunity_repo: OpportunityRepository | None = None,
    ) -> None:
        """
        Initialize detector.

        Args:
            settings: Application settings
            api_client: Kalshi API client
            opportunity_repo: Repository for persisting opportunities
        """
        self.settings = settings
        self.api_client = api_client
        self.opportunity_repo = opportunity_repo

        self._strategies: list[ArbitrageStrategy] = []
        self._initialize_strategies()

        self._markets_cache: list[MarketData] = []
        self._orderbooks_cache: dict[str, OrderBook] = {}
        self._last_refresh: datetime | None = None
        self._cache_ttl_seconds = 5
        self._first_scan_complete = False

    def _initialize_strategies(self) -> None:
        """Initialize enabled strategies."""
        if self.settings.arbitrage.enable_single_market:
            self._strategies.append(SingleMarketStrategy(self.settings))
            logger.info("Single market strategy enabled")

        if self.settings.arbitrage.enable_multi_outcome:
            self._strategies.append(MultiOutcomeStrategy(self.settings))
            logger.info("Multi-outcome strategy enabled")

        if self.settings.arbitrage.enable_cross_market:
            strategy = CrossMarketStrategy(self.settings)
            self._strategies.append(strategy)
            logger.info("Cross-market strategy enabled")

    def add_cross_market_relationship(
        self,
        parent_ticker: str,
        child_ticker: str,
        relationship_type: str,
        description: str = "",
    ) -> None:
        """
        Add a cross-market relationship for arbitrage detection.

        Args:
            parent_ticker: The market with higher/equal probability
            child_ticker: The market with lower/equal probability
            relationship_type: Type of relationship
            description: Human-readable description
        """
        for strategy in self._strategies:
            if isinstance(strategy, CrossMarketStrategy):
                strategy.add_relationship(
                    parent_ticker, child_ticker,
                    relationship_type, description,
                )
                logger.info(f"Added relationship: {parent_ticker} -> {child_ticker}")

    async def refresh_market_data(self, force: bool = False) -> None:
        """
        Refresh market data and order books.

        Fetches all open markets with pagination and filters for liquidity
        (markets with volume, open interest, or active bid/ask).

        Args:
            force: Force refresh even if cache is valid
        """
        now = datetime.utcnow()

        if not force and self._last_refresh:
            elapsed = (now - self._last_refresh).total_seconds()
            if elapsed < self._cache_ttl_seconds:
                return

        # Full scan on first run, limited scan on refreshes
        if not self._first_scan_complete:
            logger.info("First scan: fetching markets (limited to 200 pages)")
            max_pages = 200
        else:
            logger.info("Refresh scan: limited to 50 pages")
            max_pages = 50

        # Fetch markets with pagination, filtering for liquidity
        all_markets = await self.api_client.get_all_markets(
            status="open",
            min_volume=0,  # No volume filter
            min_open_interest=0,  # No open interest filter
            require_orderbook=False,  # Allow markets without orderbook
            max_pages=max_pages,
        )

        # Filter out parlay/multi-game markets - these are NOT arbitrage
        parlay_keywords = ["MULTIGAME", "PARLAY", "EXTENDED", "COMBO", "ACCUMULATOR"]
        self._markets_cache = [
            m for m in all_markets
            if not any(kw in m.ticker.upper() for kw in parlay_keywords)
        ]
        filtered_count = len(all_markets) - len(self._markets_cache)
        if filtered_count > 0:
            logger.info(f"Filtered out {filtered_count} parlay/multi-game markets")

        if not self._first_scan_complete:
            self._first_scan_complete = True

        # Log category breakdown for debugging
        categories: dict[str, int] = {}
        for market in self._markets_cache:
            # Extract category from event_ticker (e.g., "ESPORTS-..." -> "ESPORTS")
            category = market.event_ticker.split("-")[0] if market.event_ticker else "UNKNOWN"
            categories[category] = categories.get(category, 0) + 1
        logger.info(f"Market categories: {dict(sorted(categories.items(), key=lambda x: -x[1]))}")

        # Create synthetic orderbooks from market data (which has bid/ask prices)
        # This avoids 58k+ API calls while still providing data for strategies
        self._orderbooks_cache = {}
        for market in self._markets_cache:
            # Only create orderbook if we have price data
            if market.yes_ask is not None or market.no_ask is not None:
                orderbook = OrderBook(
                    market_ticker=market.ticker,
                    yes_bids=[OrderBookLevel(price=market.yes_bid, quantity=100)] if market.yes_bid else [],
                    yes_asks=[OrderBookLevel(price=market.yes_ask, quantity=100)] if market.yes_ask else [],
                    no_bids=[OrderBookLevel(price=market.no_bid, quantity=100)] if market.no_bid else [],
                    no_asks=[OrderBookLevel(price=market.no_ask, quantity=100)] if market.no_ask else [],
                )
                self._orderbooks_cache[market.ticker] = orderbook

        self._last_refresh = now
        logger.info(
            f"Refreshed {len(self._markets_cache)} liquid markets"
        )

    async def _fetch_orderbook(self, ticker: str) -> OrderBook:
        """Fetch order book for a market."""
        return await self.api_client.get_orderbook(ticker)

    async def detect_opportunities(
        self,
        refresh_data: bool = True,
    ) -> list[ArbitrageOpportunity]:
        """
        Detect arbitrage opportunities across all strategies.

        Args:
            refresh_data: Whether to refresh market data first

        Returns:
            List of detected opportunities, ranked by expected profit
        """
        if refresh_data:
            await self.refresh_market_data()

        if not self._markets_cache:
            logger.warning("No markets available for arbitrage detection")
            return []

        all_opportunities: list[ArbitrageOpportunity] = []

        for strategy in self._strategies:
            try:
                opportunities = await strategy.detect(
                    self._markets_cache,
                    self._orderbooks_cache,
                )
                all_opportunities.extend(opportunities)
            except Exception as e:
                logger.error(f"Strategy {strategy.__class__.__name__} failed: {e}")

        ranked = self._rank_opportunities(all_opportunities)

        if self.opportunity_repo:
            for opp in ranked:
                try:
                    await self.opportunity_repo.save(opp)
                except Exception as e:
                    logger.error(f"Failed to save opportunity {opp.opportunity_id}: {e}")

        if ranked:
            logger.info(f"Detected {len(ranked)} arbitrage opportunities")

        return ranked

    def _rank_opportunities(
        self,
        opportunities: list[ArbitrageOpportunity],
    ) -> list[ArbitrageOpportunity]:
        """
        Rank opportunities by profitability and confidence.

        Ranking factors:
        1. Net profit (primary)
        2. ROI percentage
        3. Confidence score
        4. Available quantity

        Args:
            opportunities: List of opportunities to rank

        Returns:
            Sorted list of opportunities (best first)
        """
        if not opportunities:
            return []

        def score(opp: ArbitrageOpportunity) -> float:
            profit_weight = 0.4
            roi_weight = 0.3
            confidence_weight = 0.2
            quantity_weight = 0.1

            max_profit = max(o.net_profit for o in opportunities) or 1
            max_roi = max(o.roi for o in opportunities) or 1
            max_qty = max(o.max_quantity for o in opportunities) or 1

            profit_score = opp.net_profit / max_profit
            roi_score = opp.roi / max_roi
            confidence_score = opp.confidence
            qty_score = opp.max_quantity / max_qty

            return (
                profit_weight * profit_score
                + roi_weight * roi_score
                + confidence_weight * confidence_score
                + quantity_weight * qty_score
            )

        ranked = sorted(opportunities, key=score, reverse=True)

        return ranked

    async def get_best_opportunity(
        self,
        min_profit: float | None = None,
        min_confidence: float | None = None,
    ) -> ArbitrageOpportunity | None:
        """
        Get the best available opportunity matching criteria.

        Args:
            min_profit: Minimum net profit in dollars
            min_confidence: Minimum confidence score (0-1)

        Returns:
            Best opportunity or None
        """
        opportunities = await self.detect_opportunities()

        for opp in opportunities:
            if min_profit and opp.net_profit < min_profit:
                continue
            if min_confidence and opp.confidence < min_confidence:
                continue
            return opp

        return None

    def get_status(self) -> dict[str, Any]:
        """Get detector status."""
        return {
            "strategies_enabled": [s.__class__.__name__ for s in self._strategies],
            "markets_cached": len(self._markets_cache),
            "orderbooks_cached": len(self._orderbooks_cache),
            "last_refresh": self._last_refresh.isoformat() if self._last_refresh else None,
            "cache_ttl_seconds": self._cache_ttl_seconds,
        }
