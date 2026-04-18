"""New Market Detector Agent - trades newly listed markets with inefficient pricing."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from kalshi_bot.agents.base import BaseAgent
from kalshi_bot.agents.new_market.fair_value import FairValueEstimator, FairValueEstimate
from kalshi_bot.agents.new_market.market_tracker import MarketTracker, NewMarket
from kalshi_bot.core.types import ArbitrageOpportunity, ArbitrageType, Side
from kalshi_bot.utils.logging import get_logger

if TYPE_CHECKING:
    from kalshi_bot.api.client import KalshiAPIClient
    from kalshi_bot.persistence.database import Database

logger = get_logger(__name__)


@dataclass
class NewMarketOpportunity:
    """A trading opportunity from new market mispricing."""

    market: NewMarket
    fair_value: FairValueEstimate
    current_price: int
    edge: float
    side: str
    quantity: int
    expected_profit: float

    def to_arbitrage_opportunity(self) -> ArbitrageOpportunity:
        """Convert to ArbitrageOpportunity for execution."""
        opportunity_id = (
            f"newmarket_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{str(uuid4())[:8]}"
        )

        side = Side.YES if self.side == "yes" else Side.NO
        price = self.current_price if side == Side.YES else (100 - self.current_price)

        leg = {
            "market_ticker": self.market.ticker,
            "side": side.value,
            "price": price,
            "quantity": self.quantity,
            "action": "buy",
        }

        return ArbitrageOpportunity(
            opportunity_id=opportunity_id,
            arbitrage_type=ArbitrageType.SINGLE_MARKET,
            markets=[self.market.ticker],
            expected_profit=self.expected_profit * 100,
            expected_profit_pct=self.edge,
            confidence=self.fair_value.confidence,
            legs=[leg],
            max_quantity=self.quantity,
            total_cost=self.quantity * price / 100,
            fees=0.07 * (price / 100) * (1 - price / 100) * self.quantity,
            net_profit=self.expected_profit,
            metadata={
                "source": "new_market",
                "category": self.market.category,
                "hours_since_listing": self.market.hours_since_listing,
                "fair_value": self.fair_value.fair_value,
                "fair_value_explanation": self.fair_value.explanation,
            },
        )


class NewMarketAgent(BaseAgent):
    """
    Agent that trades newly listed markets where pricing is inefficient.

    Strategy:
    - Track all markets, detect new ones
    - Compare to historical settlement rates for similar markets
    - Trade when current price diverges from historical fair value
    """

    # Minimum edge to trade
    MIN_EDGE = 0.10

    # Minimum confidence in fair value estimate
    MIN_CONFIDENCE = 0.40

    # Price bounds (avoid extreme prices)
    MIN_PRICE = 15
    MAX_PRICE = 85

    # Kelly fraction for position sizing
    KELLY_FRACTION = 0.08

    # Max percentage of capital per trade
    MAX_POSITION_PCT = 0.02

    def __init__(
        self,
        db: "Database",
        api_client: "KalshiAPIClient",
        new_market_window_hours: float = 48.0,
        min_fair_value_edge: float = 0.10,
        update_interval_minutes: int = 5,
        enabled: bool = True,
    ) -> None:
        """
        Initialize New Market Agent.

        Args:
            db: Database connection
            api_client: Kalshi API client
            new_market_window_hours: How long to consider a market "new"
            min_fair_value_edge: Minimum edge vs fair value to trade
            update_interval_minutes: How often to scan for new markets
            enabled: Whether the agent is enabled
        """
        super().__init__(
            db=db,
            name="new_market",
            update_interval_seconds=update_interval_minutes * 60,
            enabled=enabled,
        )

        self._api_client = api_client
        self._window_hours = new_market_window_hours
        self._min_edge = min_fair_value_edge

        self._tracker = MarketTracker(db, new_market_window_hours)
        self._estimator = FairValueEstimator(db)

        # Cache of recent opportunities
        self._opportunities: dict[str, NewMarketOpportunity] = {}

    async def _run_cycle(self) -> None:
        """Execute one cycle of new market detection."""
        logger.info("New market agent: scanning for new markets")

        # Load state if first run
        if not self._tracker._known_tickers:
            await self._tracker.load_known_markets()
            await self._estimator.load_historical_data()

        # Get current markets from API
        try:
            markets, _ = await self._api_client.get_markets(
                status="open",
                limit=200,
            )
        except Exception as e:
            logger.error(f"Failed to fetch markets: {e}")
            return

        # Detect new markets
        new_markets = await self._tracker.detect_new_markets(markets)

        if new_markets:
            logger.info(f"Detected {len(new_markets)} new markets")

        # Get all recent markets (within window)
        recent_markets = await self._tracker.get_recent_markets()

        # Analyze opportunities
        opportunities = []
        for market in recent_markets:
            opp = await self._analyze_market(market)
            if opp:
                opportunities.append(opp)
                self._opportunities[market.ticker] = opp

        # Update metrics
        self._status.metrics = {
            "markets_scanned": len(markets),
            "new_markets_detected": len(new_markets),
            "recent_markets": len(recent_markets),
            "opportunities_found": len(opportunities),
        }

        logger.info(
            f"New market agent: scanned {len(markets)} markets, "
            f"{len(new_markets)} new, {len(opportunities)} opportunities"
        )

    async def _analyze_market(self, market: NewMarket) -> NewMarketOpportunity | None:
        """Analyze a new market for trading opportunity."""
        # Get current price
        try:
            market_data = await self._api_client.get_market(market.ticker)
        except Exception as e:
            logger.debug(f"Failed to get market {market.ticker}: {e}")
            return None

        current_price = market_data.last_price
        if current_price is None:
            if market_data.yes_bid is not None and market_data.yes_ask is not None:
                current_price = (market_data.yes_bid + market_data.yes_ask) // 2
            else:
                return None

        # Skip extreme prices
        if current_price < self.MIN_PRICE or current_price > self.MAX_PRICE:
            return None

        # Get fair value estimate
        fair_value = await self._estimator.estimate_fair_value(
            ticker=market.ticker,
            category=market.category,
            current_price=current_price,
        )

        if fair_value.confidence < self.MIN_CONFIDENCE:
            logger.debug(
                f"Skipping {market.ticker}: low confidence {fair_value.confidence:.0%}"
            )
            return None

        # Calculate edge
        edge, side = self._estimator.calculate_edge(
            fair_value.fair_value,
            current_price,
        )

        if edge < self._min_edge:
            logger.debug(f"Skipping {market.ticker}: edge {edge:.1%} < {self._min_edge:.1%}")
            return None

        # Update fair value in database
        await self._tracker.update_fair_value(market.ticker, fair_value.fair_value)

        return NewMarketOpportunity(
            market=market,
            fair_value=fair_value,
            current_price=current_price,
            edge=edge,
            side=side,
            quantity=0,  # Will be calculated later
            expected_profit=0.0,
        )

    async def get_trading_opportunities(
        self,
        available_capital: float,
        max_opportunities: int = 5,
    ) -> list[ArbitrageOpportunity]:
        """
        Get new market trading opportunities for the bot.

        Args:
            available_capital: Available capital for trading
            max_opportunities: Maximum opportunities to return

        Returns:
            List of ArbitrageOpportunity objects
        """
        opportunities = []

        # Sort by edge * confidence
        ranked = sorted(
            self._opportunities.values(),
            key=lambda o: o.edge * o.fair_value.confidence,
            reverse=True,
        )

        for opp in ranked[:max_opportunities]:
            sized_opp = self._size_opportunity(opp, available_capital)
            if sized_opp and sized_opp.quantity > 0 and sized_opp.expected_profit > 0:
                opportunities.append(sized_opp.to_arbitrage_opportunity())

        return opportunities

    def _size_opportunity(
        self,
        opp: NewMarketOpportunity,
        available_capital: float,
    ) -> NewMarketOpportunity | None:
        """Calculate position size for an opportunity."""
        # Kelly formula
        if opp.side == "yes":
            win_prob = opp.fair_value.fair_value
            entry_price = opp.current_price
        else:
            win_prob = 1 - opp.fair_value.fair_value
            entry_price = 100 - opp.current_price

        lose_prob = 1 - win_prob
        payout_ratio = (100 - entry_price) / entry_price if entry_price > 0 else 0

        if payout_ratio > 0:
            kelly = (payout_ratio * win_prob - lose_prob) / payout_ratio
            kelly = max(0, min(kelly, self.KELLY_FRACTION))
        else:
            kelly = 0

        if kelly <= 0:
            return None

        # Cap at max position percentage
        position_value = min(
            available_capital * kelly,
            available_capital * self.MAX_POSITION_PCT,
        )

        cost_per_contract = entry_price / 100.0
        quantity = int(position_value / cost_per_contract) if cost_per_contract > 0 else 0

        if quantity < 1:
            return None

        # Expected profit
        expected_profit = (win_prob * quantity) - (quantity * cost_per_contract)
        fees = 0.07 * (entry_price / 100) * (1 - entry_price / 100) * quantity
        expected_profit -= fees

        if expected_profit <= 0:
            return None

        logger.info(
            f"New market opportunity: {opp.market.ticker} {opp.side.upper()} "
            f"@ {opp.current_price}c, fair={opp.fair_value.fair_value:.0%}, "
            f"edge={opp.edge:.1%}, qty={quantity}"
        )

        return NewMarketOpportunity(
            market=opp.market,
            fair_value=opp.fair_value,
            current_price=opp.current_price,
            edge=opp.edge,
            side=opp.side,
            quantity=quantity,
            expected_profit=expected_profit,
        )

    def get_status(self) -> dict[str, Any]:
        """Get agent status with new market-specific metrics."""
        status = super().get_status()
        status["known_markets"] = len(self._tracker._known_tickers)
        status["active_opportunities"] = len(self._opportunities)
        return status
