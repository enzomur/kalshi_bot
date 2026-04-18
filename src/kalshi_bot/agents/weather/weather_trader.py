"""Weather-specific trading strategy using NWS forecasts directly."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING
from uuid import uuid4

from kalshi_bot.agents.weather.probability_calc import ProbabilityEstimate
from kalshi_bot.core.types import ArbitrageOpportunity, ArbitrageType, Side
from kalshi_bot.utils.logging import get_logger

if TYPE_CHECKING:
    from kalshi_bot.agents.weather.agent import WeatherResearchAgent
    from kalshi_bot.api.client import KalshiAPIClient

logger = get_logger(__name__)


@dataclass
class WeatherTradingOpportunity:
    """A weather trading opportunity based on NWS forecast edge."""

    ticker: str
    nws_probability: float  # NWS forecast probability (0-1)
    market_price: int  # Current market price in cents
    edge: float  # nws_prob - market_prob (signed)
    confidence: float  # NWS forecast confidence
    side: str  # 'yes' or 'no'
    explanation: str
    hours_until_event: float

    @property
    def market_prob(self) -> float:
        """Market implied probability."""
        return self.market_price / 100.0

    @property
    def abs_edge(self) -> float:
        """Absolute edge magnitude."""
        return abs(self.edge)

    def to_arbitrage_opportunity(
        self,
        quantity: int,
        expected_profit: float,
    ) -> ArbitrageOpportunity:
        """Convert to ArbitrageOpportunity for execution."""
        opportunity_id = f"weather_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{str(uuid4())[:8]}"

        side = Side.YES if self.side == "yes" else Side.NO
        price = self.market_price if side == Side.YES else (100 - self.market_price)

        leg = {
            "market_ticker": self.ticker,
            "side": side.value,
            "price": price,
            "quantity": quantity,
            "action": "buy",
        }

        return ArbitrageOpportunity(
            opportunity_id=opportunity_id,
            arbitrage_type=ArbitrageType.SINGLE_MARKET,
            markets=[self.ticker],
            expected_profit=expected_profit * 100,
            expected_profit_pct=self.abs_edge,
            confidence=self.confidence,
            legs=[leg],
            max_quantity=quantity,
            total_cost=quantity * price / 100,
            fees=0.07 * (price / 100) * (1 - price / 100) * quantity,
            net_profit=expected_profit,
            metadata={
                "source": "weather_nws",
                "nws_probability": self.nws_probability,
                "market_prob": self.market_prob,
                "edge": self.edge,
                "explanation": self.explanation,
            },
        )


class WeatherTrader:
    """
    Trades weather markets using NWS forecast edge.

    Instead of using ML predictions, this directly compares NWS forecast
    probabilities to market prices. When NWS significantly disagrees with
    the market, we have an information advantage.
    """

    # Minimum edge to trade (10% - higher than ML because we trust NWS)
    MIN_EDGE = 0.10

    # Minimum NWS confidence to trade
    MIN_CONFIDENCE = 0.60

    # Price bounds (only trade uncertain markets)
    MIN_PRICE = 20
    MAX_PRICE = 80

    # Maximum hours until settlement (prefer near-term)
    MAX_HOURS_TO_EVENT = 72  # 3 days

    # Kelly fraction for position sizing
    KELLY_FRACTION = 0.15  # Conservative

    def __init__(
        self,
        weather_agent: "WeatherResearchAgent",
        api_client: "KalshiAPIClient",
        min_edge: float | None = None,
        min_confidence: float | None = None,
    ) -> None:
        """
        Initialize weather trader.

        Args:
            weather_agent: Weather research agent with NWS data
            api_client: Kalshi API client
            min_edge: Minimum edge threshold
            min_confidence: Minimum NWS confidence
        """
        self._weather_agent = weather_agent
        self._api_client = api_client
        self._min_edge = min_edge or self.MIN_EDGE
        self._min_confidence = min_confidence or self.MIN_CONFIDENCE

    async def find_opportunities(
        self,
        available_capital: float,
        max_opportunities: int = 5,
    ) -> list[WeatherTradingOpportunity]:
        """
        Find weather trading opportunities.

        Args:
            available_capital: Capital available for trading
            max_opportunities: Maximum opportunities to return

        Returns:
            List of WeatherTradingOpportunity sorted by edge
        """
        opportunities = []

        # Get all cached probability estimates from weather agent
        for ticker, estimate in self._weather_agent._probability_cache.items():
            opp = await self._evaluate_opportunity(ticker, estimate)
            if opp:
                opportunities.append(opp)

        # Sort by confidence-weighted edge (prefer high confidence + high edge)
        opportunities.sort(
            key=lambda o: o.abs_edge * o.confidence,
            reverse=True,
        )

        logger.info(
            f"Weather trader: found {len(opportunities)} opportunities "
            f"from {len(self._weather_agent._probability_cache)} markets"
        )

        return opportunities[:max_opportunities]

    async def _evaluate_opportunity(
        self,
        ticker: str,
        estimate: ProbabilityEstimate,
    ) -> WeatherTradingOpportunity | None:
        """Evaluate a single market for trading opportunity."""

        # Skip if too far out
        if estimate.hours_until_event > self.MAX_HOURS_TO_EVENT:
            logger.debug(f"Skipping {ticker}: {estimate.hours_until_event:.0f}h until event > {self.MAX_HOURS_TO_EVENT}h max")
            return None

        # Skip if low confidence
        if estimate.confidence < self._min_confidence:
            logger.debug(f"Skipping {ticker}: confidence {estimate.confidence:.0%} < {self._min_confidence:.0%}")
            return None

        # Get current market price
        try:
            market = await self._api_client.get_market(ticker)
        except Exception as e:
            logger.debug(f"Failed to get market {ticker}: {e}")
            return None

        market_price = market.last_price
        if market_price is None:
            if market.yes_bid is not None and market.yes_ask is not None:
                market_price = (market.yes_bid + market.yes_ask) // 2
            else:
                return None

        # Skip extreme prices
        if market_price <= self.MIN_PRICE or market_price >= self.MAX_PRICE:
            logger.debug(f"Skipping {ticker}: price {market_price}c outside {self.MIN_PRICE}-{self.MAX_PRICE}c range")
            return None

        # Calculate edge
        market_prob = market_price / 100.0
        nws_prob = estimate.probability

        edge = nws_prob - market_prob

        # Determine side
        if edge > 0:
            side = "yes"  # NWS says YES is underpriced
        else:
            side = "no"  # NWS says NO is underpriced

        abs_edge = abs(edge)

        # Check minimum edge
        if abs_edge < self._min_edge:
            logger.debug(f"Skipping {ticker}: edge {abs_edge:.1%} < {self._min_edge:.1%}")
            return None

        return WeatherTradingOpportunity(
            ticker=ticker,
            nws_probability=nws_prob,
            market_price=market_price,
            edge=edge,
            confidence=estimate.confidence,
            side=side,
            explanation=estimate.explanation,
            hours_until_event=estimate.hours_until_event,
        )

    def calculate_position_size(
        self,
        opportunity: WeatherTradingOpportunity,
        available_capital: float,
    ) -> tuple[int, float]:
        """
        Calculate position size using Kelly criterion.

        Returns:
            Tuple of (quantity, expected_profit)
        """
        # Kelly formula: f* = (bp - q) / b
        # where b = odds, p = win prob, q = lose prob

        if opportunity.side == "yes":
            win_prob = opportunity.nws_probability
            price = opportunity.market_price
        else:
            win_prob = 1 - opportunity.nws_probability
            price = 100 - opportunity.market_price

        lose_prob = 1 - win_prob
        payout_ratio = (100 - price) / price if price > 0 else 0

        if payout_ratio > 0:
            kelly = (payout_ratio * win_prob - lose_prob) / payout_ratio
        else:
            kelly = 0

        # Cap Kelly fraction
        kelly = max(0, min(kelly, self.KELLY_FRACTION))

        if kelly <= 0:
            return 0, 0.0

        # Calculate position
        position_value = available_capital * kelly
        cost_per_contract = price / 100.0
        quantity = int(position_value / cost_per_contract) if cost_per_contract > 0 else 0

        if quantity < 1:
            return 0, 0.0

        # Expected profit
        expected_cost = quantity * cost_per_contract
        expected_payout = quantity  # $1 per contract if correct
        fees = 0.07 * (price / 100) * (1 - price / 100) * quantity

        expected_profit = (win_prob * expected_payout) - (lose_prob * expected_cost) - fees

        return quantity, expected_profit

    async def get_trading_opportunities(
        self,
        available_capital: float,
        max_opportunities: int = 5,
    ) -> list[ArbitrageOpportunity]:
        """
        Get weather trading opportunities as ArbitrageOpportunity objects.

        This is the main interface for integration with the bot.
        """
        weather_opps = await self.find_opportunities(available_capital, max_opportunities * 2)

        result = []
        for opp in weather_opps:
            quantity, expected_profit = self.calculate_position_size(opp, available_capital)

            if quantity > 0 and expected_profit > 0:
                arb_opp = opp.to_arbitrage_opportunity(quantity, expected_profit)
                result.append(arb_opp)

                logger.info(
                    f"Weather opportunity: {opp.ticker} "
                    f"{opp.side.upper()} @ {opp.market_price}c, "
                    f"NWS={opp.nws_probability:.0%} vs Market={opp.market_prob:.0%}, "
                    f"edge={opp.abs_edge:.1%}, qty={quantity}"
                )

        return result[:max_opportunities]
