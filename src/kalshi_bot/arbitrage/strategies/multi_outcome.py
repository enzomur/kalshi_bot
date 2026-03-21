"""Multi-outcome arbitrage strategy: Sum of outcomes < 100 cents."""

from __future__ import annotations

import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

from kalshi_bot.arbitrage.strategies.base import ArbitrageStrategy
from kalshi_bot.config.settings import Settings
from kalshi_bot.core.types import (
    ArbitrageOpportunity,
    ArbitrageType,
    MarketData,
    OrderBook,
    Side,
)
from kalshi_bot.utils.logging import get_logger

logger = get_logger(__name__)


class MultiOutcomeStrategy(ArbitrageStrategy):
    """
    Detects multi-outcome arbitrage opportunities.

    When an event has multiple mutually exclusive outcomes (e.g., "Who will win?"),
    the sum of all YES prices should equal 100 cents (since exactly one must win).

    If sum of all YES asks < 100 cents, buy all outcomes for guaranteed profit.

    Example (3-way race):
    - Candidate A YES: 35 cents
    - Candidate B YES: 30 cents
    - Candidate C YES: 32 cents
    - Total: 97 cents
    - One outcome MUST pay $1
    - Gross profit: 3 cents per contract set (before fees)
    """

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._enabled = settings.arbitrage.enable_multi_outcome

    async def detect(
        self,
        markets: list[MarketData],
        orderbooks: dict[str, OrderBook],
    ) -> list[ArbitrageOpportunity]:
        """
        Detect multi-outcome arbitrage opportunities.

        Groups markets by event and checks if sum of YES asks < 100.
        """
        if not self._enabled:
            return []

        events = self._group_by_event(markets)
        opportunities = []

        for event_ticker, event_markets in events.items():
            if len(event_markets) < 2:
                continue

            event_orderbooks = {
                m.ticker: orderbooks[m.ticker]
                for m in event_markets
                if m.ticker in orderbooks
            }

            if len(event_orderbooks) != len(event_markets):
                continue

            opportunity = self._analyze_event(event_ticker, event_markets, event_orderbooks)
            if opportunity and self.validate_opportunity(opportunity):
                opportunities.append(opportunity)
                logger.info(
                    f"Multi-outcome arbitrage found: {event_ticker} "
                    f"({len(event_markets)} markets) profit={opportunity.net_profit:.4f}"
                )

        return opportunities

    def _group_by_event(
        self,
        markets: list[MarketData],
    ) -> dict[str, list[MarketData]]:
        """Group markets by their event ticker."""
        events: dict[str, list[MarketData]] = defaultdict(list)

        for market in markets:
            if self._check_market_status(market) and market.event_ticker:
                events[market.event_ticker].append(market)

        return events

    def _analyze_event(
        self,
        event_ticker: str,
        markets: list[MarketData],
        orderbooks: dict[str, OrderBook],
    ) -> ArbitrageOpportunity | None:
        """
        Analyze an event's markets for multi-outcome arbitrage.

        Args:
            event_ticker: Event ticker
            markets: Markets belonging to this event
            orderbooks: Order books for each market

        Returns:
            ArbitrageOpportunity if found, None otherwise
        """
        total_cost_cents = 0
        legs: list[dict[str, Any]] = []
        min_quantity = float("inf")

        for market in markets:
            orderbook = orderbooks.get(market.ticker)
            if orderbook is None:
                return None

            yes_ask = orderbook.best_yes_ask
            if yes_ask is None or yes_ask <= 0:
                return None

            total_cost_cents += yes_ask

            quantity = orderbook.yes_ask_quantity
            if quantity < self._min_liquidity:
                return None

            min_quantity = min(min_quantity, quantity)

            legs.append({
                "market": market.ticker,
                "side": Side.YES.value,
                "action": "buy",
                "price": yes_ask,
                "quantity": 0,
                "market_title": market.title,
            })

        if min_quantity == float("inf"):
            return None

        max_quantity = int(min_quantity)

        if total_cost_cents >= 100:
            return None

        gross_profit_cents = 100 - total_cost_cents

        if gross_profit_cents < self._min_profit_cents:
            return None

        for leg in legs:
            leg["quantity"] = max_quantity

        total_fees = self.calculate_total_fees(legs)

        total_cost_dollars = (total_cost_cents * max_quantity) / 100
        gross_profit_dollars = (gross_profit_cents * max_quantity) / 100
        net_profit_dollars = gross_profit_dollars - total_fees

        if net_profit_dollars <= 0:
            return None

        return ArbitrageOpportunity(
            opportunity_id=f"mo-{event_ticker}-{uuid.uuid4().hex[:8]}",
            arbitrage_type=ArbitrageType.MULTI_OUTCOME,
            markets=[m.ticker for m in markets],
            expected_profit=gross_profit_cents,
            expected_profit_pct=gross_profit_cents / total_cost_cents,
            confidence=self._calculate_confidence(orderbooks, max_quantity),
            legs=legs,
            max_quantity=max_quantity,
            total_cost=total_cost_dollars,
            fees=total_fees,
            net_profit=net_profit_dollars,
            detected_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(seconds=30),
            metadata={
                "event_ticker": event_ticker,
                "num_outcomes": len(markets),
                "total_cost_cents": total_cost_cents,
                "gross_profit_cents": gross_profit_cents,
            },
        )

    def _calculate_confidence(
        self,
        orderbooks: dict[str, OrderBook],
        quantity: int,
    ) -> float:
        """
        Calculate confidence score for multi-outcome opportunity.

        Lower confidence than single market due to:
        - More legs = more execution risk
        - Prices may move before all legs fill
        """
        base_confidence = 0.7

        num_legs = len(orderbooks)
        leg_penalty = 0.03 * (num_legs - 2)

        min_depth = float("inf")
        for orderbook in orderbooks.values():
            depth = sum(level.quantity for level in orderbook.yes_asks[:3])
            min_depth = min(min_depth, depth)

        depth_factor = min((min_depth / 100) * 0.1, 0.1)

        confidence = max(base_confidence - leg_penalty + depth_factor, 0.5)
        return min(confidence, 0.95)
