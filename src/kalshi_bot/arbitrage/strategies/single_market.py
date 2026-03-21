"""Single market arbitrage strategy: YES + NO < 100 cents."""

from __future__ import annotations

import uuid
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


class SingleMarketStrategy(ArbitrageStrategy):
    """
    Detects single market arbitrage opportunities.

    In prediction markets, YES and NO contracts are complementary:
    - If YES wins, YES pays $1, NO pays $0
    - If NO wins, NO pays $1, YES pays $0

    If you can buy both YES and NO for less than $1 total,
    you're guaranteed a profit at settlement.

    Example:
    - Buy YES at 45 cents
    - Buy NO at 52 cents
    - Total cost: 97 cents
    - Guaranteed payout: $1.00
    - Gross profit: 3 cents per contract (before fees)
    """

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._enabled = settings.arbitrage.enable_single_market

    async def detect(
        self,
        markets: list[MarketData],
        orderbooks: dict[str, OrderBook],
    ) -> list[ArbitrageOpportunity]:
        """
        Detect single market arbitrage opportunities.

        Looks for markets where:
        - best_yes_ask + best_no_ask < 100 cents
        - After accounting for fees, there's still profit
        """
        if not self._enabled:
            return []

        opportunities = []

        for market in markets:
            if not self._check_market_status(market):
                continue

            orderbook = orderbooks.get(market.ticker)
            if orderbook is None:
                continue

            opportunity = self._analyze_market(market, orderbook)
            if opportunity and self.validate_opportunity(opportunity):
                opportunities.append(opportunity)
                logger.info(
                    f"Single market arbitrage found: {market.ticker} "
                    f"profit={opportunity.net_profit:.4f}"
                )

        return opportunities

    def _analyze_market(
        self,
        market: MarketData,
        orderbook: OrderBook,
    ) -> ArbitrageOpportunity | None:
        """
        Analyze a single market for arbitrage.

        Args:
            market: Market data
            orderbook: Order book for the market

        Returns:
            ArbitrageOpportunity if found, None otherwise
        """
        yes_ask = orderbook.best_yes_ask
        no_ask = orderbook.best_no_ask

        if yes_ask is None or no_ask is None:
            return None

        if yes_ask <= 0 or yes_ask >= 100:
            return None
        if no_ask <= 0 or no_ask >= 100:
            return None

        total_cost_cents = yes_ask + no_ask

        if total_cost_cents >= 100:
            return None

        gross_profit_cents = 100 - total_cost_cents

        yes_quantity = orderbook.yes_ask_quantity
        no_quantity = orderbook.no_ask_quantity
        max_quantity = min(yes_quantity, no_quantity)

        if max_quantity < self._min_liquidity:
            return None

        legs: list[dict[str, Any]] = [
            {
                "market": market.ticker,
                "side": Side.YES.value,
                "action": "buy",
                "price": yes_ask,
                "quantity": max_quantity,
            },
            {
                "market": market.ticker,
                "side": Side.NO.value,
                "action": "buy",
                "price": no_ask,
                "quantity": max_quantity,
            },
        ]

        total_fees = self.calculate_total_fees(legs)

        total_cost_dollars = (total_cost_cents * max_quantity) / 100
        gross_profit_dollars = (gross_profit_cents * max_quantity) / 100
        net_profit_dollars = gross_profit_dollars - total_fees

        if net_profit_dollars <= 0:
            return None

        return ArbitrageOpportunity(
            opportunity_id=f"sm-{market.ticker}-{uuid.uuid4().hex[:8]}",
            arbitrage_type=ArbitrageType.SINGLE_MARKET,
            markets=[market.ticker],
            expected_profit=gross_profit_cents,
            expected_profit_pct=gross_profit_cents / total_cost_cents,
            confidence=self._calculate_confidence(orderbook, max_quantity),
            legs=legs,
            max_quantity=max_quantity,
            total_cost=total_cost_dollars,
            fees=total_fees,
            net_profit=net_profit_dollars,
            detected_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(seconds=30),
            metadata={
                "yes_ask": yes_ask,
                "no_ask": no_ask,
                "gross_profit_cents": gross_profit_cents,
                "market_title": market.title,
            },
        )

    def _calculate_confidence(
        self,
        orderbook: OrderBook,
        quantity: int,
    ) -> float:
        """
        Calculate confidence score for the opportunity.

        Factors:
        - Depth of order book (more depth = more confidence)
        - Spread tightness
        - Volume available
        """
        base_confidence = 0.8

        yes_depth = sum(level.quantity for level in orderbook.yes_asks[:3])
        no_depth = sum(level.quantity for level in orderbook.no_asks[:3])
        min_depth = min(yes_depth, no_depth)

        depth_factor = min(min_depth / 100, 0.15)

        spread = (orderbook.best_yes_ask or 0) - (orderbook.best_yes_bid or 0)
        if spread <= 2:
            spread_factor = 0.05
        elif spread <= 5:
            spread_factor = 0.02
        else:
            spread_factor = 0.0

        confidence = min(base_confidence + depth_factor + spread_factor, 0.98)

        return confidence
