"""Cross-market arbitrage strategy: exploit logical dependencies between markets."""

from __future__ import annotations

import re
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


class CrossMarketStrategy(ArbitrageStrategy):
    """
    Detects cross-market arbitrage opportunities.

    Exploits logical relationships between different markets:

    1. Subset relationships: P(A wins state X) <= P(A wins overall)
       - If "A wins Florida" is priced higher than "A wins election", arbitrage exists
       - Buy NO on state, buy YES on overall

    2. Temporal relationships: P(event by date X) <= P(event by date Y) where X < Y
       - If "Fed raises by March" > "Fed raises by June", arbitrage exists

    3. Magnitude relationships: P(value > X) <= P(value > Y) where X > Y
       - If "BTC > 100k" > "BTC > 90k", arbitrage exists

    This strategy requires identifying these relationships, which may need
    manual configuration or pattern matching on market titles.
    """

    RELATIONSHIP_PATTERNS = [
        {
            "type": "subset",
            "description": "State vs national election outcomes",
            "parent_pattern": "win",
            "child_patterns": ["win florida", "win georgia", "win arizona"],
        },
    ]

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._enabled = settings.arbitrage.enable_cross_market
        self._allow_dynamic_detection = getattr(
            settings.arbitrage, 'allow_dynamic_detection', False
        )
        self._configured_relationships: list[dict[str, Any]] = []

    def add_relationship(
        self,
        parent_ticker: str,
        child_ticker: str,
        relationship_type: str,
        description: str = "",
    ) -> None:
        """
        Add a known logical relationship between markets.

        Args:
            parent_ticker: The market that should have higher/equal probability
            child_ticker: The market that should have lower/equal probability
            relationship_type: Type of relationship (subset, temporal, magnitude)
            description: Human-readable description
        """
        self._configured_relationships.append({
            "parent": parent_ticker,
            "child": child_ticker,
            "type": relationship_type,
            "description": description,
        })

    async def detect(
        self,
        markets: list[MarketData],
        orderbooks: dict[str, OrderBook],
    ) -> list[ArbitrageOpportunity]:
        """
        Detect cross-market arbitrage opportunities.

        Checks configured relationships for price inconsistencies.
        """
        if not self._enabled:
            return []

        opportunities = []

        market_map = {m.ticker: m for m in markets}

        for relationship in self._configured_relationships:
            parent_ticker = relationship["parent"]
            child_ticker = relationship["child"]

            if parent_ticker not in market_map or child_ticker not in market_map:
                continue

            parent_market = market_map[parent_ticker]
            child_market = market_map[child_ticker]

            if not self._check_market_status(parent_market):
                continue
            if not self._check_market_status(child_market):
                continue

            parent_ob = orderbooks.get(parent_ticker)
            child_ob = orderbooks.get(child_ticker)

            if parent_ob is None or child_ob is None:
                continue

            opportunity = self._analyze_relationship(
                parent_market, child_market,
                parent_ob, child_ob,
                relationship,
            )

            if opportunity and self.validate_opportunity(opportunity):
                opportunities.append(opportunity)
                logger.info(
                    f"Cross-market arbitrage found: {parent_ticker} vs {child_ticker} "
                    f"profit={opportunity.net_profit:.4f}"
                )

        # Only run dynamic detection if explicitly enabled
        # DISABLED BY DEFAULT - dynamic detection is risky and may produce false positives
        if self._allow_dynamic_detection:
            dynamic_opps = await self._detect_dynamic_relationships(markets, orderbooks)
            opportunities.extend(dynamic_opps)

        return opportunities

    def _analyze_relationship(
        self,
        parent_market: MarketData,
        child_market: MarketData,
        parent_ob: OrderBook,
        child_ob: OrderBook,
        relationship: dict[str, Any],
    ) -> ArbitrageOpportunity | None:
        """
        Analyze a relationship for arbitrage.

        For subset relationships: child probability should <= parent probability.
        If child_yes_bid > parent_yes_ask, we can:
        - Sell YES on child (collect child_yes_bid)
        - Buy YES on parent (pay parent_yes_ask)

        But on Kalshi we can only buy, so we need:
        - Buy NO on child (pay child_no_ask = 100 - child_yes_bid)
        - Buy YES on parent (pay parent_yes_ask)

        Profit if: parent_yes_ask + child_no_ask < 100
        Which means: parent_yes_ask + (100 - child_yes_bid) < 100
        Simplified: parent_yes_ask < child_yes_bid
        """
        parent_yes_ask = parent_ob.best_yes_ask
        child_yes_bid = child_ob.best_yes_bid
        child_no_ask = child_ob.best_no_ask

        if parent_yes_ask is None or child_yes_bid is None or child_no_ask is None:
            return None

        if parent_yes_ask >= child_yes_bid:
            return None

        total_cost_cents = parent_yes_ask + child_no_ask

        if total_cost_cents >= 100:
            return None

        gross_profit_cents = 100 - total_cost_cents

        parent_quantity = parent_ob.yes_ask_quantity
        child_quantity = child_ob.no_ask_quantity
        max_quantity = min(parent_quantity, child_quantity)

        if max_quantity < self._min_liquidity:
            return None

        legs: list[dict[str, Any]] = [
            {
                "market": parent_market.ticker,
                "side": Side.YES.value,
                "action": "buy",
                "price": parent_yes_ask,
                "quantity": max_quantity,
                "rationale": "Parent event (higher probability)",
            },
            {
                "market": child_market.ticker,
                "side": Side.NO.value,
                "action": "buy",
                "price": child_no_ask,
                "quantity": max_quantity,
                "rationale": "Child event (subset, should be lower probability)",
            },
        ]

        total_fees = self.calculate_total_fees(legs)

        total_cost_dollars = (total_cost_cents * max_quantity) / 100
        gross_profit_dollars = (gross_profit_cents * max_quantity) / 100
        net_profit_dollars = gross_profit_dollars - total_fees

        if net_profit_dollars <= 0:
            return None

        return ArbitrageOpportunity(
            opportunity_id=f"cm-{parent_market.ticker}-{child_market.ticker}-{uuid.uuid4().hex[:8]}",
            arbitrage_type=ArbitrageType.CROSS_MARKET,
            markets=[parent_market.ticker, child_market.ticker],
            expected_profit=gross_profit_cents,
            expected_profit_pct=gross_profit_cents / total_cost_cents,
            confidence=self._calculate_confidence(parent_ob, child_ob, max_quantity),
            legs=legs,
            max_quantity=max_quantity,
            total_cost=total_cost_dollars,
            fees=total_fees,
            net_profit=net_profit_dollars,
            detected_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(seconds=30),
            metadata={
                "relationship_type": relationship["type"],
                "parent_market": parent_market.ticker,
                "child_market": child_market.ticker,
                "parent_yes_ask": parent_yes_ask,
                "child_yes_bid": child_yes_bid,
                "child_no_ask": child_no_ask,
                "description": relationship.get("description", ""),
            },
        )

    async def _detect_dynamic_relationships(
        self,
        markets: list[MarketData],
        orderbooks: dict[str, OrderBook],
    ) -> list[ArbitrageOpportunity]:
        """
        Attempt to detect relationships dynamically based on market titles.

        Detects:
        1. Magnitude relationships: "BTC > 100k" vs "BTC > 90k"
        2. Temporal relationships: "Event by March" vs "Event by June"
        3. Subset relationships: "Win Florida" vs "Win Election"

        This is a heuristic approach and may produce false positives.
        Manual verification of relationships is recommended.
        """
        opportunities = []

        # Group markets by event for easier comparison
        event_markets: dict[str, list[MarketData]] = {}
        for market in markets:
            if market.event_ticker:
                if market.event_ticker not in event_markets:
                    event_markets[market.event_ticker] = []
                event_markets[market.event_ticker].append(market)

        # Also group by title keywords for cross-event detection
        keyword_markets: dict[str, list[MarketData]] = {}
        for market in markets:
            # Extract key terms from title
            title_lower = market.title.lower()
            keywords = self._extract_keywords(title_lower)
            for keyword in keywords:
                if keyword not in keyword_markets:
                    keyword_markets[keyword] = []
                keyword_markets[keyword].append(market)

        # 1. Detect magnitude relationships (e.g., "above $X" markets)
        magnitude_opps = self._detect_magnitude_relationships(
            markets, orderbooks, keyword_markets
        )
        opportunities.extend(magnitude_opps)

        # 2. Detect temporal relationships (e.g., "by date X" markets)
        temporal_opps = self._detect_temporal_relationships(
            markets, orderbooks, event_markets
        )
        opportunities.extend(temporal_opps)

        if opportunities:
            logger.info(
                f"Dynamic relationship detection found {len(opportunities)} opportunities"
            )

        return opportunities

    def _extract_keywords(self, title: str) -> list[str]:
        """Extract meaningful keywords from market title."""
        keywords = []

        # Bitcoin/crypto keywords
        if any(term in title for term in ["bitcoin", "btc", "crypto"]):
            keywords.append("bitcoin")

        # Stock market keywords
        if any(term in title for term in ["s&p", "dow", "nasdaq", "spy"]):
            keywords.append("stocks")

        # Election keywords
        if any(term in title for term in ["win", "election", "president", "vote"]):
            keywords.append("election")

        # Economic keywords
        if any(term in title for term in ["fed", "rate", "inflation", "gdp", "unemployment"]):
            keywords.append("economy")

        # Weather keywords
        if any(term in title for term in ["temperature", "weather", "hurricane", "storm"]):
            keywords.append("weather")

        return keywords

    def _detect_magnitude_relationships(
        self,
        markets: list[MarketData],
        orderbooks: dict[str, OrderBook],
        keyword_markets: dict[str, list[MarketData]],
    ) -> list[ArbitrageOpportunity]:
        """
        Detect magnitude relationship violations.

        If "BTC > 100k" is priced higher than "BTC > 90k", there's arbitrage
        since 100k > 90k means the first implies the second.

        Pattern: P(X > higher_threshold) should be <= P(X > lower_threshold)
        """
        opportunities = []

        # Patterns to detect magnitude thresholds in titles
        magnitude_patterns = [
            # "above $100,000" or "above 100000" or "above $100k"
            r'above\s*\$?([\d,]+(?:\.\d+)?[kmb]?)',
            # "over $100,000"
            r'over\s*\$?([\d,]+(?:\.\d+)?[kmb]?)',
            # "greater than $100,000"
            r'greater\s+than\s*\$?([\d,]+(?:\.\d+)?[kmb]?)',
            # "> $100,000" or ">$100k"
            r'>\s*\$?([\d,]+(?:\.\d+)?[kmb]?)',
            # "at least $100,000"
            r'at\s+least\s*\$?([\d,]+(?:\.\d+)?[kmb]?)',
            # "reach $100,000"
            r'reach\s*\$?([\d,]+(?:\.\d+)?[kmb]?)',
        ]

        # Group markets that might have magnitude relationships
        for keyword in ["bitcoin", "stocks"]:
            if keyword not in keyword_markets:
                continue

            related_markets = keyword_markets[keyword]
            if len(related_markets) < 2:
                continue

            # Extract threshold values from each market
            threshold_markets: list[tuple[float, MarketData]] = []

            for market in related_markets:
                title_lower = market.title.lower()

                for pattern in magnitude_patterns:
                    match = re.search(pattern, title_lower)
                    if match:
                        value_str = match.group(1).replace(",", "").lower()

                        # Handle k/m/b suffixes
                        multiplier = 1
                        if value_str.endswith("k"):
                            multiplier = 1000
                            value_str = value_str[:-1]
                        elif value_str.endswith("m"):
                            multiplier = 1000000
                            value_str = value_str[:-1]
                        elif value_str.endswith("b"):
                            multiplier = 1000000000
                            value_str = value_str[:-1]

                        try:
                            threshold = float(value_str) * multiplier
                            threshold_markets.append((threshold, market))
                            break
                        except ValueError:
                            continue

            if len(threshold_markets) < 2:
                continue

            # Sort by threshold value (descending)
            threshold_markets.sort(key=lambda x: x[0], reverse=True)

            # Check for inversions: higher threshold should have lower/equal probability
            for i in range(len(threshold_markets) - 1):
                higher_threshold, higher_market = threshold_markets[i]
                lower_threshold, lower_market = threshold_markets[i + 1]

                higher_ob = orderbooks.get(higher_market.ticker)
                lower_ob = orderbooks.get(lower_market.ticker)

                if higher_ob is None or lower_ob is None:
                    continue

                # Get YES ask prices (cost to buy YES)
                higher_yes_ask = higher_ob.best_yes_ask
                lower_yes_ask = lower_ob.best_yes_ask

                if higher_yes_ask is None or lower_yes_ask is None:
                    continue

                # VIOLATION: Higher threshold priced higher than lower threshold
                # This shouldn't happen: P(X > 100k) should be <= P(X > 90k)
                if higher_yes_ask > lower_yes_ask:
                    # Arbitrage: Buy NO on higher threshold, Buy YES on lower threshold
                    # Because if X > lower_threshold, we might or might not have X > higher_threshold
                    # But the market is pricing it backwards

                    higher_no_ask = higher_ob.best_no_ask
                    if higher_no_ask is None:
                        continue

                    # Total cost should be < 100 for guaranteed profit
                    total_cost_cents = lower_yes_ask + higher_no_ask

                    if total_cost_cents >= 100:
                        # No guaranteed arbitrage, but there's still a pricing anomaly
                        # This could be value betting opportunity
                        continue

                    gross_profit_cents = 100 - total_cost_cents
                    max_quantity = min(
                        lower_ob.yes_ask_quantity,
                        higher_ob.no_ask_quantity,
                        100,
                    )

                    if max_quantity < self._min_liquidity:
                        continue

                    legs: list[dict[str, Any]] = [
                        {
                            "market": lower_market.ticker,
                            "side": Side.YES.value,
                            "action": "buy",
                            "price": lower_yes_ask,
                            "quantity": max_quantity,
                            "rationale": f"Lower threshold ({lower_threshold}) - should have higher prob",
                        },
                        {
                            "market": higher_market.ticker,
                            "side": Side.NO.value,
                            "action": "buy",
                            "price": higher_no_ask,
                            "quantity": max_quantity,
                            "rationale": f"Higher threshold ({higher_threshold}) - mispriced higher",
                        },
                    ]

                    total_fees = self.calculate_total_fees(legs)
                    total_cost_dollars = (total_cost_cents * max_quantity) / 100
                    gross_profit_dollars = (gross_profit_cents * max_quantity) / 100
                    net_profit_dollars = gross_profit_dollars - total_fees

                    if net_profit_dollars <= 0:
                        continue

                    opp = ArbitrageOpportunity(
                        opportunity_id=f"cm-mag-{higher_market.ticker[:10]}-{uuid.uuid4().hex[:8]}",
                        arbitrage_type=ArbitrageType.CROSS_MARKET,
                        markets=[lower_market.ticker, higher_market.ticker],
                        expected_profit=gross_profit_cents,
                        expected_profit_pct=gross_profit_cents / total_cost_cents,
                        confidence=self._calculate_confidence(lower_ob, higher_ob, max_quantity),
                        legs=legs,
                        max_quantity=max_quantity,
                        total_cost=total_cost_dollars,
                        fees=total_fees,
                        net_profit=net_profit_dollars,
                        detected_at=datetime.utcnow(),
                        expires_at=datetime.utcnow() + timedelta(seconds=30),
                        metadata={
                            "relationship_type": "magnitude",
                            "higher_threshold": higher_threshold,
                            "lower_threshold": lower_threshold,
                            "higher_yes_ask": higher_yes_ask,
                            "lower_yes_ask": lower_yes_ask,
                            "description": f"Magnitude inversion: {higher_threshold} priced above {lower_threshold}",
                        },
                    )

                    if self.validate_opportunity(opp):
                        opportunities.append(opp)
                        logger.info(
                            f"Magnitude relationship arbitrage: {higher_market.ticker} vs {lower_market.ticker} "
                            f"profit={net_profit_dollars:.4f}"
                        )

        return opportunities

    def _detect_temporal_relationships(
        self,
        markets: list[MarketData],
        orderbooks: dict[str, OrderBook],
        event_markets: dict[str, list[MarketData]],
    ) -> list[ArbitrageOpportunity]:
        """
        Detect temporal relationship violations.

        If "Event by March" is priced higher than "Event by June", there's arbitrage
        since an earlier deadline implies a stricter condition.

        Pattern: P(Event by earlier_date) should be <= P(Event by later_date)
        """
        opportunities = []

        # Month patterns to detect deadlines
        month_patterns = [
            (r'\b(january|jan)\b', 1),
            (r'\b(february|feb)\b', 2),
            (r'\b(march|mar)\b', 3),
            (r'\b(april|apr)\b', 4),
            (r'\b(may)\b', 5),
            (r'\b(june|jun)\b', 6),
            (r'\b(july|jul)\b', 7),
            (r'\b(august|aug)\b', 8),
            (r'\b(september|sep|sept)\b', 9),
            (r'\b(october|oct)\b', 10),
            (r'\b(november|nov)\b', 11),
            (r'\b(december|dec)\b', 12),
        ]

        # Quarter patterns
        quarter_patterns = [
            (r'\bq1\b', 1),
            (r'\bq2\b', 2),
            (r'\bq3\b', 3),
            (r'\bq4\b', 4),
        ]

        # Check within each event for temporal relationships
        for event_ticker, related_markets in event_markets.items():
            if len(related_markets) < 2:
                continue

            # Try to extract temporal info from each market
            dated_markets: list[tuple[int, MarketData]] = []

            for market in related_markets:
                title_lower = market.title.lower()

                # Check for "by [month]" or "before [month]"
                temporal_match = re.search(r'\b(by|before|through|end of)\b', title_lower)
                if not temporal_match:
                    continue

                # Find the month
                month_val = None
                for pattern, month in month_patterns:
                    if re.search(pattern, title_lower):
                        month_val = month
                        break

                # Also check quarters
                if month_val is None:
                    for pattern, quarter in quarter_patterns:
                        if re.search(pattern, title_lower):
                            month_val = quarter * 3  # Q1=3, Q2=6, etc.
                            break

                if month_val is not None:
                    dated_markets.append((month_val, market))

            if len(dated_markets) < 2:
                continue

            # Sort by date (ascending - earlier first)
            dated_markets.sort(key=lambda x: x[0])

            # Check for inversions: earlier deadline should have lower/equal probability
            for i in range(len(dated_markets) - 1):
                earlier_month, earlier_market = dated_markets[i]
                later_month, later_market = dated_markets[i + 1]

                if earlier_month >= later_month:
                    continue

                earlier_ob = orderbooks.get(earlier_market.ticker)
                later_ob = orderbooks.get(later_market.ticker)

                if earlier_ob is None or later_ob is None:
                    continue

                earlier_yes_ask = earlier_ob.best_yes_ask
                later_yes_ask = later_ob.best_yes_ask

                if earlier_yes_ask is None or later_yes_ask is None:
                    continue

                # VIOLATION: Earlier deadline priced higher than later deadline
                # P(by March) should be <= P(by June)
                if earlier_yes_ask > later_yes_ask:
                    earlier_no_ask = earlier_ob.best_no_ask
                    if earlier_no_ask is None:
                        continue

                    total_cost_cents = later_yes_ask + earlier_no_ask

                    if total_cost_cents >= 100:
                        continue

                    gross_profit_cents = 100 - total_cost_cents
                    max_quantity = min(
                        later_ob.yes_ask_quantity,
                        earlier_ob.no_ask_quantity,
                        100,
                    )

                    if max_quantity < self._min_liquidity:
                        continue

                    legs: list[dict[str, Any]] = [
                        {
                            "market": later_market.ticker,
                            "side": Side.YES.value,
                            "action": "buy",
                            "price": later_yes_ask,
                            "quantity": max_quantity,
                            "rationale": f"Later deadline (month {later_month}) - should have higher prob",
                        },
                        {
                            "market": earlier_market.ticker,
                            "side": Side.NO.value,
                            "action": "buy",
                            "price": earlier_no_ask,
                            "quantity": max_quantity,
                            "rationale": f"Earlier deadline (month {earlier_month}) - mispriced higher",
                        },
                    ]

                    total_fees = self.calculate_total_fees(legs)
                    total_cost_dollars = (total_cost_cents * max_quantity) / 100
                    gross_profit_dollars = (gross_profit_cents * max_quantity) / 100
                    net_profit_dollars = gross_profit_dollars - total_fees

                    if net_profit_dollars <= 0:
                        continue

                    opp = ArbitrageOpportunity(
                        opportunity_id=f"cm-temp-{earlier_market.ticker[:10]}-{uuid.uuid4().hex[:8]}",
                        arbitrage_type=ArbitrageType.CROSS_MARKET,
                        markets=[later_market.ticker, earlier_market.ticker],
                        expected_profit=gross_profit_cents,
                        expected_profit_pct=gross_profit_cents / total_cost_cents,
                        confidence=self._calculate_confidence(later_ob, earlier_ob, max_quantity),
                        legs=legs,
                        max_quantity=max_quantity,
                        total_cost=total_cost_dollars,
                        fees=total_fees,
                        net_profit=net_profit_dollars,
                        detected_at=datetime.utcnow(),
                        expires_at=datetime.utcnow() + timedelta(seconds=30),
                        metadata={
                            "relationship_type": "temporal",
                            "earlier_month": earlier_month,
                            "later_month": later_month,
                            "earlier_yes_ask": earlier_yes_ask,
                            "later_yes_ask": later_yes_ask,
                            "description": f"Temporal inversion: month {earlier_month} priced above month {later_month}",
                        },
                    )

                    if self.validate_opportunity(opp):
                        opportunities.append(opp)
                        logger.info(
                            f"Temporal relationship arbitrage: {earlier_market.ticker} vs {later_market.ticker} "
                            f"profit={net_profit_dollars:.4f}"
                        )

        return opportunities

    def _calculate_confidence(
        self,
        parent_ob: OrderBook,
        child_ob: OrderBook,
        quantity: int,
    ) -> float:
        """
        Calculate confidence for cross-market opportunity.

        Lower base confidence due to:
        - Reliance on logical relationships
        - Execution timing risk across markets
        """
        base_confidence = 0.65

        parent_depth = sum(level.quantity for level in parent_ob.yes_asks[:3])
        child_depth = sum(level.quantity for level in child_ob.no_asks[:3])
        min_depth = min(parent_depth, child_depth)

        depth_factor = min((min_depth / 100) * 0.15, 0.15)

        confidence = base_confidence + depth_factor
        return min(confidence, 0.90)
