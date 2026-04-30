"""Polymarket API client for cross-market arbitrage.

Polymarket uses a CLOB (Central Limit Order Book) system. This client
fetches market data and prices for comparison with Kalshi markets.

API Documentation: https://docs.polymarket.com/

Note: This client is READ-ONLY for arbitrage signal detection.
Actual Polymarket trading requires separate infrastructure.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any
from enum import Enum

import httpx

from src.observability.logging import get_logger

logger = get_logger(__name__)


class PolymarketCategory(str, Enum):
    """Market categories on Polymarket."""

    POLITICS = "politics"
    CRYPTO = "crypto"
    SPORTS = "sports"
    CULTURE = "culture"
    SCIENCE = "science"
    BUSINESS = "business"


@dataclass
class PolymarketOutcome:
    """A single outcome in a Polymarket market."""

    outcome_id: str
    name: str
    price: float  # 0-1 probability

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome_id": self.outcome_id,
            "name": self.name,
            "price": self.price,
        }


@dataclass
class PolymarketMarket:
    """A Polymarket market with pricing info."""

    market_id: str
    question: str
    description: str
    category: str
    outcomes: list[PolymarketOutcome]
    volume: float
    liquidity: float
    end_date: datetime | None
    is_active: bool

    # For binary markets (YES/NO)
    yes_price: float | None = None
    no_price: float | None = None

    @property
    def is_binary(self) -> bool:
        """Check if this is a simple YES/NO market."""
        return len(self.outcomes) == 2

    @property
    def spread(self) -> float | None:
        """Calculate spread for binary markets."""
        if self.yes_price is not None and self.no_price is not None:
            return abs(1.0 - self.yes_price - self.no_price)
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "market_id": self.market_id,
            "question": self.question,
            "category": self.category,
            "outcomes": [o.to_dict() for o in self.outcomes],
            "volume": self.volume,
            "liquidity": self.liquidity,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "is_active": self.is_active,
            "yes_price": self.yes_price,
            "no_price": self.no_price,
            "is_binary": self.is_binary,
        }


@dataclass
class ArbitrageOpportunity:
    """Cross-market arbitrage opportunity."""

    kalshi_ticker: str
    polymarket_id: str

    kalshi_yes_price: float  # 0-1
    kalshi_no_price: float   # 0-1
    polymarket_yes_price: float
    polymarket_no_price: float

    # Arbitrage calculation
    # Buy YES on one platform, NO on other
    total_cost: float  # Cost to lock in both sides
    guaranteed_profit: float  # $1 - total_cost - fees

    # Direction: which platform to buy YES vs NO
    buy_yes_on: str  # "kalshi" or "polymarket"
    buy_no_on: str

    fees: float = 0.02  # Estimated total fees both sides

    @property
    def profit_pct(self) -> float:
        """Profit as percentage of cost."""
        if self.total_cost == 0:
            return 0.0
        return self.guaranteed_profit / self.total_cost

    @property
    def is_profitable(self) -> bool:
        """Check if arbitrage is profitable after fees."""
        return self.guaranteed_profit > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "kalshi_ticker": self.kalshi_ticker,
            "polymarket_id": self.polymarket_id,
            "kalshi_yes": self.kalshi_yes_price,
            "kalshi_no": self.kalshi_no_price,
            "polymarket_yes": self.polymarket_yes_price,
            "polymarket_no": self.polymarket_no_price,
            "total_cost": self.total_cost,
            "guaranteed_profit": self.guaranteed_profit,
            "profit_pct": self.profit_pct,
            "buy_yes_on": self.buy_yes_on,
            "buy_no_on": self.buy_no_on,
            "is_profitable": self.is_profitable,
        }


# Known market mappings between Kalshi and Polymarket
# Format: kalshi_pattern -> polymarket_slug
MARKET_MAPPINGS: dict[str, str] = {
    # BTC price markets (hourly)
    "BTC-": "bitcoin-",
    "ETH-": "ethereum-",

    # Elections
    "PRES-": "presidential-election-",

    # Fed rate decisions
    "FED-": "fed-",
    "FOMC-": "fomc-",
}


class PolymarketClient:
    """Client for Polymarket REST API.

    Provides market data fetching for arbitrage detection.
    This is a READ-ONLY client - no trading functionality.
    """

    # Polymarket CLOB API
    BASE_URL = "https://clob.polymarket.com"
    GAMMA_URL = "https://gamma-api.polymarket.com"

    def __init__(
        self,
        cache_ttl_seconds: int = 60,  # 1 minute cache
    ) -> None:
        """
        Initialize Polymarket client.

        Args:
            cache_ttl_seconds: How long to cache market data
        """
        self._cache_ttl = cache_ttl_seconds
        self._market_cache: dict[str, tuple[datetime, PolymarketMarket]] = {}
        self._client: httpx.AsyncClient | None = None

    async def connect(self) -> None:
        """Initialize HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=30,
                headers={"Accept": "application/json"},
            )

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> PolymarketClient:
        await self.connect()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    async def get_markets(
        self,
        category: PolymarketCategory | None = None,
        active_only: bool = True,
        limit: int = 100,
    ) -> list[PolymarketMarket]:
        """
        Fetch markets from Polymarket.

        Args:
            category: Filter by category
            active_only: Only return active markets
            limit: Maximum markets to return

        Returns:
            List of PolymarketMarket objects
        """
        await self.connect()
        if self._client is None:
            return []

        try:
            # Use Gamma API for market listing
            url = f"{self.GAMMA_URL}/markets"
            params: dict[str, Any] = {
                "limit": limit,
                "active": active_only,
            }

            if category:
                params["category"] = category.value

            response = await self._client.get(url, params=params)

            if response.status_code != 200:
                logger.error(f"Polymarket API error: {response.status_code}")
                return []

            data = response.json()
            markets = self._parse_markets(data)

            logger.info(f"Fetched {len(markets)} markets from Polymarket")
            return markets

        except Exception as e:
            logger.error(f"Failed to fetch Polymarket markets: {e}")
            return []

    async def get_market(self, market_id: str) -> PolymarketMarket | None:
        """
        Fetch a specific market by ID.

        Args:
            market_id: Polymarket market/condition ID

        Returns:
            PolymarketMarket or None if not found
        """
        # Check cache
        cache_key = market_id
        if cache_key in self._market_cache:
            cached_time, cached_market = self._market_cache[cache_key]
            age = (datetime.now(timezone.utc) - cached_time).total_seconds()
            if age < self._cache_ttl:
                return cached_market

        await self.connect()
        if self._client is None:
            return None

        try:
            url = f"{self.GAMMA_URL}/markets/{market_id}"
            response = await self._client.get(url)

            if response.status_code != 200:
                return None

            data = response.json()
            market = self._parse_single_market(data)

            if market:
                self._market_cache[cache_key] = (datetime.now(timezone.utc), market)

            return market

        except Exception as e:
            logger.error(f"Failed to fetch market {market_id}: {e}")
            return None

    async def get_orderbook(
        self,
        token_id: str,
    ) -> dict[str, Any]:
        """
        Fetch orderbook for a specific token.

        Args:
            token_id: The token/outcome ID

        Returns:
            Orderbook data with bids and asks
        """
        await self.connect()
        if self._client is None:
            return {}

        try:
            url = f"{self.BASE_URL}/book"
            params = {"token_id": token_id}

            response = await self._client.get(url, params=params)

            if response.status_code != 200:
                return {}

            return response.json()

        except Exception as e:
            logger.error(f"Failed to fetch orderbook: {e}")
            return {}

    async def get_price(self, token_id: str) -> float | None:
        """
        Get best price for a token.

        Args:
            token_id: The token/outcome ID

        Returns:
            Best price (0-1) or None
        """
        try:
            url = f"{self.BASE_URL}/price"
            params = {"token_id": token_id}

            await self.connect()
            if self._client is None:
                return None

            response = await self._client.get(url, params=params)

            if response.status_code != 200:
                return None

            data = response.json()
            return float(data.get("price", 0))

        except Exception as e:
            logger.error(f"Failed to get price: {e}")
            return None

    def _parse_markets(self, data: list[dict]) -> list[PolymarketMarket]:
        """Parse market list response."""
        markets = []

        for item in data:
            market = self._parse_single_market(item)
            if market:
                markets.append(market)

        return markets

    def _parse_single_market(self, data: dict) -> PolymarketMarket | None:
        """Parse a single market from API response."""
        try:
            outcomes = []
            tokens = data.get("tokens", [])

            yes_price = None
            no_price = None

            for token in tokens:
                outcome = PolymarketOutcome(
                    outcome_id=token.get("token_id", ""),
                    name=token.get("outcome", ""),
                    price=float(token.get("price", 0)),
                )
                outcomes.append(outcome)

                # Track YES/NO prices
                if outcome.name.upper() == "YES":
                    yes_price = outcome.price
                elif outcome.name.upper() == "NO":
                    no_price = outcome.price

            end_date = None
            if data.get("end_date_iso"):
                try:
                    end_date = datetime.fromisoformat(
                        data["end_date_iso"].replace("Z", "+00:00")
                    )
                except (ValueError, TypeError):
                    pass

            return PolymarketMarket(
                market_id=data.get("condition_id", data.get("id", "")),
                question=data.get("question", ""),
                description=data.get("description", ""),
                category=data.get("category", ""),
                outcomes=outcomes,
                volume=float(data.get("volume", 0)),
                liquidity=float(data.get("liquidity", 0)),
                end_date=end_date,
                is_active=data.get("active", True),
                yes_price=yes_price,
                no_price=no_price,
            )

        except Exception as e:
            logger.warning(f"Failed to parse market: {e}")
            return None

    def find_matching_market(
        self,
        kalshi_ticker: str,
        polymarket_markets: list[PolymarketMarket],
    ) -> PolymarketMarket | None:
        """
        Find Polymarket market matching a Kalshi ticker.

        Uses keyword matching and known mappings.

        Args:
            kalshi_ticker: Kalshi market ticker
            polymarket_markets: List of Polymarket markets to search

        Returns:
            Matching PolymarketMarket or None
        """
        ticker_upper = kalshi_ticker.upper()

        # Check known mappings
        for kalshi_prefix, poly_prefix in MARKET_MAPPINGS.items():
            if ticker_upper.startswith(kalshi_prefix):
                # Search for matching Polymarket market
                for market in polymarket_markets:
                    if poly_prefix in market.market_id.lower():
                        return market
                    if poly_prefix in market.question.lower():
                        return market

        # Keyword matching for BTC hourly markets
        if "BTC" in ticker_upper:
            for market in polymarket_markets:
                if "bitcoin" in market.question.lower():
                    return market

        return None

    def calculate_arbitrage(
        self,
        kalshi_ticker: str,
        kalshi_yes: float,
        kalshi_no: float,
        polymarket: PolymarketMarket,
        fees: float = 0.02,
    ) -> ArbitrageOpportunity | None:
        """
        Calculate arbitrage opportunity between platforms.

        Arbitrage exists when:
        - kalshi_yes + polymarket_no < 1.0 - fees (buy YES Kalshi, NO Poly)
        - OR kalshi_no + polymarket_yes < 1.0 - fees (buy NO Kalshi, YES Poly)

        Args:
            kalshi_ticker: Kalshi market ticker
            kalshi_yes: Kalshi YES price (0-1)
            kalshi_no: Kalshi NO price (0-1)
            polymarket: Polymarket market data
            fees: Estimated total fees

        Returns:
            ArbitrageOpportunity if profitable, None otherwise
        """
        if polymarket.yes_price is None or polymarket.no_price is None:
            return None

        poly_yes = polymarket.yes_price
        poly_no = polymarket.no_price

        # Strategy 1: Buy YES on Kalshi, NO on Polymarket
        cost_1 = kalshi_yes + poly_no
        profit_1 = 1.0 - cost_1 - fees

        # Strategy 2: Buy NO on Kalshi, YES on Polymarket
        cost_2 = kalshi_no + poly_yes
        profit_2 = 1.0 - cost_2 - fees

        # Choose better strategy
        if profit_1 > profit_2 and profit_1 > 0:
            return ArbitrageOpportunity(
                kalshi_ticker=kalshi_ticker,
                polymarket_id=polymarket.market_id,
                kalshi_yes_price=kalshi_yes,
                kalshi_no_price=kalshi_no,
                polymarket_yes_price=poly_yes,
                polymarket_no_price=poly_no,
                total_cost=cost_1,
                guaranteed_profit=profit_1,
                buy_yes_on="kalshi",
                buy_no_on="polymarket",
                fees=fees,
            )
        elif profit_2 > 0:
            return ArbitrageOpportunity(
                kalshi_ticker=kalshi_ticker,
                polymarket_id=polymarket.market_id,
                kalshi_yes_price=kalshi_yes,
                kalshi_no_price=kalshi_no,
                polymarket_yes_price=poly_yes,
                polymarket_no_price=poly_no,
                total_cost=cost_2,
                guaranteed_profit=profit_2,
                buy_yes_on="polymarket",
                buy_no_on="kalshi",
                fees=fees,
            )

        return None

    def get_cache_status(self) -> dict[str, Any]:
        """Get cache statistics."""
        return {
            "cached_markets": len(self._market_cache),
            "cache_ttl": self._cache_ttl,
        }
