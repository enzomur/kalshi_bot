"""Cross-Market Arbitrage Strategy.

Detects arbitrage opportunities between Kalshi and Polymarket.
When the combined cost of YES on one platform + NO on the other
is less than $1.00 minus fees, we have a risk-free profit.

This is the highest-confidence strategy as arbitrage is mathematically
guaranteed (assuming execution).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from src.core.types import Signal
from src.data.polymarket_client import (
    PolymarketClient,
    PolymarketMarket,
    ArbitrageOpportunity,
)
from src.strategies.base import Strategy
from src.observability.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ArbitrageSignalMetadata:
    """Metadata for arbitrage signals."""

    kalshi_ticker: str
    polymarket_id: str
    kalshi_yes: float
    kalshi_no: float
    poly_yes: float
    poly_no: float
    total_cost: float
    guaranteed_profit: float
    profit_pct: float
    buy_yes_on: str
    buy_no_on: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "kalshi_ticker": self.kalshi_ticker,
            "polymarket_id": self.polymarket_id,
            "kalshi_yes": self.kalshi_yes,
            "kalshi_no": self.kalshi_no,
            "poly_yes": self.poly_yes,
            "poly_no": self.poly_no,
            "total_cost": self.total_cost,
            "guaranteed_profit": self.guaranteed_profit,
            "profit_pct": self.profit_pct,
            "buy_yes_on": self.buy_yes_on,
            "buy_no_on": self.buy_no_on,
        }


class ArbitrageStrategy(Strategy):
    """
    Cross-market arbitrage strategy between Kalshi and Polymarket.

    Detects when combined prices create risk-free profit opportunity.
    This is the highest-weight strategy as arbitrage is mathematically
    guaranteed when executed properly.

    Note: This strategy only generates signals for the Kalshi side.
    Actual cross-platform execution requires separate infrastructure.
    """

    # Arbitrage thresholds
    DEFAULT_MIN_PROFIT = 0.01  # 1% minimum profit after fees
    DEFAULT_FEES = 0.02  # 2% estimated total fees (both platforms)

    # Market filters
    MIN_LIQUIDITY = 1000  # Minimum combined liquidity
    MIN_VOLUME = 500  # Minimum 24h volume

    # Confidence is high for arbitrage (mathematically guaranteed)
    ARBITRAGE_CONFIDENCE = 0.95

    def __init__(
        self,
        polymarket_client: PolymarketClient | None = None,
        db=None,
        enabled: bool = True,
        min_profit: float = DEFAULT_MIN_PROFIT,
        fees: float = DEFAULT_FEES,
    ) -> None:
        """
        Initialize arbitrage strategy.

        Args:
            polymarket_client: Client for Polymarket API
            db: Optional database connection
            enabled: Whether strategy is active
            min_profit: Minimum profit percentage to signal
            fees: Estimated total fees for both platforms
        """
        # Arbitrage has very low edge threshold since profit is guaranteed
        super().__init__(
            db=db,
            enabled=enabled,
            min_edge=0.01,  # 1% edge for arbitrage
            min_confidence=0.90,
        )
        self._polymarket = polymarket_client
        self._min_profit = min_profit
        self._fees = fees

        # Cache for Polymarket markets
        self._poly_markets: list[PolymarketMarket] = []
        self._cache_time: datetime | None = None
        self._cache_ttl = 120  # 2 minute cache

    @property
    def name(self) -> str:
        return "arbitrage"

    async def generate_signals(
        self,
        markets: list[dict[str, Any]],
    ) -> list[Signal]:
        """
        Scan for arbitrage opportunities between Kalshi and Polymarket.

        Args:
            markets: Kalshi market data

        Returns:
            List of arbitrage signals (for Kalshi side only)
        """
        if not self._enabled:
            return []

        if self._polymarket is None:
            logger.debug("No Polymarket client configured")
            return []

        signals = []
        opportunities_found = 0
        markets_checked = 0

        # Refresh Polymarket cache
        await self._refresh_polymarket_cache()

        if not self._poly_markets:
            logger.debug("No Polymarket markets available")
            return []

        for market in markets:
            ticker = market.get("ticker", "")

            # Get Kalshi prices
            kalshi_yes, kalshi_no = self._get_kalshi_prices(market)
            if kalshi_yes is None or kalshi_no is None:
                continue

            markets_checked += 1

            # Find matching Polymarket market
            poly_market = self._polymarket.find_matching_market(
                ticker, self._poly_markets
            )
            if poly_market is None:
                continue

            # Check liquidity requirements
            if poly_market.liquidity < self.MIN_LIQUIDITY:
                continue

            # Calculate arbitrage opportunity
            arb = self._polymarket.calculate_arbitrage(
                kalshi_ticker=ticker,
                kalshi_yes=kalshi_yes,
                kalshi_no=kalshi_no,
                polymarket=poly_market,
                fees=self._fees,
            )

            if arb is None or not arb.is_profitable:
                continue

            if arb.profit_pct < self._min_profit:
                continue

            opportunities_found += 1

            # Generate signal for Kalshi side
            signal = self._create_arbitrage_signal(arb, market)
            if signal:
                signals.append(signal)
                logger.info(
                    f"Arbitrage opportunity: {ticker} "
                    f"profit={arb.profit_pct:.2%} "
                    f"(buy {arb.buy_yes_on.upper()} YES, {arb.buy_no_on.upper()} NO)"
                )

        self.update_metrics(
            markets_checked=markets_checked,
            poly_markets_available=len(self._poly_markets),
            opportunities_found=opportunities_found,
            signals_generated=len(signals),
        )
        self.record_run()

        return signals

    async def _refresh_polymarket_cache(self) -> None:
        """Refresh Polymarket market cache."""
        now = datetime.now(timezone.utc)

        if self._cache_time is not None:
            age = (now - self._cache_time).total_seconds()
            if age < self._cache_ttl and self._poly_markets:
                return

        if self._polymarket is None:
            return

        try:
            self._poly_markets = await self._polymarket.get_markets(
                active_only=True,
                limit=200,
            )
            self._cache_time = now
            logger.debug(f"Refreshed Polymarket cache: {len(self._poly_markets)} markets")

        except Exception as e:
            logger.error(f"Failed to refresh Polymarket cache: {e}")

    def _get_kalshi_prices(
        self, market: dict[str, Any]
    ) -> tuple[float | None, float | None]:
        """Extract YES and NO prices from Kalshi market data."""
        # Try to get prices from bid/ask
        yes_bid = market.get("yes_bid")
        yes_ask = market.get("yes_ask")
        no_bid = market.get("no_bid")
        no_ask = market.get("no_ask")

        yes_price = None
        no_price = None

        if yes_bid is not None and yes_ask is not None:
            yes_price = (float(yes_bid) + float(yes_ask)) / 2 / 100
        elif market.get("last_price") is not None:
            yes_price = float(market["last_price"]) / 100

        if no_bid is not None and no_ask is not None:
            no_price = (float(no_bid) + float(no_ask)) / 2 / 100
        elif yes_price is not None:
            no_price = 1.0 - yes_price

        return yes_price, no_price

    def _create_arbitrage_signal(
        self,
        arb: ArbitrageOpportunity,
        market: dict[str, Any],
    ) -> Signal | None:
        """Create signal for the Kalshi side of an arbitrage."""
        # Determine which side to trade on Kalshi
        if arb.buy_yes_on == "kalshi":
            direction = "yes"
            target_prob = 1.0  # We expect YES to settle at $1
            market_prob = arb.kalshi_yes_price
        else:
            direction = "no"
            target_prob = 1.0  # We expect NO to settle at $1
            market_prob = arb.kalshi_no_price

        # Edge is the guaranteed profit percentage
        edge = arb.profit_pct

        metadata = ArbitrageSignalMetadata(
            kalshi_ticker=arb.kalshi_ticker,
            polymarket_id=arb.polymarket_id,
            kalshi_yes=arb.kalshi_yes_price,
            kalshi_no=arb.kalshi_no_price,
            poly_yes=arb.polymarket_yes_price,
            poly_no=arb.polymarket_no_price,
            total_cost=arb.total_cost,
            guaranteed_profit=arb.guaranteed_profit,
            profit_pct=arb.profit_pct,
            buy_yes_on=arb.buy_yes_on,
            buy_no_on=arb.buy_no_on,
        )

        return self.create_signal(
            market_ticker=arb.kalshi_ticker,
            direction=direction,
            target_probability=target_prob,
            market_probability=market_prob,
            confidence=self.ARBITRAGE_CONFIDENCE,
            max_position=100,  # Larger size for arbitrage
            metadata={
                "arbitrage": metadata.to_dict(),
                "market_price_cents": int(market_prob * 100),
                "strategy_type": "cross_market_arbitrage",
            },
            expires_in_hours=0.25,  # 15 min expiry - arbitrage is time-sensitive
        )

    def get_arbitrage_status(self) -> dict[str, Any]:
        """Get arbitrage strategy status."""
        return {
            "enabled": self._enabled,
            "polymarket_connected": self._polymarket is not None,
            "poly_markets_cached": len(self._poly_markets),
            "cache_age": (
                (datetime.now(timezone.utc) - self._cache_time).total_seconds()
                if self._cache_time
                else None
            ),
            "config": {
                "min_profit": self._min_profit,
                "fees": self._fees,
                "min_liquidity": self.MIN_LIQUIDITY,
            },
        }


class SingleMarketArbitrage:
    """
    Detects arbitrage within a single Kalshi market.

    When YES + NO prices are less than $1.00 (minus fees), there's
    an arbitrage opportunity by buying both sides.
    """

    DEFAULT_FEES = 0.01  # 1% for single platform

    @classmethod
    def find_opportunities(
        cls,
        markets: list[dict[str, Any]],
        min_profit: float = 0.005,
        fees: float = DEFAULT_FEES,
    ) -> list[dict[str, Any]]:
        """
        Find single-market arbitrage opportunities.

        Args:
            markets: List of Kalshi market data
            min_profit: Minimum profit to report
            fees: Estimated fees

        Returns:
            List of arbitrage opportunities
        """
        opportunities = []

        for market in markets:
            yes_ask = market.get("yes_ask")
            no_ask = market.get("no_ask")

            if yes_ask is None or no_ask is None:
                continue

            # Convert to dollars (prices are in cents)
            yes_price = float(yes_ask) / 100
            no_price = float(no_ask) / 100

            total_cost = yes_price + no_price
            profit = 1.0 - total_cost - fees

            if profit > min_profit:
                opportunities.append({
                    "ticker": market.get("ticker"),
                    "yes_ask": yes_price,
                    "no_ask": no_price,
                    "total_cost": total_cost,
                    "profit": profit,
                    "profit_pct": profit / total_cost if total_cost > 0 else 0,
                })

        return opportunities
