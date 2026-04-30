"""Market Making Strategy.

Captures spread by posting both bid and ask orders on liquid markets.
Uses inventory management to skew quotes when position builds up.

This strategy is only active on high-liquidity markets where:
- Spread is wide enough to profit after fees
- Volume is sufficient for order flow
- We can manage inventory risk
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.core.types import Signal, Side
from src.strategies.base import Strategy
from src.observability.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MarketMakingQuote:
    """A two-sided quote for market making."""

    ticker: str
    bid_price: int  # Cents
    ask_price: int  # Cents
    bid_size: int
    ask_size: int
    mid_price: float
    spread: int  # Cents
    spread_pct: float

    # Inventory adjustment
    inventory_skew: float = 0.0  # Positive = long, negative = short

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "bid_price": self.bid_price,
            "ask_price": self.ask_price,
            "bid_size": self.bid_size,
            "ask_size": self.ask_size,
            "mid_price": self.mid_price,
            "spread": self.spread,
            "spread_pct": self.spread_pct,
            "inventory_skew": self.inventory_skew,
        }


@dataclass
class InventoryState:
    """Tracks inventory position for a market."""

    ticker: str
    position: int = 0  # Positive = long YES, negative = short (long NO)
    avg_price: float = 0.0
    unrealized_pnl: float = 0.0
    max_position: int = 50

    @property
    def is_max_long(self) -> bool:
        return self.position >= self.max_position

    @property
    def is_max_short(self) -> bool:
        return self.position <= -self.max_position

    @property
    def inventory_ratio(self) -> float:
        """Position as ratio of max position (-1 to 1)."""
        if self.max_position == 0:
            return 0.0
        return self.position / self.max_position


class MarketMakingStrategy(Strategy):
    """
    Market making strategy for spread capture.

    Posts two-sided quotes on liquid markets, capturing the bid-ask spread.
    Uses inventory management to skew quotes when position builds up.

    Key parameters:
    - min_spread: Minimum spread to quote (must cover fees + profit)
    - max_inventory: Maximum position before stopping one side
    - skew_factor: How much to adjust quotes based on inventory
    """

    # Minimum requirements
    MIN_SPREAD_CENTS = 3  # 3 cent minimum spread
    MIN_VOLUME = 1000  # Minimum 24h volume
    MIN_OPEN_INTEREST = 500  # Minimum open interest

    # Default configuration
    DEFAULT_MAX_INVENTORY = 50  # Max position per market
    DEFAULT_SKEW_FACTOR = 0.5  # Price adjustment per inventory unit
    DEFAULT_QUOTE_SIZE = 10  # Default order size

    # Fee assumption
    FEES_PER_SIDE = 0.01  # 1 cent per contract per side

    def __init__(
        self,
        db=None,
        enabled: bool = True,
        max_inventory: int = DEFAULT_MAX_INVENTORY,
        skew_factor: float = DEFAULT_SKEW_FACTOR,
        quote_size: int = DEFAULT_QUOTE_SIZE,
        min_spread_cents: int = MIN_SPREAD_CENTS,
        target_markets: list[str] | None = None,
    ) -> None:
        """
        Initialize market making strategy.

        Args:
            db: Optional database connection
            enabled: Whether strategy is active
            max_inventory: Maximum position per market
            skew_factor: Quote skew per inventory unit
            quote_size: Default quote size
            min_spread_cents: Minimum spread to trade
            target_markets: Specific markets to make (None = auto-select)
        """
        super().__init__(
            db=db,
            enabled=enabled,
            min_edge=0.02,  # 2% minimum edge
            min_confidence=0.50,  # Lower confidence - we're capturing spread
        )
        self._max_inventory = max_inventory
        self._skew_factor = skew_factor
        self._quote_size = quote_size
        self._min_spread = min_spread_cents
        self._target_markets = target_markets or []

        # Track inventory per market
        self._inventory: dict[str, InventoryState] = {}

    @property
    def name(self) -> str:
        return "market_make"

    async def generate_signals(
        self,
        markets: list[dict[str, Any]],
    ) -> list[Signal]:
        """
        Generate market making signals for liquid markets.

        Args:
            markets: Market data from Kalshi

        Returns:
            List of signals (bid and ask as separate signals)
        """
        if not self._enabled:
            return []

        signals = []
        markets_analyzed = 0
        quotes_generated = 0

        for market in markets:
            ticker = market.get("ticker", "")

            # Filter to target markets if specified
            if self._target_markets and ticker not in self._target_markets:
                continue

            # Check liquidity requirements
            if not self._is_liquid_market(market):
                continue

            markets_analyzed += 1

            # Calculate quote
            quote = self._calculate_quote(market)
            if quote is None:
                continue

            # Check spread is wide enough
            min_profitable_spread = (2 * self.FEES_PER_SIDE * 100) + 1  # In cents
            if quote.spread < max(self._min_spread, min_profitable_spread):
                continue

            quotes_generated += 1

            # Generate signals for each side
            bid_signal, ask_signal = self._create_quote_signals(quote, market)

            if bid_signal:
                signals.append(bid_signal)
            if ask_signal:
                signals.append(ask_signal)

        self.update_metrics(
            markets_analyzed=markets_analyzed,
            quotes_generated=quotes_generated,
            signals_generated=len(signals),
            active_inventories=len(self._inventory),
        )
        self.record_run()

        return signals

    def _is_liquid_market(self, market: dict[str, Any]) -> bool:
        """Check if market meets liquidity requirements."""
        volume = market.get("volume", 0)
        open_interest = market.get("open_interest", 0)

        if volume < self.MIN_VOLUME:
            return False
        if open_interest < self.MIN_OPEN_INTEREST:
            return False

        # Must have both bid and ask
        if market.get("yes_bid") is None or market.get("yes_ask") is None:
            return False

        return True

    def _calculate_quote(self, market: dict[str, Any]) -> MarketMakingQuote | None:
        """Calculate optimal bid/ask quote for a market."""
        ticker = market.get("ticker", "")

        yes_bid = market.get("yes_bid")
        yes_ask = market.get("yes_ask")

        if yes_bid is None or yes_ask is None:
            return None

        bid = int(yes_bid)
        ask = int(yes_ask)
        spread = ask - bid
        mid = (bid + ask) / 2

        if spread <= 0:
            return None

        # Get or create inventory state
        inv = self._inventory.get(ticker)
        if inv is None:
            inv = InventoryState(ticker=ticker, max_position=self._max_inventory)
            self._inventory[ticker] = inv

        # Calculate inventory skew
        # When long, lower bid and ask to encourage selling
        # When short, raise bid and ask to encourage buying
        skew = inv.inventory_ratio * self._skew_factor

        # Adjust prices based on inventory
        adjusted_bid = bid - int(skew * spread / 2)
        adjusted_ask = ask - int(skew * spread / 2)

        # Ensure valid prices (1-99 cents)
        adjusted_bid = max(1, min(99, adjusted_bid))
        adjusted_ask = max(1, min(99, adjusted_ask))

        # Ensure bid < ask
        if adjusted_bid >= adjusted_ask:
            adjusted_bid = adjusted_ask - 1

        # Determine sizes based on inventory
        bid_size = self._quote_size if not inv.is_max_long else 0
        ask_size = self._quote_size if not inv.is_max_short else 0

        return MarketMakingQuote(
            ticker=ticker,
            bid_price=adjusted_bid,
            ask_price=adjusted_ask,
            bid_size=bid_size,
            ask_size=ask_size,
            mid_price=mid,
            spread=adjusted_ask - adjusted_bid,
            spread_pct=(adjusted_ask - adjusted_bid) / mid if mid > 0 else 0,
            inventory_skew=skew,
        )

    def _create_quote_signals(
        self,
        quote: MarketMakingQuote,
        market: dict[str, Any],
    ) -> tuple[Signal | None, Signal | None]:
        """Create bid and ask signals from a quote."""
        bid_signal = None
        ask_signal = None

        # Expected profit from spread capture
        spread_profit = quote.spread / 100 - (2 * self.FEES_PER_SIDE)
        edge = spread_profit / 2  # Per side

        if edge < self._min_edge:
            return None, None

        # Confidence based on spread width and liquidity
        volume = market.get("volume", 0)
        confidence = min(0.70, 0.50 + (volume / 10000) * 0.2)

        # Bid signal (buying YES at bid price)
        if quote.bid_size > 0:
            bid_signal = self.create_signal(
                market_ticker=quote.ticker,
                direction="yes",
                target_probability=(quote.mid_price + quote.spread / 4) / 100,
                market_probability=quote.bid_price / 100,
                confidence=confidence,
                max_position=quote.bid_size,
                metadata={
                    "market_price_cents": quote.bid_price,
                    "strategy_type": "market_making",
                    "side": "bid",
                    "quote": quote.to_dict(),
                },
                expires_in_hours=0.1,  # 6 min - MM quotes are short-lived
            )

        # Ask signal (selling YES / buying NO at ask price)
        if quote.ask_size > 0:
            ask_signal = self.create_signal(
                market_ticker=quote.ticker,
                direction="no",
                target_probability=(100 - quote.mid_price + quote.spread / 4) / 100,
                market_probability=(100 - quote.ask_price) / 100,
                confidence=confidence,
                max_position=quote.ask_size,
                metadata={
                    "market_price_cents": 100 - quote.ask_price,
                    "strategy_type": "market_making",
                    "side": "ask",
                    "quote": quote.to_dict(),
                },
                expires_in_hours=0.1,
            )

        return bid_signal, ask_signal

    def update_inventory(
        self,
        ticker: str,
        side: Side,
        quantity: int,
        price: float,
    ) -> None:
        """
        Update inventory after a fill.

        Args:
            ticker: Market ticker
            side: Side that was filled (YES or NO)
            quantity: Number of contracts
            price: Fill price
        """
        inv = self._inventory.get(ticker)
        if inv is None:
            inv = InventoryState(ticker=ticker, max_position=self._max_inventory)
            self._inventory[ticker] = inv

        if side == Side.YES:
            # Bought YES = long position
            new_position = inv.position + quantity
        else:
            # Bought NO = short position (from YES perspective)
            new_position = inv.position - quantity

        # Update average price (simplified)
        if inv.position == 0:
            inv.avg_price = price
        else:
            total_value = inv.avg_price * abs(inv.position) + price * quantity
            inv.avg_price = total_value / (abs(inv.position) + quantity)

        inv.position = new_position

        logger.debug(
            f"MM inventory update: {ticker} position={new_position} avg_price={inv.avg_price:.2f}"
        )

    def reset_inventory(self, ticker: str | None = None) -> None:
        """Reset inventory for a market or all markets."""
        if ticker:
            if ticker in self._inventory:
                del self._inventory[ticker]
        else:
            self._inventory.clear()

    def get_inventory_status(self) -> dict[str, Any]:
        """Get current inventory status."""
        return {
            "markets": {
                ticker: {
                    "position": inv.position,
                    "avg_price": inv.avg_price,
                    "inventory_ratio": inv.inventory_ratio,
                    "is_max_long": inv.is_max_long,
                    "is_max_short": inv.is_max_short,
                }
                for ticker, inv in self._inventory.items()
            },
            "config": {
                "max_inventory": self._max_inventory,
                "skew_factor": self._skew_factor,
                "quote_size": self._quote_size,
                "min_spread_cents": self._min_spread,
            },
        }
