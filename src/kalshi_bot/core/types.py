"""Core types, enums, and dataclasses for the Kalshi bot."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class Side(str, Enum):
    """Order side."""

    YES = "yes"
    NO = "no"


class OrderType(str, Enum):
    """Order type."""

    LIMIT = "limit"
    MARKET = "market"


class OrderStatus(str, Enum):
    """Order status."""

    PENDING = "pending"
    OPEN = "open"
    RESTING = "resting"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    REJECTED = "rejected"
    EXECUTED = "executed"


class ArbitrageType(str, Enum):
    """Type of arbitrage opportunity."""

    SINGLE_MARKET = "single_market"  # YES + NO < 100
    MULTI_OUTCOME = "multi_outcome"  # Sum of outcomes < 100
    CROSS_MARKET = "cross_market"  # Correlated markets


@dataclass
class Position:
    """Represents a position in a market."""

    market_ticker: str
    side: Side
    quantity: int
    average_price: float  # In cents
    market_exposure: float  # Total exposure in dollars
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def total_cost(self) -> float:
        """Total cost of the position in dollars."""
        return (self.average_price * self.quantity) / 100

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "market_ticker": self.market_ticker,
            "side": self.side.value,
            "quantity": self.quantity,
            "average_price": self.average_price,
            "market_exposure": self.market_exposure,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class Order:
    """Represents an order."""

    order_id: str
    market_ticker: str
    side: Side
    order_type: OrderType
    price: int  # In cents (1-99)
    quantity: int
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    remaining_quantity: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self) -> None:
        if self.remaining_quantity == 0:
            self.remaining_quantity = self.quantity

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "order_id": self.order_id,
            "market_ticker": self.market_ticker,
            "side": self.side.value,
            "order_type": self.order_type.value,
            "price": self.price,
            "quantity": self.quantity,
            "status": self.status.value,
            "filled_quantity": self.filled_quantity,
            "remaining_quantity": self.remaining_quantity,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class Trade:
    """Represents an executed trade."""

    trade_id: str
    order_id: str
    market_ticker: str
    side: Side
    price: int  # In cents
    quantity: int
    fee: float  # In dollars
    executed_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def total_cost(self) -> float:
        """Total cost including fees in dollars."""
        return (self.price * self.quantity) / 100 + self.fee

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trade_id": self.trade_id,
            "order_id": self.order_id,
            "market_ticker": self.market_ticker,
            "side": self.side.value,
            "price": self.price,
            "quantity": self.quantity,
            "fee": self.fee,
            "executed_at": self.executed_at.isoformat(),
        }


@dataclass
class OrderBookLevel:
    """Single level in an order book."""

    price: int  # In cents
    quantity: int


@dataclass
class OrderBook:
    """Order book for a market."""

    market_ticker: str
    yes_bids: list[OrderBookLevel] = field(default_factory=list)
    yes_asks: list[OrderBookLevel] = field(default_factory=list)
    no_bids: list[OrderBookLevel] = field(default_factory=list)
    no_asks: list[OrderBookLevel] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def best_yes_bid(self) -> int | None:
        """Best YES bid price."""
        return self.yes_bids[0].price if self.yes_bids else None

    @property
    def best_yes_ask(self) -> int | None:
        """Best YES ask price."""
        return self.yes_asks[0].price if self.yes_asks else None

    @property
    def best_no_bid(self) -> int | None:
        """Best NO bid price."""
        return self.no_bids[0].price if self.no_bids else None

    @property
    def best_no_ask(self) -> int | None:
        """Best NO ask price."""
        return self.no_asks[0].price if self.no_asks else None

    @property
    def yes_bid_quantity(self) -> int:
        """Total quantity at best YES bid."""
        return self.yes_bids[0].quantity if self.yes_bids else 0

    @property
    def yes_ask_quantity(self) -> int:
        """Total quantity at best YES ask."""
        return self.yes_asks[0].quantity if self.yes_asks else 0

    @property
    def no_bid_quantity(self) -> int:
        """Total quantity at best NO bid."""
        return self.no_bids[0].quantity if self.no_bids else 0

    @property
    def no_ask_quantity(self) -> int:
        """Total quantity at best NO ask."""
        return self.no_asks[0].quantity if self.no_asks else 0


@dataclass
class MarketData:
    """Market information and current state."""

    ticker: str
    event_ticker: str
    title: str
    subtitle: str | None = None
    status: str = "open"
    yes_bid: int | None = None
    yes_ask: int | None = None
    no_bid: int | None = None
    no_ask: int | None = None
    last_price: int | None = None
    volume: int = 0
    open_interest: int = 0
    close_time: datetime | None = None
    expiration_time: datetime | None = None

    @property
    def mid_price(self) -> float | None:
        """Calculate mid price from best bid/ask."""
        if self.yes_bid is not None and self.yes_ask is not None:
            return (self.yes_bid + self.yes_ask) / 2
        return None

    @property
    def spread(self) -> int | None:
        """Calculate spread in cents."""
        if self.yes_bid is not None and self.yes_ask is not None:
            return self.yes_ask - self.yes_bid
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "ticker": self.ticker,
            "event_ticker": self.event_ticker,
            "title": self.title,
            "subtitle": self.subtitle,
            "status": self.status,
            "yes_bid": self.yes_bid,
            "yes_ask": self.yes_ask,
            "no_bid": self.no_bid,
            "no_ask": self.no_ask,
            "last_price": self.last_price,
            "volume": self.volume,
            "open_interest": self.open_interest,
            "close_time": self.close_time.isoformat() if self.close_time else None,
            "expiration_time": self.expiration_time.isoformat() if self.expiration_time else None,
        }


@dataclass
class Candlestick:
    """OHLC candlestick data from Kalshi's candlestick API."""

    ticker: str
    end_period_ts: int  # Unix timestamp for end of period
    open_price: int  # In cents (0-100)
    high_price: int  # In cents (0-100)
    low_price: int  # In cents (0-100)
    close_price: int  # In cents (0-100)
    volume: int
    open_interest: int
    yes_price: int | None = None  # YES price if available
    no_price: int | None = None  # NO price if available

    @property
    def period_datetime(self) -> datetime:
        """Convert end_period_ts to datetime."""
        return datetime.fromtimestamp(self.end_period_ts)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "ticker": self.ticker,
            "end_period_ts": self.end_period_ts,
            "open_price": self.open_price,
            "high_price": self.high_price,
            "low_price": self.low_price,
            "close_price": self.close_price,
            "volume": self.volume,
            "open_interest": self.open_interest,
            "yes_price": self.yes_price,
            "no_price": self.no_price,
        }


@dataclass
class ArbitrageOpportunity:
    """Represents a detected arbitrage opportunity."""

    opportunity_id: str
    arbitrage_type: ArbitrageType
    markets: list[str]  # Market tickers involved
    expected_profit: float  # In cents per contract
    expected_profit_pct: float  # As percentage
    confidence: float  # 0-1 confidence score
    legs: list[dict[str, Any]]  # Trade legs to execute
    max_quantity: int  # Maximum contracts available
    total_cost: float  # Total cost in dollars
    fees: float  # Expected fees in dollars
    net_profit: float  # Net profit after fees in dollars
    detected_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """Check if opportunity is still valid."""
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        return self.net_profit > 0 and self.max_quantity > 0

    @property
    def roi(self) -> float:
        """Return on investment."""
        if self.total_cost <= 0:
            return 0.0
        return self.net_profit / self.total_cost

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "opportunity_id": self.opportunity_id,
            "arbitrage_type": self.arbitrage_type.value,
            "markets": self.markets,
            "expected_profit": self.expected_profit,
            "expected_profit_pct": self.expected_profit_pct,
            "confidence": self.confidence,
            "legs": self.legs,
            "max_quantity": self.max_quantity,
            "total_cost": self.total_cost,
            "fees": self.fees,
            "net_profit": self.net_profit,
            "detected_at": self.detected_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "roi": self.roi,
            "metadata": self.metadata,
        }
