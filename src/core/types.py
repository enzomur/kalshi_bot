"""Core types, enums, and dataclasses for the Kalshi trading bot.

This module defines the fundamental types used throughout the system.
The Signal dataclass is the primary interface between strategies and the risk engine.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal


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


class TradingMode(str, Enum):
    """Trading mode - controls execution behavior.

    PAPER: All trades are simulated. No real money at risk.
    LIVE_PROBATION: Real trades with reduced limits ($500 max position).
    LIVE_FULL: Full live trading with normal limits.
    """

    PAPER = "paper"
    LIVE_PROBATION = "live_probation"
    LIVE_FULL = "live_full"


@dataclass(frozen=True)
class Signal:
    """A trading signal emitted by a strategy.

    Signals are immutable and represent a strategy's recommendation.
    The Risk Engine evaluates signals and decides whether to execute.

    Attributes:
        signal_id: Unique identifier for this signal.
        strategy_name: Name of the strategy that generated this signal.
        market_ticker: Kalshi market ticker to trade.
        direction: Whether to buy YES or NO contracts.
        target_probability: Strategy's estimated probability (0-1).
        confidence: Strategy's confidence in this estimate (0-1).
        edge: Difference between target_prob and market_prob.
        max_position: Maximum contracts the strategy recommends.
        metadata: Additional strategy-specific data.
        created_at: When the signal was created.
        expires_at: When the signal should be considered stale.
    """

    signal_id: str
    strategy_name: str
    market_ticker: str
    direction: Literal["yes", "no"]
    target_probability: float
    confidence: float
    edge: float
    max_position: int
    metadata: dict[str, Any]
    created_at: datetime
    expires_at: datetime | None

    def __post_init__(self) -> None:
        """Validate signal parameters."""
        if not 0 <= self.target_probability <= 1:
            raise ValueError(
                f"target_probability must be in [0, 1], got {self.target_probability}"
            )
        if not 0 <= self.confidence <= 1:
            raise ValueError(
                f"confidence must be in [0, 1], got {self.confidence}"
            )
        if self.max_position < 0:
            raise ValueError(
                f"max_position must be non-negative, got {self.max_position}"
            )

    @classmethod
    def create(
        cls,
        strategy_name: str,
        market_ticker: str,
        direction: Literal["yes", "no"],
        target_probability: float,
        market_probability: float,
        confidence: float,
        max_position: int,
        metadata: dict[str, Any] | None = None,
        expires_at: datetime | None = None,
    ) -> Signal:
        """Factory method to create a Signal with computed fields.

        Args:
            strategy_name: Name of the strategy.
            market_ticker: Kalshi market ticker.
            direction: "yes" or "no".
            target_probability: Strategy's probability estimate.
            market_probability: Current market probability (from prices).
            confidence: Strategy's confidence (0-1).
            max_position: Maximum contracts to hold.
            metadata: Additional data.
            expires_at: Optional expiration time.

        Returns:
            A new Signal instance.
        """
        edge = target_probability - market_probability
        if direction == "no":
            edge = -edge  # For NO bets, edge is inverted

        return cls(
            signal_id=str(uuid.uuid4()),
            strategy_name=strategy_name,
            market_ticker=market_ticker,
            direction=direction,
            target_probability=target_probability,
            confidence=confidence,
            edge=abs(edge),  # Store as positive
            max_position=max_position,
            metadata=metadata or {},
            created_at=datetime.now(timezone.utc),
            expires_at=expires_at,
        )

    @property
    def is_expired(self) -> bool:
        """Check if the signal has expired."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "signal_id": self.signal_id,
            "strategy_name": self.strategy_name,
            "market_ticker": self.market_ticker,
            "direction": self.direction,
            "target_probability": self.target_probability,
            "confidence": self.confidence,
            "edge": self.edge,
            "max_position": self.max_position,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }


@dataclass
class MarketData:
    """Market information and current state."""

    ticker: str
    event_ticker: str
    title: str
    subtitle: str | None = None
    status: str = "open"
    yes_bid: int | None = None  # In cents
    yes_ask: int | None = None  # In cents
    no_bid: int | None = None  # In cents
    no_ask: int | None = None  # In cents
    last_price: int | None = None  # In cents
    volume: int = 0
    open_interest: int = 0
    close_time: datetime | None = None
    expiration_time: datetime | None = None
    result: str | None = None  # Settlement result: "yes" or "no"

    @property
    def mid_price(self) -> float | None:
        """Calculate mid price from best bid/ask."""
        if self.yes_bid is not None and self.yes_ask is not None:
            return (self.yes_bid + self.yes_ask) / 2
        return None

    @property
    def yes_probability(self) -> float | None:
        """Convert mid price to probability."""
        mid = self.mid_price
        return mid / 100 if mid is not None else None

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
            "expiration_time": (
                self.expiration_time.isoformat() if self.expiration_time else None
            ),
            "result": self.result,
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
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

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

    def total_liquidity_at_price(
        self, side: Literal["yes", "no"], price: int
    ) -> int:
        """Get total quantity available at a given price."""
        if side == "yes":
            levels = self.yes_asks
        else:
            levels = self.no_asks

        total = 0
        for level in levels:
            if level.price <= price:
                total += level.quantity
        return total


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
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

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
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

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
class Fill:
    """Represents an execution fill."""

    fill_id: str
    order_id: str
    market_ticker: str
    side: Side
    price: int  # In cents
    quantity: int
    fee: float  # In dollars
    executed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def total_cost(self) -> float:
        """Total cost including fees in dollars."""
        return (self.price * self.quantity) / 100 + self.fee

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "fill_id": self.fill_id,
            "order_id": self.order_id,
            "market_ticker": self.market_ticker,
            "side": self.side.value,
            "price": self.price,
            "quantity": self.quantity,
            "fee": self.fee,
            "executed_at": self.executed_at.isoformat(),
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
    yes_price: int | None = None
    no_price: int | None = None

    @property
    def period_datetime(self) -> datetime:
        """Convert end_period_ts to datetime."""
        return datetime.fromtimestamp(self.end_period_ts, tz=timezone.utc)

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
