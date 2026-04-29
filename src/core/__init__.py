"""Core types, exceptions, and mode management."""

from src.core.types import (
    Signal,
    TradingMode,
    Side,
    OrderType,
    OrderStatus,
    MarketData,
    OrderBook,
    OrderBookLevel,
    Position,
    Order,
    Fill,
    Candlestick,
)
from src.core.mode import ModeManager, ModeConfig, get_mode_manager, verify_mode_on_startup
from src.core.exceptions import (
    KalshiBotError,
    ConfigurationError,
    ModeError,
    SignatureError,
    AuthenticationError,
    RateLimitError,
    CircuitBreakerError,
    ExecutionError,
    OrderError,
    InsufficientBalanceError,
    WebSocketError,
    APIError,
    RiskError,
    PositionLimitError,
)

__all__ = [
    # Types
    "Signal",
    "TradingMode",
    "Side",
    "OrderType",
    "OrderStatus",
    "MarketData",
    "OrderBook",
    "OrderBookLevel",
    "Position",
    "Order",
    "Fill",
    "Candlestick",
    # Mode
    "ModeManager",
    "ModeConfig",
    "get_mode_manager",
    "verify_mode_on_startup",
    # Exceptions
    "KalshiBotError",
    "ConfigurationError",
    "ModeError",
    "SignatureError",
    "AuthenticationError",
    "RateLimitError",
    "CircuitBreakerError",
    "ExecutionError",
    "OrderError",
    "InsufficientBalanceError",
    "WebSocketError",
    "APIError",
    "RiskError",
    "PositionLimitError",
]
