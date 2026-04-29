"""Custom exceptions for the Kalshi trading bot."""

from __future__ import annotations

from typing import Any


class KalshiBotError(Exception):
    """Base exception for all Kalshi bot errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} - Details: {self.details}"
        return self.message


class ConfigurationError(KalshiBotError):
    """Configuration related errors."""

    pass


class ModeError(KalshiBotError):
    """Trading mode related errors."""

    def __init__(
        self,
        message: str,
        current_mode: str | None = None,
        required_mode: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details)
        self.current_mode = current_mode
        self.required_mode = required_mode


class SignatureError(KalshiBotError):
    """Cryptographic signature verification errors."""

    pass


class AuthenticationError(KalshiBotError):
    """Authentication or authorization errors."""

    pass


class RateLimitError(KalshiBotError):
    """Rate limit exceeded error."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details)
        self.retry_after = retry_after


class CircuitBreakerError(KalshiBotError):
    """Circuit breaker triggered error."""

    def __init__(
        self,
        breaker_type: str,
        message: str | None = None,
        cooldown_remaining: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        msg = message or f"Circuit breaker triggered: {breaker_type}"
        super().__init__(msg, details)
        self.breaker_type = breaker_type
        self.cooldown_remaining = cooldown_remaining


class ExecutionError(KalshiBotError):
    """Trade execution error."""

    def __init__(
        self,
        message: str,
        order_id: str | None = None,
        market_ticker: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details)
        self.order_id = order_id
        self.market_ticker = market_ticker


class OrderError(KalshiBotError):
    """Order-specific error."""

    def __init__(
        self,
        message: str,
        order_id: str | None = None,
        status: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details)
        self.order_id = order_id
        self.status = status


class InsufficientBalanceError(KalshiBotError):
    """Insufficient balance for operation."""

    def __init__(
        self,
        required: float,
        available: float,
        message: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        msg = (
            message
            or f"Insufficient balance: required ${required:.2f}, available ${available:.2f}"
        )
        super().__init__(msg, details)
        self.required = required
        self.available = available


class WebSocketError(KalshiBotError):
    """WebSocket connection error."""

    def __init__(
        self,
        message: str,
        reconnect_attempt: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details)
        self.reconnect_attempt = reconnect_attempt


class APIError(KalshiBotError):
    """API request error."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        response_body: dict[str, Any] | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details)
        self.status_code = status_code
        self.response_body = response_body


class RiskError(KalshiBotError):
    """Risk management error."""

    def __init__(
        self,
        message: str,
        signal_id: str | None = None,
        reason: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details)
        self.signal_id = signal_id
        self.reason = reason


class PositionLimitError(KalshiBotError):
    """Position limit exceeded."""

    def __init__(
        self,
        message: str,
        current_position: int | None = None,
        max_position: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details)
        self.current_position = current_position
        self.max_position = max_position
