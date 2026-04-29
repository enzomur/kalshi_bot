"""Base broker interface and common execution types.

This module defines the abstract interface that all brokers must implement,
ensuring consistent behavior between paper and live trading.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.core.types import Order, Fill, Position, Side


@dataclass
class ExecutionResult:
    """Result of executing a trade."""

    success: bool
    order: Order | None
    fills: list[Fill]
    total_cost: float  # In dollars
    total_fees: float  # In dollars
    filled_quantity: int
    average_price: float  # In cents
    error_message: str | None = None
    execution_time_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def net_cost(self) -> float:
        """Total cost including fees."""
        return self.total_cost + self.total_fees


class BaseBroker(ABC):
    """Abstract base class for trade execution brokers.

    All brokers (paper and live) must implement this interface to ensure
    consistent behavior and easy switching between modes.
    """

    @property
    @abstractmethod
    def is_paper(self) -> bool:
        """Return True if this is a paper trading broker."""
        pass

    @property
    @abstractmethod
    def balance(self) -> float:
        """Get current available balance in dollars."""
        pass

    @abstractmethod
    async def execute(
        self,
        market_ticker: str,
        side: Side,
        quantity: int,
        price: int,
        decision_id: str | None = None,
    ) -> ExecutionResult:
        """
        Execute a trade.

        Args:
            market_ticker: Market to trade
            side: YES or NO
            quantity: Number of contracts
            price: Price in cents (1-99)
            decision_id: Optional risk decision ID for tracking

        Returns:
            ExecutionResult with trade details
        """
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a pending order.

        Args:
            order_id: Order to cancel

        Returns:
            True if cancelled, False if not found or already filled
        """
        pass

    @abstractmethod
    async def cancel_all_orders(self, market_ticker: str | None = None) -> int:
        """
        Cancel all pending orders.

        Args:
            market_ticker: Optional filter by market

        Returns:
            Number of orders cancelled
        """
        pass

    @abstractmethod
    async def get_positions(self) -> list[Position]:
        """Get all current positions."""
        pass

    @abstractmethod
    async def get_position(
        self, market_ticker: str, side: Side
    ) -> Position | None:
        """Get a specific position."""
        pass

    @abstractmethod
    async def sync_positions(self) -> None:
        """Sync positions with external source (API for live, DB for paper)."""
        pass

    @abstractmethod
    def get_status(self) -> dict[str, Any]:
        """Get broker status including balance, positions, and metrics."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Clean up broker resources."""
        pass
