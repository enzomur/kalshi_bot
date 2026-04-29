"""Base Strategy class for all trading strategies.

IMPORTANT: Strategies MUST NOT import from execution/.
They emit Signals only - the Risk Engine is the sole gatekeeper to execution.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from src.core.types import Signal

if TYPE_CHECKING:
    from src.ledger.database import Database


@dataclass
class StrategyStatus:
    """Status information for a strategy."""

    name: str
    enabled: bool
    signals_generated: int = 0
    last_run: datetime | None = None
    last_error: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "enabled": self.enabled,
            "signals_generated": self.signals_generated,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "last_error": self.last_error,
            "metrics": self.metrics,
        }


class Strategy(ABC):
    """
    Abstract base class for trading strategies.

    Strategies analyze market data and emit Signal objects when they
    identify trading opportunities. They NEVER execute trades directly.

    The Risk Engine is the sole gatekeeper - it evaluates signals,
    applies position limits, calculates Kelly sizing, and decides
    whether to execute.

    Subclasses must implement:
        - generate_signals(): Analyze data and return Signal objects
        - name property: Unique identifier for the strategy
    """

    def __init__(
        self,
        db: Database | None = None,
        enabled: bool = True,
        min_edge: float = 0.05,
        min_confidence: float = 0.50,
    ) -> None:
        """
        Initialize the strategy.

        Args:
            db: Optional database connection for historical data.
            enabled: Whether the strategy is active.
            min_edge: Minimum edge required to generate a signal.
            min_confidence: Minimum confidence required to generate a signal.
        """
        self._db = db
        self._enabled = enabled
        self._min_edge = min_edge
        self._min_confidence = min_confidence
        self._status = StrategyStatus(name=self.name, enabled=enabled)

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name identifying this strategy."""
        pass

    @property
    def enabled(self) -> bool:
        """Check if strategy is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable the strategy."""
        self._enabled = value
        self._status.enabled = value

    @abstractmethod
    async def generate_signals(
        self,
        markets: list[dict[str, Any]],
    ) -> list[Signal]:
        """
        Analyze markets and generate trading signals.

        This is the core method that subclasses implement. It should:
        1. Filter markets relevant to this strategy
        2. Analyze each market for opportunities
        3. Return Signal objects for identified opportunities

        Args:
            markets: List of market data dictionaries from the API.

        Returns:
            List of Signal objects. May be empty if no opportunities found.
        """
        pass

    def create_signal(
        self,
        market_ticker: str,
        direction: str,
        target_probability: float,
        market_probability: float,
        confidence: float,
        max_position: int,
        metadata: dict[str, Any] | None = None,
        expires_in_hours: float | None = 1.0,
    ) -> Signal | None:
        """
        Helper to create a Signal with validation.

        Returns None if the signal doesn't meet minimum thresholds.

        Args:
            market_ticker: Kalshi market ticker.
            direction: "yes" or "no".
            target_probability: Strategy's probability estimate (0-1).
            market_probability: Current market probability (0-1).
            confidence: Confidence in the estimate (0-1).
            max_position: Maximum contracts to recommend.
            metadata: Additional strategy-specific data.
            expires_in_hours: Hours until signal expires.

        Returns:
            Signal object or None if thresholds not met.
        """
        # Calculate edge
        if direction == "yes":
            edge = target_probability - market_probability
        else:
            edge = market_probability - target_probability

        # Check thresholds
        if edge < self._min_edge:
            return None
        if confidence < self._min_confidence:
            return None

        # Calculate expiration
        from datetime import timedelta

        expires_at = None
        if expires_in_hours is not None:
            expires_at = datetime.now(timezone.utc) + timedelta(hours=expires_in_hours)

        # Add market price to metadata
        full_metadata = metadata or {}
        full_metadata["market_price_cents"] = int(market_probability * 100)

        signal = Signal.create(
            strategy_name=self.name,
            market_ticker=market_ticker,
            direction=direction,
            target_probability=target_probability,
            market_probability=market_probability,
            confidence=confidence,
            max_position=max_position,
            metadata=full_metadata,
            expires_at=expires_at,
        )

        self._status.signals_generated += 1
        return signal

    def get_status(self) -> dict[str, Any]:
        """Get strategy status."""
        return self._status.to_dict()

    def update_metrics(self, **kwargs: Any) -> None:
        """Update status metrics."""
        self._status.metrics.update(kwargs)

    def record_run(self, error: str | None = None) -> None:
        """Record that the strategy ran."""
        self._status.last_run = datetime.now(timezone.utc)
        self._status.last_error = error
