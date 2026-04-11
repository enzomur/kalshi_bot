"""Base class for all trading agents."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from kalshi_bot.utils.logging import get_logger

if TYPE_CHECKING:
    from kalshi_bot.persistence.database import Database

logger = get_logger(__name__)


@dataclass
class AgentStatus:
    """Status information for an agent."""

    name: str
    enabled: bool
    running: bool
    last_run: datetime | None = None
    last_error: str | None = None
    run_count: int = 0
    error_count: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "enabled": self.enabled,
            "running": self.running,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "last_error": self.last_error,
            "run_count": self.run_count,
            "error_count": self.error_count,
            "metrics": self.metrics,
        }


class BaseAgent(ABC):
    """
    Base class for all trading agents.

    Agents run periodic tasks to gather intelligence, validate trades,
    or improve models. Each agent has its own update interval and
    can be enabled/disabled independently.
    """

    def __init__(
        self,
        db: Database,
        name: str,
        update_interval_seconds: float = 300.0,
        enabled: bool = True,
    ) -> None:
        """
        Initialize the agent.

        Args:
            db: Database connection
            name: Agent name for logging
            update_interval_seconds: How often to run the agent loop
            enabled: Whether the agent is enabled
        """
        self._db = db
        self._name = name
        self._update_interval = update_interval_seconds
        self._enabled = enabled
        self._running = False
        self._status = AgentStatus(
            name=name,
            enabled=enabled,
            running=False,
        )

    @property
    def name(self) -> str:
        """Get agent name."""
        return self._name

    @property
    def enabled(self) -> bool:
        """Check if agent is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable the agent."""
        self._enabled = value
        self._status.enabled = value

    @property
    def running(self) -> bool:
        """Check if agent is currently running."""
        return self._running

    def get_status(self) -> dict[str, Any]:
        """Get agent status."""
        return self._status.to_dict()

    async def start(self, shutdown_event: asyncio.Event) -> None:
        """
        Start the agent's main loop.

        Args:
            shutdown_event: Event to signal shutdown
        """
        if not self._enabled:
            logger.info(f"Agent {self._name} is disabled, not starting")
            return

        self._running = True
        self._status.running = True
        logger.info(f"Starting agent: {self._name}")

        try:
            while not shutdown_event.is_set():
                try:
                    await self._run_cycle()
                    self._status.run_count += 1
                    self._status.last_run = datetime.utcnow()
                    self._status.last_error = None
                except Exception as e:
                    self._status.error_count += 1
                    self._status.last_error = str(e)
                    logger.error(f"Agent {self._name} error: {e}")

                try:
                    await asyncio.wait_for(
                        shutdown_event.wait(),
                        timeout=self._update_interval,
                    )
                    break
                except asyncio.TimeoutError:
                    pass

        finally:
            self._running = False
            self._status.running = False
            logger.info(f"Agent {self._name} stopped")

    @abstractmethod
    async def _run_cycle(self) -> None:
        """
        Execute one cycle of the agent's work.

        Subclasses must implement this method.
        """
        pass

    async def run_once(self) -> None:
        """Run a single cycle (useful for testing)."""
        await self._run_cycle()
        self._status.run_count += 1
        self._status.last_run = datetime.utcnow()
