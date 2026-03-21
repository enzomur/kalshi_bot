"""Portfolio manager for tracking positions and value."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from kalshi_bot.api.client import KalshiAPIClient
from kalshi_bot.config.settings import Settings
from kalshi_bot.core.types import Position
from kalshi_bot.persistence.models import PortfolioSnapshotRepository, PositionRepository
from kalshi_bot.portfolio.profit_lock import ProfitLock
from kalshi_bot.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PortfolioState:
    """Current state of the portfolio."""

    balance: float
    positions: list[Position]
    positions_value: float
    total_value: float
    unrealized_pnl: float
    realized_pnl: float
    locked_principal: float
    tradeable_balance: float
    peak_value: float
    drawdown: float
    profit_locked: bool
    last_sync: datetime | None


class PortfolioManager:
    """
    Manages portfolio state and synchronization with Kalshi.

    Responsibilities:
    - Sync positions with Kalshi API
    - Track portfolio value
    - Enforce profit lock
    - Calculate P&L
    - Persist snapshots
    """

    def __init__(
        self,
        settings: Settings,
        api_client: KalshiAPIClient,
        profit_lock: ProfitLock,
        position_repo: PositionRepository | None = None,
        snapshot_repo: PortfolioSnapshotRepository | None = None,
    ) -> None:
        """
        Initialize portfolio manager.

        Args:
            settings: Application settings
            api_client: Kalshi API client
            profit_lock: Profit lock instance
            position_repo: Position repository
            snapshot_repo: Snapshot repository
        """
        self.settings = settings
        self.api_client = api_client
        self.profit_lock = profit_lock
        self.position_repo = position_repo
        self.snapshot_repo = snapshot_repo

        self._balance: float = 0.0
        self._positions: list[Position] = []
        self._realized_pnl: float = 0.0
        self._peak_value: float = 0.0
        self._last_sync: datetime | None = None
        self._sync_interval = settings.portfolio.sync_interval
        self._sync_lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize portfolio state from Kalshi and database."""
        await self.profit_lock.initialize()

        await self.sync_with_kalshi()

        if self.snapshot_repo:
            try:
                peak = await self.snapshot_repo.get_peak_value()
                if peak > 0:
                    self._peak_value = peak
            except Exception as e:
                logger.error(f"Failed to load peak value: {e}")

        logger.info(
            f"Portfolio initialized: balance=${self._balance:.2f}, "
            f"positions={len(self._positions)}, "
            f"total_value=${self.total_value:.2f}"
        )

    async def sync_with_kalshi(self) -> None:
        """Synchronize portfolio state with Kalshi API."""
        async with self._sync_lock:
            try:
                balance_data = await self.api_client.get_balance()
                self._balance = balance_data.get("balance", 0) / 100

                self._positions = await self.api_client.get_positions()

                self._last_sync = datetime.utcnow()

                total = self.total_value
                if total > self._peak_value:
                    self._peak_value = total

                lock_triggered = await self.profit_lock.update_portfolio_value(total)
                if lock_triggered:
                    logger.info("Profit lock was triggered during sync")

                if self.position_repo:
                    await self.position_repo.clear_all()
                    for position in self._positions:
                        await self.position_repo.save(position)

                logger.debug(
                    f"Portfolio synced: balance=${self._balance:.2f}, "
                    f"positions={len(self._positions)}"
                )

            except Exception as e:
                logger.error(f"Failed to sync with Kalshi: {e}")
                raise

    async def take_snapshot(self) -> dict[str, Any]:
        """Take and persist a portfolio snapshot."""
        state = self.get_state()

        snapshot = {
            "balance": state.balance,
            "portfolio_value": state.total_value,
            "positions_value": state.positions_value,
            "total_value": state.total_value,
            "unrealized_pnl": state.unrealized_pnl,
            "realized_pnl": state.realized_pnl,
            "locked_principal": state.locked_principal,
            "tradeable_balance": state.tradeable_balance,
            "peak_value": state.peak_value,
            "drawdown": state.drawdown,
            "profit_locked": state.profit_locked,
        }

        if self.snapshot_repo:
            try:
                await self.snapshot_repo.save(snapshot)
            except Exception as e:
                logger.error(f"Failed to save snapshot: {e}")

        return snapshot

    async def needs_sync(self) -> bool:
        """Check if portfolio needs synchronization."""
        if self._last_sync is None:
            return True

        elapsed = (datetime.utcnow() - self._last_sync).total_seconds()
        return elapsed >= self._sync_interval

    async def ensure_synced(self) -> None:
        """Ensure portfolio is synced, refreshing if needed."""
        if await self.needs_sync():
            await self.sync_with_kalshi()

    def get_position(self, market_ticker: str) -> Position | None:
        """Get position for a specific market."""
        for position in self._positions:
            if position.market_ticker == market_ticker:
                return position
        return None

    def get_state(self) -> PortfolioState:
        """Get current portfolio state."""
        positions_value = sum(p.market_exposure for p in self._positions)
        total_value = self._balance + positions_value
        unrealized_pnl = sum(p.unrealized_pnl for p in self._positions)

        drawdown = 0.0
        if self._peak_value > 0:
            drawdown = (self._peak_value - total_value) / self._peak_value

        tradeable = self.profit_lock.get_tradeable_balance(total_value)

        return PortfolioState(
            balance=self._balance,
            positions=self._positions.copy(),
            positions_value=positions_value,
            total_value=total_value,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=self._realized_pnl,
            locked_principal=self.profit_lock.locked_amount,
            tradeable_balance=tradeable,
            peak_value=self._peak_value,
            drawdown=drawdown,
            profit_locked=self.profit_lock.is_locked,
            last_sync=self._last_sync,
        )

    @property
    def balance(self) -> float:
        """Get cash balance."""
        return self._balance

    @property
    def positions(self) -> list[Position]:
        """Get all positions."""
        return self._positions.copy()

    @property
    def total_value(self) -> float:
        """Get total portfolio value."""
        positions_value = sum(p.market_exposure for p in self._positions)
        return self._balance + positions_value

    @property
    def tradeable_balance(self) -> float:
        """Get tradeable balance (respecting profit lock)."""
        return self.profit_lock.get_tradeable_balance(self.total_value)

    @property
    def available_for_trading(self) -> float:
        """Get cash available for new trades."""
        if self.profit_lock.is_locked:
            tradeable = self.tradeable_balance
            positions_value = sum(p.market_exposure for p in self._positions)
            return max(0.0, tradeable - positions_value)
        return self._balance

    def record_realized_pnl(self, amount: float) -> None:
        """Record realized P&L from a trade."""
        self._realized_pnl += amount

    async def update_balance(self, new_balance: float) -> None:
        """Update balance after a trade (before sync)."""
        self._balance = new_balance

        total = self.total_value
        if total > self._peak_value:
            self._peak_value = total

        await self.profit_lock.update_portfolio_value(total)

    def get_status(self) -> dict[str, Any]:
        """Get portfolio status as dictionary."""
        state = self.get_state()
        return {
            "balance": state.balance,
            "positions_count": len(state.positions),
            "positions_value": state.positions_value,
            "total_value": state.total_value,
            "unrealized_pnl": state.unrealized_pnl,
            "realized_pnl": state.realized_pnl,
            "locked_principal": state.locked_principal,
            "tradeable_balance": state.tradeable_balance,
            "available_for_trading": self.available_for_trading,
            "peak_value": state.peak_value,
            "drawdown": state.drawdown,
            "drawdown_pct": f"{state.drawdown:.2%}",
            "profit_locked": state.profit_locked,
            "last_sync": state.last_sync.isoformat() if state.last_sync else None,
            "profit_lock": self.profit_lock.get_status(),
        }
