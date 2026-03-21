"""Profit locking mechanism to protect initial principal after 2x ROI."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from kalshi_bot.config.settings import Settings
from kalshi_bot.core.exceptions import ProfitLockError
from kalshi_bot.persistence.database import Database
from kalshi_bot.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ProfitLockState:
    """State of the profit lock mechanism."""

    initial_principal: float
    profit_lock_multiplier: float
    locked_principal: float
    is_locked: bool
    locked_at: datetime | None
    peak_value: float
    total_locked_amount: float


class ProfitLock:
    """
    Implements profit locking after achieving target ROI.

    When the portfolio reaches 2x (or configured multiplier) of the initial
    principal, the initial principal is "locked" and removed from the
    tradeable balance. This ensures that even if subsequent trades lose
    money, the original investment is protected.

    Example:
    - Initial principal: $1,000
    - Profit lock multiplier: 2.0
    - When portfolio value reaches $2,000:
      - Lock $1,000 (initial principal)
      - Tradeable balance: $1,000 (profits only)
    - If portfolio drops to $1,500:
      - Locked: $1,000 (untouchable)
      - Tradeable: $500
    - The locked $1,000 can never be used for trading
    """

    def __init__(
        self,
        settings: Settings,
        database: Database | None = None,
    ) -> None:
        """
        Initialize profit lock.

        Args:
            settings: Application settings
            database: Database for persistence
        """
        self.settings = settings
        self.database = database

        self._initial_principal = settings.portfolio.initial_principal
        self._profit_lock_multiplier = settings.portfolio.profit_lock_multiplier

        self._locked_principal: float = 0.0
        self._is_locked: bool = False
        self._locked_at: datetime | None = None
        self._peak_value: float = self._initial_principal

    async def initialize(self) -> None:
        """Load state from database if available."""
        if self.database is None:
            return

        try:
            locked_str = await self.database.get_state("profit_lock.locked_principal")
            if locked_str:
                self._locked_principal = float(locked_str)

            is_locked_str = await self.database.get_state("profit_lock.is_locked")
            if is_locked_str:
                self._is_locked = is_locked_str.lower() == "true"

            locked_at_str = await self.database.get_state("profit_lock.locked_at")
            if locked_at_str:
                self._locked_at = datetime.fromisoformat(locked_at_str)

            peak_str = await self.database.get_state("profit_lock.peak_value")
            if peak_str:
                self._peak_value = float(peak_str)

            initial_str = await self.database.get_state("profit_lock.initial_principal")
            if initial_str:
                self._initial_principal = float(initial_str)

            logger.info(
                f"Profit lock initialized: locked=${self._locked_principal:.2f}, "
                f"is_locked={self._is_locked}"
            )

        except Exception as e:
            logger.error(f"Failed to load profit lock state: {e}")

    async def _save_state(self) -> None:
        """Save state to database."""
        if self.database is None:
            return

        try:
            await self.database.set_state(
                "profit_lock.locked_principal", str(self._locked_principal)
            )
            await self.database.set_state(
                "profit_lock.is_locked", str(self._is_locked).lower()
            )
            if self._locked_at:
                await self.database.set_state(
                    "profit_lock.locked_at", self._locked_at.isoformat()
                )
            await self.database.set_state(
                "profit_lock.peak_value", str(self._peak_value)
            )
            await self.database.set_state(
                "profit_lock.initial_principal", str(self._initial_principal)
            )
        except Exception as e:
            logger.error(f"Failed to save profit lock state: {e}")

    async def update_portfolio_value(self, total_value: float) -> bool:
        """
        Update with current portfolio value and check for lock trigger.

        Args:
            total_value: Current total portfolio value

        Returns:
            True if profit lock was just triggered
        """
        if total_value > self._peak_value:
            self._peak_value = total_value

        if self._is_locked:
            return False

        target_value = self._initial_principal * self._profit_lock_multiplier

        if total_value >= target_value:
            self._locked_principal = self._initial_principal
            self._is_locked = True
            self._locked_at = datetime.utcnow()

            logger.info(
                f"PROFIT LOCK TRIGGERED: Locking ${self._locked_principal:.2f} "
                f"(portfolio value: ${total_value:.2f}, "
                f"target was: ${target_value:.2f})"
            )

            await self._save_state()
            return True

        return False

    def get_tradeable_balance(self, total_value: float) -> float:
        """
        Calculate the tradeable balance (excluding locked principal).

        Args:
            total_value: Current total portfolio value

        Returns:
            Amount available for trading
        """
        if not self._is_locked:
            return total_value

        tradeable = total_value - self._locked_principal
        return max(0.0, tradeable)

    def validate_trade(self, trade_cost: float, available_balance: float) -> None:
        """
        Validate that a trade doesn't use locked principal.

        Args:
            trade_cost: Cost of proposed trade
            available_balance: Current cash balance

        Raises:
            ProfitLockError: If trade would use locked funds
        """
        if not self._is_locked:
            return

        if trade_cost > available_balance:
            raise ProfitLockError(
                "Trade would exceed available balance",
                locked_amount=self._locked_principal,
                attempted_use=trade_cost - available_balance,
            )

    def set_initial_principal(self, amount: float) -> None:
        """
        Set or update the initial principal.

        Should only be called during initialization or if user deposits more.

        Args:
            amount: New initial principal amount
        """
        if self._is_locked:
            logger.warning(
                "Cannot change initial principal after profit lock is active"
            )
            return

        self._initial_principal = amount
        self._peak_value = max(self._peak_value, amount)
        logger.info(f"Initial principal set to ${amount:.2f}")

    def add_to_principal(self, amount: float) -> None:
        """
        Add to initial principal (e.g., new deposit).

        Args:
            amount: Amount to add
        """
        if self._is_locked:
            logger.info(
                f"Adding ${amount:.2f} to locked principal "
                f"(profit lock already active)"
            )
            self._locked_principal += amount
        else:
            self._initial_principal += amount
            logger.info(
                f"Added ${amount:.2f} to initial principal. "
                f"New total: ${self._initial_principal:.2f}"
            )

    async def manual_lock(self, amount: float | None = None) -> None:
        """
        Manually trigger profit lock.

        Args:
            amount: Amount to lock (defaults to initial principal)
        """
        lock_amount = amount or self._initial_principal

        self._locked_principal = lock_amount
        self._is_locked = True
        self._locked_at = datetime.utcnow()

        logger.info(f"Manual profit lock triggered: ${lock_amount:.2f}")
        await self._save_state()

    async def reset(self, new_principal: float | None = None) -> None:
        """
        Reset profit lock (use with caution).

        Args:
            new_principal: New initial principal (defaults to current)
        """
        self._is_locked = False
        self._locked_principal = 0.0
        self._locked_at = None

        if new_principal is not None:
            self._initial_principal = new_principal
            self._peak_value = new_principal

        logger.warning("Profit lock has been RESET")
        await self._save_state()

    def get_state(self) -> ProfitLockState:
        """Get current profit lock state."""
        return ProfitLockState(
            initial_principal=self._initial_principal,
            profit_lock_multiplier=self._profit_lock_multiplier,
            locked_principal=self._locked_principal,
            is_locked=self._is_locked,
            locked_at=self._locked_at,
            peak_value=self._peak_value,
            total_locked_amount=self._locked_principal,
        )

    def get_status(self) -> dict[str, Any]:
        """Get status as dictionary."""
        state = self.get_state()
        return {
            "initial_principal": state.initial_principal,
            "profit_lock_multiplier": state.profit_lock_multiplier,
            "locked_principal": state.locked_principal,
            "is_locked": state.is_locked,
            "locked_at": state.locked_at.isoformat() if state.locked_at else None,
            "peak_value": state.peak_value,
            "target_value": state.initial_principal * state.profit_lock_multiplier,
            "progress_to_lock": (
                self._peak_value / (state.initial_principal * state.profit_lock_multiplier)
                if state.initial_principal > 0
                else 0
            ),
        }

    @property
    def is_locked(self) -> bool:
        """Check if profit lock is active."""
        return self._is_locked

    @property
    def locked_amount(self) -> float:
        """Get locked principal amount."""
        return self._locked_principal

    @property
    def initial_principal(self) -> float:
        """Get initial principal."""
        return self._initial_principal
