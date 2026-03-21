"""Position and exposure limits for risk management."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kalshi_bot.config.settings import Settings
from kalshi_bot.core.exceptions import PositionLimitError
from kalshi_bot.core.types import Position
from kalshi_bot.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LimitCheck:
    """Result of a limit check."""

    passed: bool
    limit_type: str
    current_value: float
    max_value: float
    message: str


class PositionLimits:
    """
    Manages position and exposure limits.

    Enforces:
    - Maximum position size per market
    - Maximum total exposure as percentage of portfolio
    - Maximum concentration in any single market
    - Maximum number of concurrent positions
    """

    def __init__(self, settings: Settings) -> None:
        """
        Initialize position limits.

        Args:
            settings: Application settings
        """
        self.settings = settings

        self._max_position_size = settings.trading.max_position_size
        self._max_position_pct = settings.trading.max_position_pct
        self._max_concentration = 0.25
        self._max_concurrent_positions = 20
        self._max_total_exposure_pct = 0.80

    def check_position_size(
        self,
        proposed_quantity: int,
        current_quantity: int = 0,
    ) -> LimitCheck:
        """
        Check if proposed position size is within limits.

        Args:
            proposed_quantity: Quantity to add
            current_quantity: Existing position quantity

        Returns:
            LimitCheck result
        """
        total = current_quantity + proposed_quantity

        if total > self._max_position_size:
            return LimitCheck(
                passed=False,
                limit_type="max_position_size",
                current_value=float(total),
                max_value=float(self._max_position_size),
                message=f"Position size {total} exceeds maximum {self._max_position_size}",
            )

        return LimitCheck(
            passed=True,
            limit_type="max_position_size",
            current_value=float(total),
            max_value=float(self._max_position_size),
            message="Position size within limits",
        )

    def check_position_value(
        self,
        proposed_value: float,
        portfolio_value: float,
        current_position_value: float = 0.0,
    ) -> LimitCheck:
        """
        Check if proposed position value is within percentage limits.

        Args:
            proposed_value: Value of position to add
            portfolio_value: Total portfolio value
            current_position_value: Value of existing position in same market

        Returns:
            LimitCheck result
        """
        total_value = current_position_value + proposed_value
        pct = total_value / portfolio_value if portfolio_value > 0 else 0

        if pct > self._max_position_pct:
            return LimitCheck(
                passed=False,
                limit_type="max_position_pct",
                current_value=pct,
                max_value=self._max_position_pct,
                message=f"Position {pct:.1%} of portfolio exceeds maximum {self._max_position_pct:.1%}",
            )

        return LimitCheck(
            passed=True,
            limit_type="max_position_pct",
            current_value=pct,
            max_value=self._max_position_pct,
            message="Position value within limits",
        )

    def check_concentration(
        self,
        market_ticker: str,
        positions: list[Position],
        proposed_value: float,
        portfolio_value: float,
    ) -> LimitCheck:
        """
        Check concentration in a single market.

        Args:
            market_ticker: Market being traded
            positions: Current positions
            proposed_value: Value to add
            portfolio_value: Total portfolio value

        Returns:
            LimitCheck result
        """
        current_value = sum(
            p.market_exposure for p in positions if p.market_ticker == market_ticker
        )
        total_value = current_value + proposed_value
        concentration = total_value / portfolio_value if portfolio_value > 0 else 0

        if concentration > self._max_concentration:
            return LimitCheck(
                passed=False,
                limit_type="max_concentration",
                current_value=concentration,
                max_value=self._max_concentration,
                message=f"Concentration {concentration:.1%} in {market_ticker} exceeds maximum {self._max_concentration:.1%}",
            )

        return LimitCheck(
            passed=True,
            limit_type="max_concentration",
            current_value=concentration,
            max_value=self._max_concentration,
            message="Concentration within limits",
        )

    def check_total_exposure(
        self,
        positions: list[Position],
        proposed_value: float,
        portfolio_value: float,
    ) -> LimitCheck:
        """
        Check total market exposure.

        Args:
            positions: Current positions
            proposed_value: Value to add
            portfolio_value: Total portfolio value

        Returns:
            LimitCheck result
        """
        current_exposure = sum(p.market_exposure for p in positions)
        total_exposure = current_exposure + proposed_value
        exposure_pct = total_exposure / portfolio_value if portfolio_value > 0 else 0

        if exposure_pct > self._max_total_exposure_pct:
            return LimitCheck(
                passed=False,
                limit_type="max_total_exposure",
                current_value=exposure_pct,
                max_value=self._max_total_exposure_pct,
                message=f"Total exposure {exposure_pct:.1%} exceeds maximum {self._max_total_exposure_pct:.1%}",
            )

        return LimitCheck(
            passed=True,
            limit_type="max_total_exposure",
            current_value=exposure_pct,
            max_value=self._max_total_exposure_pct,
            message="Total exposure within limits",
        )

    def check_position_count(
        self,
        positions: list[Position],
        is_new_position: bool = True,
    ) -> LimitCheck:
        """
        Check number of concurrent positions.

        Args:
            positions: Current positions
            is_new_position: Whether this would be a new position

        Returns:
            LimitCheck result
        """
        current_count = len([p for p in positions if p.quantity > 0])
        new_count = current_count + (1 if is_new_position else 0)

        if new_count > self._max_concurrent_positions:
            return LimitCheck(
                passed=False,
                limit_type="max_concurrent_positions",
                current_value=float(new_count),
                max_value=float(self._max_concurrent_positions),
                message=f"Position count {new_count} exceeds maximum {self._max_concurrent_positions}",
            )

        return LimitCheck(
            passed=True,
            limit_type="max_concurrent_positions",
            current_value=float(new_count),
            max_value=float(self._max_concurrent_positions),
            message="Position count within limits",
        )

    def check_all(
        self,
        market_ticker: str,
        proposed_quantity: int,
        proposed_value: float,
        positions: list[Position],
        portfolio_value: float,
    ) -> list[LimitCheck]:
        """
        Run all limit checks.

        Args:
            market_ticker: Market being traded
            proposed_quantity: Quantity to add
            proposed_value: Value to add
            positions: Current positions
            portfolio_value: Total portfolio value

        Returns:
            List of all LimitCheck results
        """
        current_position = next(
            (p for p in positions if p.market_ticker == market_ticker),
            None,
        )

        checks = [
            self.check_position_size(
                proposed_quantity,
                current_position.quantity if current_position else 0,
            ),
            self.check_position_value(
                proposed_value,
                portfolio_value,
                current_position.market_exposure if current_position else 0.0,
            ),
            self.check_concentration(
                market_ticker, positions, proposed_value, portfolio_value
            ),
            self.check_total_exposure(positions, proposed_value, portfolio_value),
            self.check_position_count(positions, current_position is None),
        ]

        return checks

    def validate_trade(
        self,
        market_ticker: str,
        proposed_quantity: int,
        proposed_value: float,
        positions: list[Position],
        portfolio_value: float,
    ) -> None:
        """
        Validate a trade against all limits.

        Args:
            market_ticker: Market being traded
            proposed_quantity: Quantity to add
            proposed_value: Value to add
            positions: Current positions
            portfolio_value: Total portfolio value

        Raises:
            PositionLimitError: If any limit is exceeded
        """
        checks = self.check_all(
            market_ticker, proposed_quantity, proposed_value, positions, portfolio_value
        )

        failures = [c for c in checks if not c.passed]

        if failures:
            first_failure = failures[0]
            raise PositionLimitError(
                first_failure.message,
                current_position=int(first_failure.current_value),
                max_position=int(first_failure.max_value),
            )

    def get_max_allowed_quantity(
        self,
        market_ticker: str,
        price_per_contract: float,
        positions: list[Position],
        portfolio_value: float,
    ) -> int:
        """
        Calculate maximum allowed quantity for a trade.

        Args:
            market_ticker: Market to trade
            price_per_contract: Price per contract
            positions: Current positions
            portfolio_value: Total portfolio value

        Returns:
            Maximum allowed quantity
        """
        current_position = next(
            (p for p in positions if p.market_ticker == market_ticker),
            None,
        )
        current_qty = current_position.quantity if current_position else 0
        current_value = current_position.market_exposure if current_position else 0.0

        max_by_size = self._max_position_size - current_qty

        max_by_pct = int(
            (portfolio_value * self._max_position_pct - current_value)
            / price_per_contract
        )

        max_by_concentration = int(
            (portfolio_value * self._max_concentration - current_value)
            / price_per_contract
        )

        current_exposure = sum(p.market_exposure for p in positions)
        max_by_exposure = int(
            (portfolio_value * self._max_total_exposure_pct - current_exposure)
            / price_per_contract
        )

        return max(0, min(max_by_size, max_by_pct, max_by_concentration, max_by_exposure))

    def get_status(self) -> dict[str, Any]:
        """Get current limit settings."""
        return {
            "max_position_size": self._max_position_size,
            "max_position_pct": self._max_position_pct,
            "max_concentration": self._max_concentration,
            "max_concurrent_positions": self._max_concurrent_positions,
            "max_total_exposure_pct": self._max_total_exposure_pct,
        }
