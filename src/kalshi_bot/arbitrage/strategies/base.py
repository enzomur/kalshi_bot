"""Base class for arbitrage strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from kalshi_bot.config.settings import Settings
from kalshi_bot.core.types import ArbitrageOpportunity, MarketData, OrderBook
from kalshi_bot.utils.logging import get_logger

logger = get_logger(__name__)


class ArbitrageStrategy(ABC):
    """
    Abstract base class for arbitrage detection strategies.

    All arbitrage strategies inherit from this class and implement
    the detect() method to find specific types of opportunities.
    """

    def __init__(self, settings: Settings) -> None:
        """
        Initialize strategy.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self._min_profit_cents = settings.trading.min_profit_cents
        self._min_liquidity = settings.arbitrage.min_liquidity
        self._max_spread = settings.arbitrage.max_spread
        self._min_net_profit_pct = getattr(
            settings.trading, 'min_net_profit_pct', 0.02
        )  # Default 2% minimum

    @abstractmethod
    async def detect(
        self,
        markets: list[MarketData],
        orderbooks: dict[str, OrderBook],
    ) -> list[ArbitrageOpportunity]:
        """
        Detect arbitrage opportunities.

        Args:
            markets: List of market data
            orderbooks: Dictionary of order books keyed by ticker

        Returns:
            List of detected opportunities
        """
        pass

    @staticmethod
    def calculate_fee(price: int, quantity: int = 1) -> float:
        """
        Calculate Kalshi trading fee.

        Fee formula: 0.07 * P * (1-P) per contract
        Where P is price/100 (price as decimal)

        Args:
            price: Price in cents (1-99)
            quantity: Number of contracts

        Returns:
            Total fee in dollars
        """
        p = price / 100.0
        fee_per_contract = 0.07 * p * (1 - p)
        return fee_per_contract * quantity

    @staticmethod
    def calculate_total_fees(legs: list[dict[str, Any]]) -> float:
        """
        Calculate total fees for all legs of an arbitrage.

        Args:
            legs: List of trade legs with price and quantity

        Returns:
            Total fees in dollars
        """
        total_fee = 0.0
        for leg in legs:
            price = leg.get("price", 0)
            quantity = leg.get("quantity", 1)
            total_fee += ArbitrageStrategy.calculate_fee(price, quantity)
        return total_fee

    def validate_opportunity(self, opportunity: ArbitrageOpportunity) -> bool:
        """
        Validate that an opportunity meets minimum requirements.

        Args:
            opportunity: Opportunity to validate

        Returns:
            True if opportunity is valid
        """
        if opportunity.expected_profit < self._min_profit_cents:
            return False

        if opportunity.max_quantity < self._min_liquidity:
            return False

        if opportunity.net_profit <= 0:
            return False

        # Check minimum net profit percentage (2% default)
        if opportunity.total_cost > 0:
            net_profit_pct = opportunity.net_profit / opportunity.total_cost
            if net_profit_pct < self._min_net_profit_pct:
                logger.debug(
                    f"Opportunity rejected: net profit {net_profit_pct:.2%} "
                    f"< minimum {self._min_net_profit_pct:.2%}"
                )
                return False

        return True

    def _check_market_status(self, market: MarketData) -> bool:
        """Check if market is tradeable."""
        return market.status == "open"

    def _check_spread(self, orderbook: OrderBook) -> bool:
        """Check if spread is within acceptable range."""
        if orderbook.best_yes_bid is None or orderbook.best_yes_ask is None:
            return False

        spread = orderbook.best_yes_ask - orderbook.best_yes_bid
        return spread <= self._max_spread

    def _get_available_quantity(
        self,
        orderbook: OrderBook,
        side: str,
        is_yes: bool = True,
    ) -> int:
        """Get available quantity at best price."""
        if side == "buy":
            if is_yes:
                return orderbook.yes_ask_quantity
            return orderbook.no_ask_quantity
        else:
            if is_yes:
                return orderbook.yes_bid_quantity
            return orderbook.no_bid_quantity
