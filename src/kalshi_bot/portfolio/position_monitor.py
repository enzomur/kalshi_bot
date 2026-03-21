"""Position monitor for take-profit automation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, AsyncIterator

from kalshi_bot.core.types import Position, Side
from kalshi_bot.utils.logging import get_logger

if TYPE_CHECKING:
    from kalshi_bot.api.client import KalshiAPIClient
    from kalshi_bot.config.settings import Settings

logger = get_logger(__name__)


@dataclass
class SellSignal:
    """Signal to sell a position."""

    position: Position
    reason: str
    current_price: float
    pnl_pct: float


class PositionMonitor:
    """
    Monitors positions for take-profit conditions.

    Checks each position's current market price against entry price
    and generates sell signals when profit targets are hit.
    """

    def __init__(self, settings: Settings) -> None:
        """
        Initialize position monitor.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self._take_profit_pct = settings.portfolio.take_profit_pct

    async def check_positions(
        self,
        positions: list[Position],
        api_client: KalshiAPIClient,
    ) -> AsyncIterator[SellSignal]:
        """
        Check all positions for take-profit conditions.

        Args:
            positions: List of current positions
            api_client: API client for fetching market prices

        Yields:
            SellSignal for each position that hits profit target
        """
        for position in positions:
            if position.quantity <= 0:
                continue

            try:
                current_price = await self._get_current_price(position, api_client)
                if current_price is None:
                    continue

                entry_price = position.average_price
                if entry_price <= 0:
                    # Try to get entry price from fills API
                    entry_price = await self._get_entry_price_from_fills(
                        position, api_client
                    )
                    if entry_price <= 0:
                        continue

                pnl_pct = (current_price - entry_price) / entry_price

                if pnl_pct >= self._take_profit_pct:
                    logger.info(
                        f"Take-profit triggered for {position.market_ticker}: "
                        f"entry={entry_price:.0f}c, current={current_price:.0f}c, "
                        f"pnl={pnl_pct:.1%}"
                    )
                    yield SellSignal(
                        position=position,
                        reason="take_profit",
                        current_price=current_price,
                        pnl_pct=pnl_pct,
                    )

            except Exception as e:
                logger.warning(
                    f"Error checking position {position.market_ticker}: {e}"
                )

    async def _get_current_price(
        self,
        position: Position,
        api_client: KalshiAPIClient,
    ) -> float | None:
        """
        Get current market price for a position.

        For YES positions, we look at the best bid (what we can sell for).
        For NO positions, we look at the best no bid.

        Args:
            position: The position to price
            api_client: API client

        Returns:
            Current price in cents, or None if unavailable
        """
        market = await api_client.get_market(position.market_ticker)

        if position.side == Side.YES:
            return float(market.yes_bid) if market.yes_bid is not None else None
        else:
            return float(market.no_bid) if market.no_bid is not None else None

    async def _get_entry_price_from_fills(
        self,
        position: Position,
        api_client: KalshiAPIClient,
    ) -> float:
        """
        Get average entry price from fills history.

        Args:
            position: The position
            api_client: API client

        Returns:
            Average entry price in cents, or 0 if unavailable
        """
        try:
            fills_data = await api_client.get_fills(ticker=position.market_ticker, limit=500)
            fills = fills_data.get("fills", [])

            if not fills:
                return 0.0

            total_cost = 0.0
            total_qty = 0

            for fill in fills:
                # Only count buys on our side
                if fill.get("action") == "buy" and fill.get("side") == position.side.value:
                    # Use the correct price field based on side
                    if position.side == Side.YES:
                        price = fill.get("yes_price", 0)
                    else:
                        price = fill.get("no_price", 0)
                    qty = fill.get("count", 0)
                    # fee_cost is in centi-cents (divide by 100 to get cents)
                    fee = float(fill.get("fee_cost", 0)) / 100
                    if price > 0 and qty > 0:
                        total_cost += float(price) * float(qty) + fee
                        total_qty += qty

            if total_qty > 0:
                avg_price = total_cost / total_qty
                logger.info(f"Calculated entry price for {position.market_ticker}: {avg_price:.4f}c from {total_qty} contracts")
                return avg_price

            return 0.0
        except Exception as e:
            logger.warning(f"Failed to get fills for {position.market_ticker}: {e}")
            return 0.0
