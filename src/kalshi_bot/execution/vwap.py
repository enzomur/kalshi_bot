"""Volume-Weighted Average Price (VWAP) calculator for realistic execution pricing.

Based on: "Unravelling the Probabilistic Forest: Arbitrage in Prediction Markets"
(arXiv:2508.03474v1)

The paper uses VWAP analysis to determine realistic fill prices when executing
arbitrage trades across order book depth. This accounts for slippage and helps
filter out opportunities that aren't profitable after execution costs.

Key formula: VWAP = Σ(price_i × volume_i) / Σ(volume_i)

The paper uses a $0.05 minimum profit threshold to account for execution risk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from kalshi_bot.core.types import OrderBook, OrderBookLevel, Side
from kalshi_bot.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class VWAPAnalysis:
    """Result of VWAP calculation for a single order."""

    market_ticker: str
    side: Side  # YES or NO
    action: str  # "buy" or "sell"
    quantity: int  # Requested quantity
    filled_quantity: int  # Quantity that can be filled
    vwap: float  # Volume-weighted average price in cents
    best_price: float  # Best available price in cents
    worst_price: float  # Worst price in the fill in cents
    slippage_cents: float  # VWAP - best_price (positive = worse execution)
    slippage_pct: float  # Slippage as percentage of best price
    total_cost: float  # Total cost in dollars
    levels_consumed: int  # Number of order book levels used
    fully_filled: bool  # Whether requested quantity can be fully filled
    price_levels: list[tuple[int, int]] = field(default_factory=list)  # (price, qty) pairs


@dataclass
class SlippageEstimate:
    """Estimated slippage for different order sizes."""

    market_ticker: str
    side: Side
    action: str
    estimates: list[tuple[int, float]]  # (quantity, slippage_cents) pairs
    max_quantity_at_threshold: int  # Max quantity with slippage < threshold
    threshold_cents: float  # Slippage threshold used


@dataclass
class MultiLegVWAPResult:
    """VWAP analysis for a multi-leg arbitrage trade."""

    legs: list[VWAPAnalysis]
    total_cost: float  # Total cost in dollars
    total_slippage_cents: float  # Sum of slippage across legs
    average_slippage_pct: float  # Average slippage percentage
    max_quantity: int  # Maximum quantity fillable across all legs
    all_filled: bool  # Whether all legs can be fully filled
    expected_profit: float  # Expected profit after slippage in dollars
    profitable: bool  # Whether trade is still profitable


class VWAPCalculator:
    """
    Calculate realistic fill prices across order book depth.

    Provides VWAP analysis for:
    - Single orders (buy/sell YES or NO)
    - Multi-leg arbitrage trades
    - Slippage estimation for position sizing
    """

    def __init__(
        self,
        depth_levels: int = 5,
        default_liquidity: int = 100,
    ) -> None:
        """
        Initialize VWAP calculator.

        Args:
            depth_levels: Number of order book levels to consider
            default_liquidity: Default quantity when level data is missing
        """
        self.depth_levels = depth_levels
        self.default_liquidity = default_liquidity

    def calculate_vwap(
        self,
        orderbook: OrderBook,
        side: Side,
        action: str,
        quantity: int,
    ) -> VWAPAnalysis:
        """
        Calculate VWAP for an order.

        Args:
            orderbook: Order book for the market
            side: YES or NO side
            action: "buy" or "sell"
            quantity: Number of contracts

        Returns:
            VWAPAnalysis with fill details
        """
        # Get relevant price levels
        levels = self._get_price_levels(orderbook, side, action)

        if not levels:
            return VWAPAnalysis(
                market_ticker=orderbook.market_ticker,
                side=side,
                action=action,
                quantity=quantity,
                filled_quantity=0,
                vwap=0.0,
                best_price=0.0,
                worst_price=0.0,
                slippage_cents=0.0,
                slippage_pct=0.0,
                total_cost=0.0,
                levels_consumed=0,
                fully_filled=False,
            )

        # Calculate VWAP across levels
        total_value = 0.0
        total_quantity = 0
        worst_price = levels[0].price
        price_levels_used: list[tuple[int, int]] = []

        for level in levels:
            available = level.quantity
            needed = quantity - total_quantity

            if needed <= 0:
                break

            fill_qty = min(available, needed)
            total_value += level.price * fill_qty
            total_quantity += fill_qty
            worst_price = level.price
            price_levels_used.append((level.price, fill_qty))

        best_price = levels[0].price
        vwap = total_value / total_quantity if total_quantity > 0 else 0.0

        # Slippage is the difference from best price
        # For buys: higher VWAP = worse (positive slippage)
        # For sells: lower VWAP = worse (positive slippage)
        if action == "buy":
            slippage_cents = vwap - best_price
        else:
            slippage_cents = best_price - vwap

        slippage_pct = (slippage_cents / best_price * 100) if best_price > 0 else 0.0

        # Total cost in dollars
        total_cost = (vwap * total_quantity) / 100

        return VWAPAnalysis(
            market_ticker=orderbook.market_ticker,
            side=side,
            action=action,
            quantity=quantity,
            filled_quantity=total_quantity,
            vwap=vwap,
            best_price=best_price,
            worst_price=worst_price,
            slippage_cents=slippage_cents,
            slippage_pct=slippage_pct,
            total_cost=total_cost,
            levels_consumed=len(price_levels_used),
            fully_filled=total_quantity >= quantity,
            price_levels=price_levels_used,
        )

    def _get_price_levels(
        self,
        orderbook: OrderBook,
        side: Side,
        action: str,
    ) -> list[OrderBookLevel]:
        """Get relevant price levels from order book."""
        if side == Side.YES:
            if action == "buy":
                # Buying YES: take from asks (ascending price)
                return orderbook.yes_asks[:self.depth_levels]
            else:
                # Selling YES: hit bids (descending price)
                return orderbook.yes_bids[:self.depth_levels]
        else:  # NO
            if action == "buy":
                return orderbook.no_asks[:self.depth_levels]
            else:
                return orderbook.no_bids[:self.depth_levels]

    def estimate_slippage(
        self,
        orderbook: OrderBook,
        side: Side,
        action: str,
        max_quantity: int = 1000,
        threshold_cents: float = 5.0,
    ) -> SlippageEstimate:
        """
        Estimate slippage at various order sizes.

        Args:
            orderbook: Order book for the market
            side: YES or NO
            action: "buy" or "sell"
            max_quantity: Maximum quantity to analyze
            threshold_cents: Slippage threshold for max quantity calculation

        Returns:
            SlippageEstimate with slippage at various sizes
        """
        estimates: list[tuple[int, float]] = []
        max_quantity_at_threshold = 0

        # Sample at various quantities
        quantities = [1, 5, 10, 25, 50, 100, 250, 500, 1000]
        quantities = [q for q in quantities if q <= max_quantity]

        for qty in quantities:
            analysis = self.calculate_vwap(orderbook, side, action, qty)
            slippage = analysis.slippage_cents

            estimates.append((qty, slippage))

            if slippage <= threshold_cents:
                max_quantity_at_threshold = qty

        return SlippageEstimate(
            market_ticker=orderbook.market_ticker,
            side=side,
            action=action,
            estimates=estimates,
            max_quantity_at_threshold=max_quantity_at_threshold,
            threshold_cents=threshold_cents,
        )

    def calculate_multi_leg_vwap(
        self,
        legs: list[dict[str, Any]],
        orderbooks: dict[str, OrderBook],
        expected_gross_profit: float = 0.0,
    ) -> MultiLegVWAPResult:
        """
        Calculate VWAP for a multi-leg arbitrage trade.

        Args:
            legs: List of leg definitions with market, side, action, quantity
            orderbooks: Dictionary of order books keyed by ticker
            expected_gross_profit: Expected gross profit in dollars (before slippage)

        Returns:
            MultiLegVWAPResult with combined analysis
        """
        leg_analyses: list[VWAPAnalysis] = []
        total_cost = 0.0
        total_slippage = 0.0
        min_filled_pct = 1.0

        for leg in legs:
            ticker = leg["market"]
            orderbook = orderbooks.get(ticker)

            if orderbook is None:
                # Create empty analysis for missing orderbook
                analysis = VWAPAnalysis(
                    market_ticker=ticker,
                    side=Side(leg["side"]),
                    action=leg["action"],
                    quantity=leg["quantity"],
                    filled_quantity=0,
                    vwap=0.0,
                    best_price=0.0,
                    worst_price=0.0,
                    slippage_cents=0.0,
                    slippage_pct=0.0,
                    total_cost=0.0,
                    levels_consumed=0,
                    fully_filled=False,
                )
            else:
                analysis = self.calculate_vwap(
                    orderbook,
                    Side(leg["side"]),
                    leg["action"],
                    leg["quantity"],
                )

            leg_analyses.append(analysis)

            if leg["action"] == "buy":
                total_cost += analysis.total_cost
            else:
                total_cost -= analysis.total_cost  # Sells generate credit

            total_slippage += analysis.slippage_cents

            # Track minimum fill percentage
            if analysis.quantity > 0:
                fill_pct = analysis.filled_quantity / analysis.quantity
                min_filled_pct = min(min_filled_pct, fill_pct)

        # Calculate max quantity that can be filled across all legs
        max_quantity = int(min_filled_pct * legs[0]["quantity"]) if legs else 0

        # Average slippage percentage
        if leg_analyses:
            avg_slippage = sum(a.slippage_pct for a in leg_analyses) / len(leg_analyses)
        else:
            avg_slippage = 0.0

        # Expected profit after slippage
        slippage_cost = (total_slippage * max_quantity) / 100  # Convert to dollars
        expected_profit = expected_gross_profit - slippage_cost

        return MultiLegVWAPResult(
            legs=leg_analyses,
            total_cost=total_cost,
            total_slippage_cents=total_slippage,
            average_slippage_pct=avg_slippage,
            max_quantity=max_quantity,
            all_filled=all(a.fully_filled for a in leg_analyses),
            expected_profit=expected_profit,
            profitable=expected_profit > 0,
        )

    def find_profitable_quantity(
        self,
        legs: list[dict[str, Any]],
        orderbooks: dict[str, OrderBook],
        gross_profit_per_contract: float,
        min_profit: float = 0.05,
    ) -> int:
        """
        Find maximum quantity that maintains minimum profit threshold.

        Based on the paper's $0.05 minimum profit threshold.

        Args:
            legs: Leg definitions
            orderbooks: Order books
            gross_profit_per_contract: Gross profit in dollars per contract
            min_profit: Minimum acceptable profit in dollars

        Returns:
            Maximum profitable quantity
        """
        # Binary search for maximum profitable quantity
        low, high = 1, 1000
        result = 0

        while low <= high:
            mid = (low + high) // 2

            # Update leg quantities
            test_legs = []
            for leg in legs:
                test_leg = leg.copy()
                test_leg["quantity"] = mid
                test_legs.append(test_leg)

            # Calculate VWAP
            analysis = self.calculate_multi_leg_vwap(
                test_legs,
                orderbooks,
                gross_profit_per_contract * mid,
            )

            if analysis.expected_profit >= min_profit and analysis.all_filled:
                result = mid
                low = mid + 1
            else:
                high = mid - 1

        return result

    def get_depth_summary(
        self,
        orderbook: OrderBook,
    ) -> dict[str, Any]:
        """
        Get summary of order book depth.

        Args:
            orderbook: Order book to analyze

        Returns:
            Dictionary with depth statistics
        """
        def level_stats(levels: list[OrderBookLevel]) -> dict[str, Any]:
            if not levels:
                return {"depth": 0, "total_qty": 0, "best_price": None, "avg_price": None}

            total_qty = sum(l.quantity for l in levels)
            if total_qty > 0:
                vwap = sum(l.price * l.quantity for l in levels) / total_qty
            else:
                vwap = 0

            return {
                "depth": len(levels),
                "total_qty": total_qty,
                "best_price": levels[0].price,
                "avg_price": vwap,
            }

        return {
            "market_ticker": orderbook.market_ticker,
            "yes_asks": level_stats(orderbook.yes_asks),
            "yes_bids": level_stats(orderbook.yes_bids),
            "no_asks": level_stats(orderbook.no_asks),
            "no_bids": level_stats(orderbook.no_bids),
        }
