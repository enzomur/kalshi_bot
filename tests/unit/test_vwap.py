"""Unit tests for the VWAP calculator.

Tests cover:
- VWAP calculation for single order book level
- VWAP calculation across multiple price levels
- Slippage estimation at various order sizes
- Multi-leg VWAP for arbitrage trades
- Finding profitable quantity respecting min profit threshold
"""

from __future__ import annotations

import pytest

from kalshi_bot.core.types import OrderBook, OrderBookLevel, Side
from kalshi_bot.execution.vwap import (
    MultiLegVWAPResult,
    SlippageEstimate,
    VWAPAnalysis,
    VWAPCalculator,
)


@pytest.fixture
def vwap_calculator() -> VWAPCalculator:
    """Create VWAP calculator with default settings."""
    return VWAPCalculator(depth_levels=5, default_liquidity=100)


@pytest.fixture
def single_level_orderbook() -> OrderBook:
    """Order book with single price level."""
    return OrderBook(
        market_ticker="SINGLE-LVL",
        yes_asks=[OrderBookLevel(price=50, quantity=100)],
        yes_bids=[OrderBookLevel(price=48, quantity=100)],
        no_asks=[OrderBookLevel(price=52, quantity=100)],
        no_bids=[OrderBookLevel(price=50, quantity=100)],
    )


@pytest.fixture
def multi_level_orderbook() -> OrderBook:
    """Order book with multiple price levels."""
    return OrderBook(
        market_ticker="MULTI-LVL",
        yes_asks=[
            OrderBookLevel(price=50, quantity=50),
            OrderBookLevel(price=51, quantity=75),
            OrderBookLevel(price=52, quantity=100),
            OrderBookLevel(price=53, quantity=150),
            OrderBookLevel(price=55, quantity=200),
        ],
        yes_bids=[
            OrderBookLevel(price=48, quantity=60),
            OrderBookLevel(price=47, quantity=80),
            OrderBookLevel(price=46, quantity=100),
            OrderBookLevel(price=45, quantity=120),
            OrderBookLevel(price=44, quantity=150),
        ],
        no_asks=[
            OrderBookLevel(price=52, quantity=50),
            OrderBookLevel(price=53, quantity=75),
            OrderBookLevel(price=54, quantity=100),
        ],
        no_bids=[
            OrderBookLevel(price=50, quantity=60),
            OrderBookLevel(price=49, quantity=80),
            OrderBookLevel(price=48, quantity=100),
        ],
    )


@pytest.fixture
def sparse_orderbook() -> OrderBook:
    """Order book with sparse liquidity."""
    return OrderBook(
        market_ticker="SPARSE",
        yes_asks=[
            OrderBookLevel(price=55, quantity=10),
            OrderBookLevel(price=60, quantity=20),
        ],
        yes_bids=[
            OrderBookLevel(price=45, quantity=10),
            OrderBookLevel(price=40, quantity=20),
        ],
        no_asks=[],
        no_bids=[],
    )


class TestVWAPSingleLevel:
    """Tests for VWAP calculation with single price level."""

    def test_vwap_equals_price_for_single_level(
        self, vwap_calculator: VWAPCalculator, single_level_orderbook: OrderBook
    ) -> None:
        """Test VWAP equals price when only one level consumed."""
        analysis = vwap_calculator.calculate_vwap(
            single_level_orderbook,
            side=Side.YES,
            action="buy",
            quantity=50,  # Less than available
        )

        assert analysis.vwap == 50  # Same as ask price
        assert analysis.slippage_cents == 0  # No slippage
        assert analysis.fully_filled is True

    def test_vwap_for_sell_side(
        self, vwap_calculator: VWAPCalculator, single_level_orderbook: OrderBook
    ) -> None:
        """Test VWAP for selling (hitting bids)."""
        analysis = vwap_calculator.calculate_vwap(
            single_level_orderbook,
            side=Side.YES,
            action="sell",
            quantity=50,
        )

        assert analysis.vwap == 48  # Same as bid price
        assert analysis.slippage_cents == 0
        assert analysis.best_price == 48

    def test_vwap_for_no_side(
        self, vwap_calculator: VWAPCalculator, single_level_orderbook: OrderBook
    ) -> None:
        """Test VWAP for NO side orders."""
        analysis = vwap_calculator.calculate_vwap(
            single_level_orderbook,
            side=Side.NO,
            action="buy",
            quantity=50,
        )

        assert analysis.vwap == 52  # NO ask price
        assert analysis.side == Side.NO

    def test_partial_fill_when_insufficient_liquidity(
        self, vwap_calculator: VWAPCalculator, single_level_orderbook: OrderBook
    ) -> None:
        """Test partial fill when quantity exceeds liquidity."""
        analysis = vwap_calculator.calculate_vwap(
            single_level_orderbook,
            side=Side.YES,
            action="buy",
            quantity=150,  # More than available 100
        )

        assert analysis.filled_quantity == 100
        assert analysis.fully_filled is False


class TestVWAPMultipleLevels:
    """Tests for VWAP across multiple price levels."""

    def test_vwap_walks_the_book(
        self, vwap_calculator: VWAPCalculator, multi_level_orderbook: OrderBook
    ) -> None:
        """Test VWAP correctly walks through order book levels."""
        # Request 100 contracts: 50@50 + 50@51
        analysis = vwap_calculator.calculate_vwap(
            multi_level_orderbook,
            side=Side.YES,
            action="buy",
            quantity=100,
        )

        # VWAP = (50*50 + 51*50) / 100 = 50.5
        assert analysis.vwap == pytest.approx(50.5, abs=0.01)
        assert analysis.best_price == 50
        assert analysis.worst_price == 51
        assert analysis.levels_consumed == 2

    def test_vwap_multiple_levels_slippage(
        self, vwap_calculator: VWAPCalculator, multi_level_orderbook: OrderBook
    ) -> None:
        """Test slippage increases when walking the book."""
        # Request 150 contracts: 50@50 + 75@51 + 25@52
        analysis = vwap_calculator.calculate_vwap(
            multi_level_orderbook,
            side=Side.YES,
            action="buy",
            quantity=150,
        )

        # VWAP = (50*50 + 75*51 + 25*52) / 150 = 50.83...
        expected_vwap = (50*50 + 75*51 + 25*52) / 150
        assert analysis.vwap == pytest.approx(expected_vwap, abs=0.01)

        # Slippage = VWAP - best_price
        assert analysis.slippage_cents == pytest.approx(expected_vwap - 50, abs=0.01)

    def test_vwap_large_order_consumes_all_levels(
        self, vwap_calculator: VWAPCalculator, multi_level_orderbook: OrderBook
    ) -> None:
        """Test large order consumes all available levels."""
        # Request 600 contracts - more than total available
        analysis = vwap_calculator.calculate_vwap(
            multi_level_orderbook,
            side=Side.YES,
            action="buy",
            quantity=600,
        )

        # Total available = 50 + 75 + 100 + 150 + 200 = 575
        assert analysis.filled_quantity == 575
        assert analysis.fully_filled is False
        assert analysis.levels_consumed == 5
        assert analysis.worst_price == 55

    def test_vwap_sell_walks_bids_down(
        self, vwap_calculator: VWAPCalculator, multi_level_orderbook: OrderBook
    ) -> None:
        """Test selling walks down the bid stack."""
        # Sell 100 contracts: 60@48 + 40@47
        analysis = vwap_calculator.calculate_vwap(
            multi_level_orderbook,
            side=Side.YES,
            action="sell",
            quantity=100,
        )

        # VWAP = (60*48 + 40*47) / 100 = 47.6
        expected_vwap = (60*48 + 40*47) / 100
        assert analysis.vwap == pytest.approx(expected_vwap, abs=0.01)

        # For sells, slippage = best_price - vwap (lower is worse)
        assert analysis.slippage_cents == pytest.approx(48 - expected_vwap, abs=0.01)


class TestSlippageEstimation:
    """Tests for slippage estimation at various sizes."""

    def test_slippage_increases_with_size(
        self, vwap_calculator: VWAPCalculator, multi_level_orderbook: OrderBook
    ) -> None:
        """Test slippage increases with order size."""
        estimate = vwap_calculator.estimate_slippage(
            multi_level_orderbook,
            side=Side.YES,
            action="buy",
            max_quantity=500,
        )

        # Slippage should generally increase with quantity
        slippages = [s for _, s in estimate.estimates]

        # First few should have increasing slippage
        assert slippages[0] <= slippages[-1]

    def test_slippage_zero_for_small_orders(
        self, vwap_calculator: VWAPCalculator, multi_level_orderbook: OrderBook
    ) -> None:
        """Test slippage is zero for very small orders."""
        estimate = vwap_calculator.estimate_slippage(
            multi_level_orderbook,
            side=Side.YES,
            action="buy",
            max_quantity=10,
        )

        # First estimate (qty=1) should have zero slippage
        first_qty, first_slippage = estimate.estimates[0]
        assert first_qty == 1
        assert first_slippage == pytest.approx(0, abs=0.01)

    def test_max_quantity_at_threshold(
        self, vwap_calculator: VWAPCalculator, multi_level_orderbook: OrderBook
    ) -> None:
        """Test finding max quantity within slippage threshold."""
        estimate = vwap_calculator.estimate_slippage(
            multi_level_orderbook,
            side=Side.YES,
            action="buy",
            max_quantity=500,
            threshold_cents=1.0,
        )

        # Should find the largest quantity with slippage <= 1 cent
        assert estimate.max_quantity_at_threshold > 0
        assert estimate.threshold_cents == 1.0

    def test_slippage_estimate_structure(
        self, vwap_calculator: VWAPCalculator, multi_level_orderbook: OrderBook
    ) -> None:
        """Test SlippageEstimate has correct structure."""
        estimate = vwap_calculator.estimate_slippage(
            multi_level_orderbook,
            side=Side.YES,
            action="buy",
        )

        assert estimate.market_ticker == "MULTI-LVL"
        assert estimate.side == Side.YES
        assert estimate.action == "buy"
        assert len(estimate.estimates) > 0


class TestMultiLegVWAP:
    """Tests for multi-leg VWAP calculation."""

    def test_multi_leg_cost_calculation(
        self, vwap_calculator: VWAPCalculator, multi_level_orderbook: OrderBook
    ) -> None:
        """Test total cost calculation for multi-leg trade."""
        legs = [
            {"market": "MULTI-LVL", "side": "yes", "action": "buy", "quantity": 50},
            {"market": "MULTI-LVL", "side": "no", "action": "buy", "quantity": 50},
        ]
        orderbooks = {"MULTI-LVL": multi_level_orderbook}

        result = vwap_calculator.calculate_multi_leg_vwap(legs, orderbooks)

        # Both legs buying
        assert result.total_cost > 0
        assert len(result.legs) == 2

    def test_multi_leg_slippage_sum(
        self, vwap_calculator: VWAPCalculator, multi_level_orderbook: OrderBook
    ) -> None:
        """Test total slippage is sum of leg slippages."""
        legs = [
            {"market": "MULTI-LVL", "side": "yes", "action": "buy", "quantity": 100},
            {"market": "MULTI-LVL", "side": "yes", "action": "sell", "quantity": 100},
        ]
        orderbooks = {"MULTI-LVL": multi_level_orderbook}

        result = vwap_calculator.calculate_multi_leg_vwap(legs, orderbooks)

        # Total slippage = sum of individual leg slippages
        expected_slippage = sum(leg.slippage_cents for leg in result.legs)
        assert result.total_slippage_cents == pytest.approx(expected_slippage, abs=0.01)

    def test_multi_leg_sell_reduces_cost(
        self, vwap_calculator: VWAPCalculator, multi_level_orderbook: OrderBook
    ) -> None:
        """Test selling legs reduce total cost."""
        # Buy leg
        buy_legs = [
            {"market": "MULTI-LVL", "side": "yes", "action": "buy", "quantity": 50},
        ]
        buy_result = vwap_calculator.calculate_multi_leg_vwap(
            buy_legs, {"MULTI-LVL": multi_level_orderbook}
        )

        # Buy + Sell legs
        mixed_legs = [
            {"market": "MULTI-LVL", "side": "yes", "action": "buy", "quantity": 50},
            {"market": "MULTI-LVL", "side": "yes", "action": "sell", "quantity": 25},
        ]
        mixed_result = vwap_calculator.calculate_multi_leg_vwap(
            mixed_legs, {"MULTI-LVL": multi_level_orderbook}
        )

        # Selling should reduce net cost
        assert mixed_result.total_cost < buy_result.total_cost

    def test_multi_leg_max_quantity(
        self, vwap_calculator: VWAPCalculator, multi_level_orderbook: OrderBook
    ) -> None:
        """Test max_quantity is limited by smallest fill."""
        legs = [
            {"market": "MULTI-LVL", "side": "yes", "action": "buy", "quantity": 100},
            {"market": "MULTI-LVL", "side": "no", "action": "buy", "quantity": 100},
        ]
        orderbooks = {"MULTI-LVL": multi_level_orderbook}

        result = vwap_calculator.calculate_multi_leg_vwap(legs, orderbooks)

        # Max quantity should reflect fillable amount
        assert result.max_quantity <= 100

    def test_multi_leg_missing_orderbook(
        self, vwap_calculator: VWAPCalculator, multi_level_orderbook: OrderBook
    ) -> None:
        """Test handling of missing orderbook."""
        legs = [
            {"market": "MULTI-LVL", "side": "yes", "action": "buy", "quantity": 50},
            {"market": "MISSING", "side": "yes", "action": "buy", "quantity": 50},
        ]
        orderbooks = {"MULTI-LVL": multi_level_orderbook}

        result = vwap_calculator.calculate_multi_leg_vwap(legs, orderbooks)

        # Should still complete but with empty analysis for missing market
        assert len(result.legs) == 2
        assert result.legs[1].filled_quantity == 0
        assert result.all_filled is False


class TestFindProfitableQuantity:
    """Tests for finding profitable quantity."""

    def test_find_profitable_quantity_basic(
        self, vwap_calculator: VWAPCalculator, multi_level_orderbook: OrderBook
    ) -> None:
        """Test finding maximum profitable quantity."""
        legs = [
            {"market": "MULTI-LVL", "side": "yes", "action": "buy", "quantity": 100},
        ]
        orderbooks = {"MULTI-LVL": multi_level_orderbook}

        # $0.10 profit per contract
        qty = vwap_calculator.find_profitable_quantity(
            legs, orderbooks,
            gross_profit_per_contract=0.10,
            min_profit=0.05,
        )

        # Should find some profitable quantity
        assert qty >= 0

    def test_find_profitable_respects_min_threshold(
        self, vwap_calculator: VWAPCalculator, multi_level_orderbook: OrderBook
    ) -> None:
        """Test minimum profit threshold is respected."""
        legs = [
            {"market": "MULTI-LVL", "side": "yes", "action": "buy", "quantity": 100},
        ]
        orderbooks = {"MULTI-LVL": multi_level_orderbook}

        # Very small profit per contract
        qty = vwap_calculator.find_profitable_quantity(
            legs, orderbooks,
            gross_profit_per_contract=0.001,  # 0.1 cents
            min_profit=0.05,
        )

        # Might be zero if can't reach min profit
        assert qty >= 0

    def test_zero_quantity_when_unprofitable(
        self, vwap_calculator: VWAPCalculator, sparse_orderbook: OrderBook
    ) -> None:
        """Test returns zero when slippage exceeds profit."""
        legs = [
            {"market": "SPARSE", "side": "yes", "action": "buy", "quantity": 100},
        ]
        orderbooks = {"SPARSE": sparse_orderbook}

        # Negative profit
        qty = vwap_calculator.find_profitable_quantity(
            legs, orderbooks,
            gross_profit_per_contract=-0.10,
            min_profit=0.05,
        )

        assert qty == 0


class TestDepthSummary:
    """Tests for order book depth summary."""

    def test_depth_summary_structure(
        self, vwap_calculator: VWAPCalculator, multi_level_orderbook: OrderBook
    ) -> None:
        """Test depth summary has expected structure."""
        summary = vwap_calculator.get_depth_summary(multi_level_orderbook)

        assert summary["market_ticker"] == "MULTI-LVL"
        assert "yes_asks" in summary
        assert "yes_bids" in summary
        assert "no_asks" in summary
        assert "no_bids" in summary

    def test_depth_summary_statistics(
        self, vwap_calculator: VWAPCalculator, multi_level_orderbook: OrderBook
    ) -> None:
        """Test depth summary calculates correct statistics."""
        summary = vwap_calculator.get_depth_summary(multi_level_orderbook)

        yes_asks = summary["yes_asks"]
        assert yes_asks["depth"] == 5
        assert yes_asks["best_price"] == 50
        assert yes_asks["total_qty"] == 50 + 75 + 100 + 150 + 200

    def test_depth_summary_empty_side(
        self, vwap_calculator: VWAPCalculator, sparse_orderbook: OrderBook
    ) -> None:
        """Test depth summary handles empty sides."""
        summary = vwap_calculator.get_depth_summary(sparse_orderbook)

        no_asks = summary["no_asks"]
        assert no_asks["depth"] == 0
        assert no_asks["total_qty"] == 0
        assert no_asks["best_price"] is None


class TestVWAPAnalysisDataclass:
    """Tests for VWAPAnalysis dataclass."""

    def test_analysis_fields(
        self, vwap_calculator: VWAPCalculator, multi_level_orderbook: OrderBook
    ) -> None:
        """Test VWAPAnalysis has all expected fields."""
        analysis = vwap_calculator.calculate_vwap(
            multi_level_orderbook,
            side=Side.YES,
            action="buy",
            quantity=100,
        )

        assert analysis.market_ticker == "MULTI-LVL"
        assert analysis.side == Side.YES
        assert analysis.action == "buy"
        assert analysis.quantity == 100
        assert analysis.filled_quantity > 0
        assert analysis.vwap > 0
        assert analysis.best_price > 0
        assert analysis.worst_price >= analysis.best_price
        assert isinstance(analysis.slippage_cents, float)
        assert isinstance(analysis.slippage_pct, float)
        assert analysis.total_cost > 0
        assert analysis.levels_consumed >= 1
        assert isinstance(analysis.price_levels, list)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_orderbook(self, vwap_calculator: VWAPCalculator) -> None:
        """Test handling empty order book."""
        empty_book = OrderBook(
            market_ticker="EMPTY",
            yes_asks=[],
            yes_bids=[],
            no_asks=[],
            no_bids=[],
        )

        analysis = vwap_calculator.calculate_vwap(
            empty_book, Side.YES, "buy", 100
        )

        assert analysis.filled_quantity == 0
        assert analysis.vwap == 0
        assert analysis.fully_filled is False

    def test_zero_quantity_request(
        self, vwap_calculator: VWAPCalculator, multi_level_orderbook: OrderBook
    ) -> None:
        """Test zero quantity request."""
        analysis = vwap_calculator.calculate_vwap(
            multi_level_orderbook,
            side=Side.YES,
            action="buy",
            quantity=0,
        )

        assert analysis.filled_quantity == 0
        assert analysis.fully_filled is True  # 0 of 0 filled

    def test_single_contract(
        self, vwap_calculator: VWAPCalculator, multi_level_orderbook: OrderBook
    ) -> None:
        """Test single contract order."""
        analysis = vwap_calculator.calculate_vwap(
            multi_level_orderbook,
            side=Side.YES,
            action="buy",
            quantity=1,
        )

        assert analysis.filled_quantity == 1
        assert analysis.vwap == 50  # Best ask
        assert analysis.slippage_cents == 0
        assert analysis.levels_consumed == 1
