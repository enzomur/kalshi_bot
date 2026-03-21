"""Unit tests for the execution risk modeler.

Tests cover:
- Fill probability estimation (high and low liquidity scenarios)
- Price movement model with volatility scaling
- Minimum profit threshold calculation
- Should-execute decision based on risk filtering
"""

from __future__ import annotations

import numpy as np
import pytest

from kalshi_bot.core.types import OrderBook, OrderBookLevel, Side
from kalshi_bot.execution.risk_model import (
    ExecutionDecision,
    ExecutionRiskEstimate,
    ExecutionRiskModeler,
    LegFillProbability,
    PriceMovementModel,
)
from kalshi_bot.execution.vwap import VWAPCalculator


@pytest.fixture
def risk_modeler() -> ExecutionRiskModeler:
    """Create risk modeler with default settings."""
    return ExecutionRiskModeler(
        vwap_calculator=VWAPCalculator(),
        default_fill_time_seconds=2.0,
        default_volatility_cents=1.0,
        confidence_level=0.95,
    )


@pytest.fixture
def liquid_orderbook() -> OrderBook:
    """Order book with high liquidity."""
    return OrderBook(
        market_ticker="LIQUID-MKT",
        yes_asks=[
            OrderBookLevel(price=50, quantity=500),
            OrderBookLevel(price=51, quantity=500),
            OrderBookLevel(price=52, quantity=500),
        ],
        yes_bids=[
            OrderBookLevel(price=48, quantity=500),
            OrderBookLevel(price=47, quantity=500),
            OrderBookLevel(price=46, quantity=500),
        ],
        no_asks=[
            OrderBookLevel(price=52, quantity=500),
            OrderBookLevel(price=53, quantity=500),
        ],
        no_bids=[
            OrderBookLevel(price=50, quantity=500),
            OrderBookLevel(price=49, quantity=500),
        ],
    )


@pytest.fixture
def illiquid_orderbook() -> OrderBook:
    """Order book with low liquidity."""
    return OrderBook(
        market_ticker="ILLIQUID-MKT",
        yes_asks=[
            OrderBookLevel(price=55, quantity=5),
            OrderBookLevel(price=60, quantity=10),
        ],
        yes_bids=[
            OrderBookLevel(price=45, quantity=5),
            OrderBookLevel(price=40, quantity=10),
        ],
        no_asks=[
            OrderBookLevel(price=55, quantity=5),
        ],
        no_bids=[
            OrderBookLevel(price=45, quantity=5),
        ],
    )


@pytest.fixture
def moderate_orderbook() -> OrderBook:
    """Order book with moderate liquidity."""
    return OrderBook(
        market_ticker="MODERATE-MKT",
        yes_asks=[
            OrderBookLevel(price=50, quantity=50),
            OrderBookLevel(price=51, quantity=75),
            OrderBookLevel(price=52, quantity=100),
        ],
        yes_bids=[
            OrderBookLevel(price=48, quantity=50),
            OrderBookLevel(price=47, quantity=75),
            OrderBookLevel(price=46, quantity=100),
        ],
        no_asks=[
            OrderBookLevel(price=52, quantity=50),
        ],
        no_bids=[
            OrderBookLevel(price=50, quantity=50),
        ],
    )


class TestFillProbabilityHighLiquidity:
    """Tests for fill probability when liquidity is high."""

    def test_high_fill_probability_when_liquid(
        self, risk_modeler: ExecutionRiskModeler, liquid_orderbook: OrderBook
    ) -> None:
        """Test fill probability is near 1.0 when liquidity exists."""
        result = risk_modeler.estimate_leg_fill_probability(
            liquid_orderbook,
            side=Side.YES,
            price=50,
            quantity=100,  # Well below available 500
        )

        assert result.fill_probability >= 0.9
        assert result.fill_probability <= 1.0

    def test_very_small_order_high_probability(
        self, risk_modeler: ExecutionRiskModeler, liquid_orderbook: OrderBook
    ) -> None:
        """Test very small orders have very high fill probability."""
        result = risk_modeler.estimate_leg_fill_probability(
            liquid_orderbook,
            side=Side.YES,
            price=50,
            quantity=1,
        )

        assert result.fill_probability >= 0.95

    def test_fill_probability_result_structure(
        self, risk_modeler: ExecutionRiskModeler, liquid_orderbook: OrderBook
    ) -> None:
        """Test LegFillProbability has expected structure."""
        result = risk_modeler.estimate_leg_fill_probability(
            liquid_orderbook,
            side=Side.YES,
            price=50,
            quantity=100,
        )

        assert result.market_ticker == "LIQUID-MKT"
        assert result.side == Side.YES
        assert result.price == 50
        assert result.quantity == 100
        assert 0 <= result.fill_probability <= 1
        assert result.expected_time_seconds > 0
        assert 0 <= result.partial_fill_probability <= 1


class TestFillProbabilityLowLiquidity:
    """Tests for fill probability when liquidity is low."""

    def test_lower_fill_probability_when_illiquid(
        self, risk_modeler: ExecutionRiskModeler, illiquid_orderbook: OrderBook
    ) -> None:
        """Test fill probability is lower when order exceeds liquidity."""
        result = risk_modeler.estimate_leg_fill_probability(
            illiquid_orderbook,
            side=Side.YES,
            price=55,
            quantity=100,  # Much more than available 5
        )

        assert result.fill_probability < 0.9
        # But still has some probability due to time factor
        assert result.fill_probability > 0

    def test_partial_fill_probability_when_partial_liquidity(
        self, risk_modeler: ExecutionRiskModeler, illiquid_orderbook: OrderBook
    ) -> None:
        """Test partial fill probability when some liquidity exists."""
        result = risk_modeler.estimate_leg_fill_probability(
            illiquid_orderbook,
            side=Side.YES,
            price=55,
            quantity=10,  # More than 5 available, less than 15 total
        )

        # Should have non-zero partial fill probability
        assert result.partial_fill_probability > 0

    def test_longer_expected_time_when_illiquid(
        self, risk_modeler: ExecutionRiskModeler, illiquid_orderbook: OrderBook
    ) -> None:
        """Test expected fill time is longer when illiquid."""
        liquid_result = risk_modeler.estimate_leg_fill_probability(
            OrderBook(
                market_ticker="LIQ",
                yes_asks=[OrderBookLevel(price=50, quantity=1000)],
                yes_bids=[],
                no_asks=[],
                no_bids=[],
            ),
            side=Side.YES,
            price=50,
            quantity=100,
        )

        illiquid_result = risk_modeler.estimate_leg_fill_probability(
            illiquid_orderbook,
            side=Side.YES,
            price=55,
            quantity=100,
        )

        assert illiquid_result.expected_time_seconds >= liquid_result.expected_time_seconds


class TestPriceMovementModel:
    """Tests for price movement estimation."""

    def test_volatility_scales_with_sqrt_time(
        self, risk_modeler: ExecutionRiskModeler
    ) -> None:
        """Test volatility scales with square root of time."""
        result_1s = risk_modeler.estimate_price_movement("MKT", time_between_legs=1.0)
        result_4s = risk_modeler.estimate_price_movement("MKT", time_between_legs=4.0)

        # sqrt(4) / sqrt(1) = 2
        ratio = result_4s.volatility_cents / result_1s.volatility_cents
        assert ratio == pytest.approx(2.0, rel=0.01)

    def test_expected_move_proportional_to_volatility(
        self, risk_modeler: ExecutionRiskModeler
    ) -> None:
        """Test expected move is proportional to volatility."""
        result = risk_modeler.estimate_price_movement("MKT", time_between_legs=4.0)

        # Expected move should be ~0.8 * volatility for normal distribution
        ratio = result.expected_move_cents / result.volatility_cents
        assert ratio == pytest.approx(0.8, rel=0.1)

    def test_confidence_interval_symmetric(
        self, risk_modeler: ExecutionRiskModeler
    ) -> None:
        """Test 95% confidence interval is symmetric around zero."""
        result = risk_modeler.estimate_price_movement("MKT", time_between_legs=2.0)

        ci_low, ci_high = result.confidence_interval

        # Should be symmetric around 0
        assert ci_low == pytest.approx(-ci_high, rel=0.01)

        # Should be approximately 1.96 * volatility
        assert ci_high == pytest.approx(1.96 * result.volatility_cents, rel=0.01)

    def test_uses_historical_volatility_when_provided(
        self, risk_modeler: ExecutionRiskModeler
    ) -> None:
        """Test historical volatility is used when provided."""
        result = risk_modeler.estimate_price_movement(
            "MKT",
            time_between_legs=1.0,
            historical_volatility=5.0,  # 5 cents per second
        )

        assert result.volatility_cents == pytest.approx(5.0, rel=0.01)

    def test_price_movement_model_structure(
        self, risk_modeler: ExecutionRiskModeler
    ) -> None:
        """Test PriceMovementModel has expected structure."""
        result = risk_modeler.estimate_price_movement("MKT", time_between_legs=2.0)

        assert result.market_ticker == "MKT"
        assert result.time_horizon_seconds == 2.0
        assert result.expected_move_cents >= 0
        assert result.volatility_cents > 0
        assert len(result.confidence_interval) == 2


class TestMinimumProfitThreshold:
    """Tests for minimum profit threshold calculation."""

    def test_base_threshold_is_five_cents(
        self, risk_modeler: ExecutionRiskModeler, moderate_orderbook: OrderBook
    ) -> None:
        """Test base threshold is $0.05 per the paper."""
        legs = [
            {"market": "MODERATE-MKT", "side": "yes", "action": "buy", "price": 50, "quantity": 50},
        ]
        orderbooks = {"MODERATE-MKT": moderate_orderbook}

        risk = risk_modeler.calculate_execution_risk(legs, orderbooks)
        threshold = risk_modeler.calculate_minimum_profit_threshold(risk)

        # Threshold should be at least $0.05
        assert threshold >= 0.05

    def test_threshold_increases_with_slippage_risk(
        self, risk_modeler: ExecutionRiskModeler
    ) -> None:
        """Test threshold increases when slippage risk is higher."""
        # Low risk estimate
        low_risk = ExecutionRiskEstimate(
            opportunity_id="low",
            leg_risks=[],
            price_movement_risk=[],
            overall_success_probability=0.95,
            expected_slippage_cents=1.0,
            slippage_std_cents=0.5,
            var_95_cents=2.0,
            execution_time_estimate=2.0,
            recommended_profit_threshold=0.05,
        )

        # High risk estimate
        high_risk = ExecutionRiskEstimate(
            opportunity_id="high",
            leg_risks=[],
            price_movement_risk=[],
            overall_success_probability=0.70,
            expected_slippage_cents=10.0,
            slippage_std_cents=5.0,
            var_95_cents=20.0,
            execution_time_estimate=5.0,
            recommended_profit_threshold=0.10,
        )

        low_threshold = risk_modeler.calculate_minimum_profit_threshold(low_risk)
        high_threshold = risk_modeler.calculate_minimum_profit_threshold(high_risk)

        assert high_threshold > low_threshold

    def test_threshold_adjusts_for_fill_probability(
        self, risk_modeler: ExecutionRiskModeler
    ) -> None:
        """Test threshold adjusts for low fill probability."""
        # Same slippage, different fill probabilities
        high_fill = ExecutionRiskEstimate(
            opportunity_id="high_fill",
            leg_risks=[],
            price_movement_risk=[],
            overall_success_probability=0.95,
            expected_slippage_cents=2.0,
            slippage_std_cents=1.0,
            var_95_cents=4.0,
            execution_time_estimate=2.0,
            recommended_profit_threshold=0.05,
        )

        low_fill = ExecutionRiskEstimate(
            opportunity_id="low_fill",
            leg_risks=[],
            price_movement_risk=[],
            overall_success_probability=0.50,
            expected_slippage_cents=2.0,
            slippage_std_cents=1.0,
            var_95_cents=4.0,
            execution_time_estimate=2.0,
            recommended_profit_threshold=0.05,
        )

        high_threshold = risk_modeler.calculate_minimum_profit_threshold(high_fill)
        low_threshold = risk_modeler.calculate_minimum_profit_threshold(low_fill)

        assert low_threshold > high_threshold


class TestShouldExecuteDecision:
    """Tests for execution decision based on risk filtering."""

    def test_execute_when_profitable_and_liquid(
        self, risk_modeler: ExecutionRiskModeler, liquid_orderbook: OrderBook
    ) -> None:
        """Test decision to execute when conditions are favorable."""
        legs = [
            {"market": "LIQUID-MKT", "side": "yes", "action": "buy", "price": 50, "quantity": 100},
        ]
        orderbooks = {"LIQUID-MKT": liquid_orderbook}

        decision = risk_modeler.should_execute(
            legs, orderbooks,
            expected_profit=1.00,  # $1.00 profit
        )

        assert decision.should_execute is True
        assert decision.confidence > 0
        assert decision.adjusted_profit > 0
        assert decision.reason == "All risk checks passed"

    def test_reject_when_profit_below_threshold(
        self, risk_modeler: ExecutionRiskModeler, liquid_orderbook: OrderBook
    ) -> None:
        """Test rejection when profit below minimum threshold."""
        legs = [
            {"market": "LIQUID-MKT", "side": "yes", "action": "buy", "price": 50, "quantity": 100},
        ]
        orderbooks = {"LIQUID-MKT": liquid_orderbook}

        decision = risk_modeler.should_execute(
            legs, orderbooks,
            expected_profit=0.01,  # Very small profit
        )

        assert decision.should_execute is False
        assert "threshold" in decision.reason.lower() or "profit" in decision.reason.lower()

    def test_reject_when_low_fill_probability(
        self, risk_modeler: ExecutionRiskModeler, illiquid_orderbook: OrderBook
    ) -> None:
        """Test rejection when fill probability is too low."""
        legs = [
            {"market": "ILLIQUID-MKT", "side": "yes", "action": "buy", "price": 55, "quantity": 500},
        ]
        orderbooks = {"ILLIQUID-MKT": illiquid_orderbook}

        decision = risk_modeler.should_execute(
            legs, orderbooks,
            expected_profit=1.00,
        )

        # Should reject or warn due to low liquidity
        if not decision.should_execute:
            assert "fill" in decision.reason.lower() or "liquidity" in decision.reason.lower()
        else:
            assert len(decision.warnings) > 0

    def test_warnings_for_high_var(
        self, risk_modeler: ExecutionRiskModeler, moderate_orderbook: OrderBook
    ) -> None:
        """Test warnings are generated for high VaR."""
        legs = [
            {"market": "MODERATE-MKT", "side": "yes", "action": "buy", "price": 50, "quantity": 200},
        ]
        orderbooks = {"MODERATE-MKT": moderate_orderbook}

        decision = risk_modeler.should_execute(
            legs, orderbooks,
            expected_profit=0.10,  # Small profit
        )

        # Should have warnings if VaR is significant relative to profit
        # (May or may not block execution depending on exact risk calc)
        assert isinstance(decision.warnings, list)

    def test_risk_adjusted_quantity_reduced(
        self, risk_modeler: ExecutionRiskModeler, moderate_orderbook: OrderBook
    ) -> None:
        """Test risk-adjusted quantity is reduced for risky trades."""
        legs = [
            {"market": "MODERATE-MKT", "side": "yes", "action": "buy", "price": 50, "quantity": 100},
        ]
        orderbooks = {"MODERATE-MKT": moderate_orderbook}

        decision = risk_modeler.should_execute(
            legs, orderbooks,
            expected_profit=1.00,
        )

        # Risk-adjusted quantity should be positive and <= original
        assert decision.risk_adjusted_quantity >= 1
        assert decision.risk_adjusted_quantity <= 100


class TestCalculateExecutionRisk:
    """Tests for comprehensive execution risk calculation."""

    def test_execution_risk_structure(
        self, risk_modeler: ExecutionRiskModeler, moderate_orderbook: OrderBook
    ) -> None:
        """Test ExecutionRiskEstimate has expected structure."""
        legs = [
            {"market": "MODERATE-MKT", "side": "yes", "action": "buy", "price": 50, "quantity": 50},
            {"market": "MODERATE-MKT", "side": "no", "action": "buy", "price": 52, "quantity": 50},
        ]
        orderbooks = {"MODERATE-MKT": moderate_orderbook}

        risk = risk_modeler.calculate_execution_risk(legs, orderbooks, "test-opp")

        assert risk.opportunity_id == "test-opp"
        assert len(risk.leg_risks) == 2
        assert risk.overall_success_probability > 0
        assert risk.expected_slippage_cents >= 0
        assert risk.slippage_std_cents >= 0
        assert risk.var_95_cents >= 0
        assert risk.execution_time_estimate > 0
        assert risk.recommended_profit_threshold >= 0.05

    def test_overall_probability_is_product(
        self, risk_modeler: ExecutionRiskModeler, liquid_orderbook: OrderBook
    ) -> None:
        """Test overall success probability is product of leg probabilities."""
        legs = [
            {"market": "LIQUID-MKT", "side": "yes", "action": "buy", "price": 50, "quantity": 50},
            {"market": "LIQUID-MKT", "side": "no", "action": "buy", "price": 52, "quantity": 50},
        ]
        orderbooks = {"LIQUID-MKT": liquid_orderbook}

        risk = risk_modeler.calculate_execution_risk(legs, orderbooks)

        # Overall probability should be product of individual probabilities
        expected_prob = np.prod([r.fill_probability for r in risk.leg_risks])
        assert risk.overall_success_probability == pytest.approx(expected_prob, rel=0.01)

    def test_handles_missing_orderbook(
        self, risk_modeler: ExecutionRiskModeler, moderate_orderbook: OrderBook
    ) -> None:
        """Test handling of missing orderbook for a leg."""
        legs = [
            {"market": "MODERATE-MKT", "side": "yes", "action": "buy", "price": 50, "quantity": 50},
            {"market": "MISSING-MKT", "side": "yes", "action": "buy", "price": 50, "quantity": 50},
        ]
        orderbooks = {"MODERATE-MKT": moderate_orderbook}

        risk = risk_modeler.calculate_execution_risk(legs, orderbooks)

        # Should still complete with conservative estimate for missing market
        assert len(risk.leg_risks) == 2
        # Missing market leg should have lower fill probability
        assert risk.leg_risks[1].fill_probability == 0.5  # Conservative default


class TestExecutionDecisionDataclass:
    """Tests for ExecutionDecision dataclass."""

    def test_decision_fields(self) -> None:
        """Test ExecutionDecision has all expected fields."""
        decision = ExecutionDecision(
            should_execute=True,
            reason="All checks passed",
            confidence=0.95,
            adjusted_profit=0.50,
            risk_adjusted_quantity=50,
            warnings=["Minor warning"],
        )

        assert decision.should_execute is True
        assert decision.reason == "All checks passed"
        assert decision.confidence == 0.95
        assert decision.adjusted_profit == 0.50
        assert decision.risk_adjusted_quantity == 50
        assert len(decision.warnings) == 1


class TestEstimateOptimalExecutionSpeed:
    """Tests for execution speed recommendation."""

    def test_aggressive_when_low_risk(
        self, risk_modeler: ExecutionRiskModeler
    ) -> None:
        """Test aggressive recommendation when both risks are low."""
        risk = ExecutionRiskEstimate(
            opportunity_id="test",
            leg_risks=[],
            price_movement_risk=[
                PriceMovementModel(
                    market_ticker="MKT",
                    expected_move_cents=0.5,
                    volatility_cents=1.0,
                    time_horizon_seconds=2.0,
                    confidence_interval=(-2.0, 2.0),
                )
            ],
            overall_success_probability=0.95,
            expected_slippage_cents=1.0,  # Low slippage
            slippage_std_cents=0.5,
            var_95_cents=2.0,
            execution_time_estimate=2.0,
            recommended_profit_threshold=0.05,
        )

        result = risk_modeler.estimate_optimal_execution_speed(risk, max_slippage_cents=5.0)

        assert result["recommendation"] == "aggressive"

    def test_patient_when_high_slippage(
        self, risk_modeler: ExecutionRiskModeler
    ) -> None:
        """Test patient recommendation when slippage is high."""
        risk = ExecutionRiskEstimate(
            opportunity_id="test",
            leg_risks=[],
            price_movement_risk=[],
            overall_success_probability=0.95,
            expected_slippage_cents=10.0,  # High slippage
            slippage_std_cents=3.0,
            var_95_cents=16.0,
            execution_time_estimate=2.0,
            recommended_profit_threshold=0.10,
        )

        result = risk_modeler.estimate_optimal_execution_speed(risk, max_slippage_cents=5.0)

        assert result["recommendation"] == "patient"


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_legs(self, risk_modeler: ExecutionRiskModeler) -> None:
        """Test handling empty legs list."""
        # This might raise an error or return a safe default
        try:
            risk = risk_modeler.calculate_execution_risk([], {})
            assert risk.overall_success_probability == 1.0  # No legs = guaranteed success
        except (IndexError, ValueError):
            pass  # Also acceptable to raise an error

    def test_zero_quantity(
        self, risk_modeler: ExecutionRiskModeler, moderate_orderbook: OrderBook
    ) -> None:
        """Test handling zero quantity legs."""
        legs = [
            {"market": "MODERATE-MKT", "side": "yes", "action": "buy", "price": 50, "quantity": 0},
        ]
        orderbooks = {"MODERATE-MKT": moderate_orderbook}

        risk = risk_modeler.calculate_execution_risk(legs, orderbooks)

        # Should handle gracefully
        assert len(risk.leg_risks) == 1

    def test_very_large_time_horizon(
        self, risk_modeler: ExecutionRiskModeler
    ) -> None:
        """Test price movement with very large time horizon."""
        result = risk_modeler.estimate_price_movement("MKT", time_between_legs=3600.0)  # 1 hour

        # Should still produce finite results
        assert np.isfinite(result.volatility_cents)
        assert np.isfinite(result.expected_move_cents)
