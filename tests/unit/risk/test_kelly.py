"""Unit tests for Kelly criterion position sizing."""

import pytest

from src.risk.kelly import calculate_kelly_size, kelly_from_edge, KellyResult


class TestCalculateKellySize:
    """Tests for calculate_kelly_size function."""

    def test_positive_ev_yes_bet(self) -> None:
        """Test Kelly sizing for a positive EV YES bet."""
        # Model thinks 60% YES, market price is 50c
        result = calculate_kelly_size(
            model_probability=0.60,
            market_price_cents=50,
            direction="yes",
            bankroll=1000.0,
            kelly_fraction=0.25,
        )

        assert result.is_positive_ev is True
        assert result.edge == pytest.approx(0.10, abs=0.01)
        assert result.full_kelly_fraction > 0
        assert result.position_contracts > 0
        assert result.position_dollars > 0

    def test_positive_ev_no_bet(self) -> None:
        """Test Kelly sizing for a positive EV NO bet."""
        # Model thinks 40% YES (60% NO), market price is 50c (NO costs 50c)
        result = calculate_kelly_size(
            model_probability=0.40,
            market_price_cents=50,
            direction="no",
            bankroll=1000.0,
            kelly_fraction=0.25,
        )

        assert result.is_positive_ev is True
        assert result.position_contracts > 0

    def test_negative_ev_returns_zero(self) -> None:
        """Test that negative EV bets return zero position."""
        # Model thinks 40% YES, market price is 50c - NO edge for YES bet
        result = calculate_kelly_size(
            model_probability=0.40,
            market_price_cents=50,
            direction="yes",
            bankroll=1000.0,
            kelly_fraction=0.25,
        )

        assert result.is_positive_ev is False
        assert result.position_contracts == 0
        assert result.position_dollars == 0.0

    def test_edge_cases_probability_bounds(self) -> None:
        """Test edge cases with probability bounds."""
        # Probability at boundary
        result = calculate_kelly_size(
            model_probability=0.0,  # Invalid
            market_price_cents=50,
            direction="yes",
            bankroll=1000.0,
        )
        assert result.is_positive_ev is False
        assert result.position_contracts == 0

        result = calculate_kelly_size(
            model_probability=1.0,  # Invalid
            market_price_cents=50,
            direction="yes",
            bankroll=1000.0,
        )
        assert result.is_positive_ev is False

    def test_invalid_price_returns_zero(self) -> None:
        """Test that invalid prices return zero position."""
        result = calculate_kelly_size(
            model_probability=0.60,
            market_price_cents=0,  # Invalid
            direction="yes",
            bankroll=1000.0,
        )
        assert result.position_contracts == 0

        result = calculate_kelly_size(
            model_probability=0.60,
            market_price_cents=100,  # Invalid
            direction="yes",
            bankroll=1000.0,
        )
        assert result.position_contracts == 0

    def test_zero_bankroll_returns_zero(self) -> None:
        """Test that zero bankroll returns zero position."""
        result = calculate_kelly_size(
            model_probability=0.60,
            market_price_cents=50,
            direction="yes",
            bankroll=0.0,
        )
        assert result.position_contracts == 0

    def test_fractional_kelly_reduces_position(self) -> None:
        """Test that fractional Kelly reduces position size."""
        full_kelly = calculate_kelly_size(
            model_probability=0.70,
            market_price_cents=50,
            direction="yes",
            bankroll=1000.0,
            kelly_fraction=1.0,
        )

        quarter_kelly = calculate_kelly_size(
            model_probability=0.70,
            market_price_cents=50,
            direction="yes",
            bankroll=1000.0,
            kelly_fraction=0.25,
        )

        assert quarter_kelly.position_contracts < full_kelly.position_contracts
        assert quarter_kelly.position_fraction < full_kelly.position_fraction

    def test_max_position_contracts_limit(self) -> None:
        """Test that max_position_contracts limits position size."""
        result = calculate_kelly_size(
            model_probability=0.90,
            market_price_cents=50,
            direction="yes",
            bankroll=10000.0,
            kelly_fraction=1.0,
            max_position_contracts=10,
        )

        assert result.position_contracts <= 10

    def test_max_position_dollars_limit(self) -> None:
        """Test that max_position_dollars limits position size."""
        result = calculate_kelly_size(
            model_probability=0.90,
            market_price_cents=50,
            direction="yes",
            bankroll=10000.0,
            kelly_fraction=1.0,
            max_position_dollars=100.0,
        )

        assert result.position_dollars <= 100.0

    def test_high_edge_high_kelly(self) -> None:
        """Test that higher edge results in higher Kelly fraction."""
        low_edge = calculate_kelly_size(
            model_probability=0.55,
            market_price_cents=50,
            direction="yes",
            bankroll=1000.0,
        )

        high_edge = calculate_kelly_size(
            model_probability=0.80,
            market_price_cents=50,
            direction="yes",
            bankroll=1000.0,
        )

        assert high_edge.full_kelly_fraction > low_edge.full_kelly_fraction

    def test_expected_value_calculation(self) -> None:
        """Test expected value calculation is correct."""
        result = calculate_kelly_size(
            model_probability=0.60,
            market_price_cents=50,
            direction="yes",
            bankroll=1000.0,
        )

        # EV should be positive for this setup
        assert result.expected_value > 0


class TestKellyFromEdge:
    """Tests for kelly_from_edge shortcut function."""

    def test_basic_edge_conversion(self) -> None:
        """Test converting edge directly to Kelly size."""
        result = kelly_from_edge(
            edge=0.10,  # 10% edge
            market_price_cents=50,
            direction="yes",
            bankroll=1000.0,
            kelly_fraction=0.25,
        )

        assert result.is_positive_ev is True
        assert result.position_contracts > 0

    def test_confidence_scaling(self) -> None:
        """Test that confidence scales down the position."""
        full_confidence = kelly_from_edge(
            edge=0.10,
            market_price_cents=50,
            direction="yes",
            bankroll=1000.0,
            confidence=1.0,
            kelly_fraction=0.25,
        )

        half_confidence = kelly_from_edge(
            edge=0.10,
            market_price_cents=50,
            direction="yes",
            bankroll=1000.0,
            confidence=0.5,
            kelly_fraction=0.25,
        )

        assert half_confidence.position_contracts < full_confidence.position_contracts

    def test_negative_edge_no_bet(self) -> None:
        """Test that negative edge returns no position."""
        result = kelly_from_edge(
            edge=-0.10,  # -10% edge (wrong direction)
            market_price_cents=50,
            direction="yes",
            bankroll=1000.0,
        )

        assert result.is_positive_ev is False
        assert result.position_contracts == 0

    def test_max_position_limit(self) -> None:
        """Test max position dollar limit."""
        result = kelly_from_edge(
            edge=0.20,
            market_price_cents=50,
            direction="yes",
            bankroll=10000.0,
            kelly_fraction=1.0,
            max_position_dollars=50.0,
        )

        assert result.position_dollars <= 50.0


class TestKellyResult:
    """Tests for KellyResult dataclass."""

    def test_result_fields(self) -> None:
        """Test that KellyResult has all expected fields."""
        result = KellyResult(
            full_kelly_fraction=0.20,
            position_fraction=0.05,
            position_dollars=50.0,
            position_contracts=100,
            edge=0.10,
            expected_value=0.20,
            is_positive_ev=True,
        )

        assert result.full_kelly_fraction == 0.20
        assert result.position_fraction == 0.05
        assert result.position_dollars == 50.0
        assert result.position_contracts == 100
        assert result.edge == 0.10
        assert result.expected_value == 0.20
        assert result.is_positive_ev is True
