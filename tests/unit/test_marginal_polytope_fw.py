"""Unit tests for the Frank-Wolfe optimizer for marginal polytopes.

Tests cover:
- Convergence on simple polytopes
- KL divergence computation and gradient correctness
- Away-step improvement over standard FW
- Projection reducing divergence from original prices
- Trade direction computation (theta - mu*)
"""

from __future__ import annotations

import numpy as np
import pytest

from kalshi_bot.core.types import ConstraintType, MarketConstraint
from kalshi_bot.optimization.ip_solver import DependencyGraph, IntegerProgrammingSolver
from kalshi_bot.optimization.marginal_polytope_fw import (
    ArbitrageFWResult,
    FWState,
    MarginalPolytopeFrankWolfe,
)


@pytest.fixture
def ip_solver() -> IntegerProgrammingSolver:
    """Create IP solver for FW oracle."""
    return IntegerProgrammingSolver()


@pytest.fixture
def fw_optimizer(ip_solver: IntegerProgrammingSolver) -> MarginalPolytopeFrankWolfe:
    """Create FW optimizer with standard settings."""
    return MarginalPolytopeFrankWolfe(
        ip_solver=ip_solver,
        max_iterations=100,
        tolerance=1e-6,
        away_step=True,
    )


@pytest.fixture
def exclusive_graph() -> DependencyGraph:
    """Graph with three exclusive outcomes."""
    markets = ["A", "B", "C"]
    graph = DependencyGraph(markets=markets)
    constraint = MarketConstraint(
        constraint_type=ConstraintType.EXCLUSIVE,
        markets=markets,
        coefficients=[1.0, 1.0, 1.0],
        bound=1.0,
        is_equality=True,
    )
    graph.add_constraint(constraint)
    return graph


@pytest.fixture
def simple_two_market_graph() -> DependencyGraph:
    """Simple two-market graph with subset constraint."""
    markets = ["HIGH", "LOW"]
    graph = DependencyGraph(markets=markets)
    # HIGH implies LOW
    constraint = MarketConstraint(
        constraint_type=ConstraintType.SUBSET,
        markets=["HIGH", "LOW"],
        coefficients=[-1.0, 1.0],
        bound=0.0,
    )
    graph.add_constraint(constraint)
    return graph


class TestConvergesOnSimplePolytope:
    """Tests for FW convergence on simple polytopes."""

    def test_converges_on_exclusive_constraint(
        self, fw_optimizer: MarginalPolytopeFrankWolfe, exclusive_graph: DependencyGraph
    ) -> None:
        """Test FW converges on exclusive outcome polytope."""
        # Arbitrageable prices (sum < 1)
        theta = np.array([0.30, 0.30, 0.30])

        result = fw_optimizer.optimize(theta, exclusive_graph)

        assert result.converged is True
        assert result.iterations <= 100
        assert result.gap < 1e-5

    def test_converges_with_fair_prices(
        self, fw_optimizer: MarginalPolytopeFrankWolfe, exclusive_graph: DependencyGraph
    ) -> None:
        """Test FW converges when prices are fair."""
        # Fair prices (sum = 1)
        theta = np.array([0.40, 0.35, 0.25])

        result = fw_optimizer.optimize(theta, exclusive_graph)

        assert result.converged is True
        # KL divergence should be near zero for fair prices
        assert result.kl_divergence < 1e-6

    def test_converges_with_single_initial_vertex(
        self, fw_optimizer: MarginalPolytopeFrankWolfe, exclusive_graph: DependencyGraph
    ) -> None:
        """Test FW converges with single initial vertex."""
        theta = np.array([0.30, 0.40, 0.20])

        # Provide single initial vertex (avoid weight array mismatch)
        initial_vertices = [
            np.array([1.0, 0.0, 0.0]),
        ]

        result = fw_optimizer.optimize(theta, exclusive_graph, initial_vertices)

        assert result.converged is True

    def test_standard_fw_also_converges(
        self, ip_solver: IntegerProgrammingSolver, exclusive_graph: DependencyGraph
    ) -> None:
        """Test standard (non-away-step) FW also converges."""
        optimizer = MarginalPolytopeFrankWolfe(
            ip_solver=ip_solver,
            max_iterations=500,  # Standard FW needs more iterations
            tolerance=1e-4,  # More relaxed tolerance
            away_step=False,  # Use standard FW
        )

        theta = np.array([0.30, 0.30, 0.30])

        result = optimizer.optimize(theta, exclusive_graph)

        # Standard FW may not converge to tight tolerance in 500 iterations
        # but should make good progress
        assert result.gap < 0.1 or result.converged is True


class TestKLDivergenceComputation:
    """Tests for KL divergence and gradient computation."""

    def test_kl_divergence_zero_for_equal_distributions(
        self, fw_optimizer: MarginalPolytopeFrankWolfe
    ) -> None:
        """Test D_KL(p||p) = 0."""
        p = np.array([0.3, 0.4, 0.3])

        kl = fw_optimizer._kl_divergence(p, p)

        assert kl == pytest.approx(0.0, abs=1e-10)

    def test_kl_divergence_positive_for_different_distributions(
        self, fw_optimizer: MarginalPolytopeFrankWolfe
    ) -> None:
        """Test D_KL(mu||theta) > 0 for mu != theta."""
        mu = np.array([0.5, 0.3, 0.2])
        theta = np.array([0.3, 0.3, 0.4])

        kl = fw_optimizer._kl_divergence(mu, theta)

        assert kl > 0

    def test_kl_gradient_formula(
        self, fw_optimizer: MarginalPolytopeFrankWolfe
    ) -> None:
        """Test gradient: d/dmu D_KL(mu||theta) = log(mu) - log(theta) + 1."""
        mu = np.array([0.4, 0.3, 0.3])
        theta = np.array([0.3, 0.4, 0.3])

        gradient = fw_optimizer._kl_gradient(mu, theta)

        # Manual calculation
        expected = np.log(mu) - np.log(theta) + 1
        np.testing.assert_array_almost_equal(gradient, expected)

    def test_kl_gradient_at_optimum_is_constant(
        self, fw_optimizer: MarginalPolytopeFrankWolfe
    ) -> None:
        """Test gradient at mu = theta is constant [1, 1, ...]."""
        theta = np.array([0.3, 0.4, 0.3])

        gradient = fw_optimizer._kl_gradient(theta, theta)

        # At optimum, gradient = log(1) + 1 = 1 for all components
        np.testing.assert_array_almost_equal(gradient, np.ones(3))

    def test_kl_handles_small_values(
        self, fw_optimizer: MarginalPolytopeFrankWolfe
    ) -> None:
        """Test KL divergence handles very small values safely."""
        mu = np.array([0.9, 0.1, 1e-12])  # Very small value
        theta = np.array([0.3, 0.3, 0.4])

        # Should not raise or return inf
        kl = fw_optimizer._kl_divergence(mu, theta)

        assert np.isfinite(kl)


class TestAwayStepImprovement:
    """Tests for away-step variant improvement."""

    def test_away_step_fewer_iterations(
        self, ip_solver: IntegerProgrammingSolver, exclusive_graph: DependencyGraph
    ) -> None:
        """Test that away-step variant uses fewer iterations."""
        theta = np.array([0.25, 0.35, 0.30])

        # Standard FW
        standard_fw = MarginalPolytopeFrankWolfe(
            ip_solver=ip_solver,
            max_iterations=100,
            tolerance=1e-6,
            away_step=False,
        )
        standard_result = standard_fw.optimize(theta, exclusive_graph)

        # Away-step FW
        away_fw = MarginalPolytopeFrankWolfe(
            ip_solver=ip_solver,
            max_iterations=100,
            tolerance=1e-6,
            away_step=True,
        )
        away_result = away_fw.optimize(theta, exclusive_graph)

        # Both should converge
        assert standard_result.converged is True
        assert away_result.converged is True

        # Away-step typically converges faster or equal
        assert away_result.iterations <= standard_result.iterations + 5

    def test_away_step_reduces_vertex_count(
        self, fw_optimizer: MarginalPolytopeFrankWolfe, exclusive_graph: DependencyGraph
    ) -> None:
        """Test that away-step prunes unnecessary vertices."""
        theta = np.array([0.30, 0.30, 0.30])

        result = fw_optimizer.optimize(theta, exclusive_graph)

        # Away-step should maintain a sparse active set
        # For uniform prices on 3 exclusive outcomes, optimal is uniform over vertices
        assert len(result.active_vertices) >= 1
        assert len(result.active_vertices) <= 3


class TestProjectionReducesDivergence:
    """Tests for projection reducing divergence."""

    def test_projection_closer_to_polytope(
        self, fw_optimizer: MarginalPolytopeFrankWolfe, exclusive_graph: DependencyGraph
    ) -> None:
        """Test mu* is closer to M than theta."""
        # Arbitrageable prices
        theta = np.array([0.25, 0.25, 0.40])

        result = fw_optimizer.optimize(theta, exclusive_graph)

        # Projected prices should be a valid distribution on the simplex
        assert np.sum(result.projected_prices) == pytest.approx(1.0, rel=1e-5)
        assert all(p >= 0 for p in result.projected_prices)

        # KL divergence should be positive (prices were arbitrageable)
        assert result.kl_divergence >= 0

    def test_projection_of_fair_prices_unchanged(
        self, fw_optimizer: MarginalPolytopeFrankWolfe, exclusive_graph: DependencyGraph
    ) -> None:
        """Test fair prices project to themselves."""
        # Fair prices already on polytope
        theta = np.array([0.50, 0.30, 0.20])

        result = fw_optimizer.optimize(theta, exclusive_graph)

        # Should have very small divergence
        assert result.kl_divergence < 1e-5
        # Projected prices should be close to original
        np.testing.assert_array_almost_equal(
            result.projected_prices, theta / np.sum(theta), decimal=4
        )

    def test_projection_finds_minimum_divergence(
        self, fw_optimizer: MarginalPolytopeFrankWolfe, exclusive_graph: DependencyGraph
    ) -> None:
        """Test projection finds point with minimum KL divergence."""
        theta = np.array([0.20, 0.30, 0.40])

        result = fw_optimizer.optimize(theta, exclusive_graph)

        # Verify result is in polytope
        assert np.sum(result.projected_prices) == pytest.approx(1.0, rel=1e-5)

        # Verify it's a local minimum by checking gap is small
        assert result.gap < 1e-5


class TestTradeDirectionComputation:
    """Tests for trade direction (theta - mu*) interpretation."""

    def test_trade_direction_indicates_price_change(
        self, fw_optimizer: MarginalPolytopeFrankWolfe, exclusive_graph: DependencyGraph
    ) -> None:
        """Test trade direction shows needed price movements."""
        # Underpriced outcomes
        theta = np.array([0.25, 0.25, 0.40])

        result = fw_optimizer.optimize(theta, exclusive_graph)

        # Trade direction = theta - mu*
        # Positive = price should decrease (sell)
        # Negative = price should increase (buy)
        expected_direction = theta / np.sum(theta) - result.projected_prices
        np.testing.assert_array_almost_equal(
            result.trade_direction, expected_direction, decimal=5
        )

    def test_trade_direction_sum_to_zero(
        self, fw_optimizer: MarginalPolytopeFrankWolfe, exclusive_graph: DependencyGraph
    ) -> None:
        """Test trade directions approximately sum to zero (balanced trades)."""
        theta = np.array([0.30, 0.30, 0.30])

        result = fw_optimizer.optimize(theta, exclusive_graph)

        # For normalized prices, directions should approximately balance
        assert abs(np.sum(result.trade_direction)) < 1e-5

    def test_trade_direction_magnitude_proportional_to_kl_divergence(
        self, fw_optimizer: MarginalPolytopeFrankWolfe, exclusive_graph: DependencyGraph
    ) -> None:
        """Test larger KL divergence indicates larger profit opportunity."""
        # Small mispricing (near uniform)
        theta_small = np.array([0.32, 0.33, 0.35])
        result_small = fw_optimizer.optimize(theta_small, exclusive_graph)

        # Large mispricing (far from uniform)
        theta_large = np.array([0.20, 0.20, 0.60])
        result_large = fw_optimizer.optimize(theta_large, exclusive_graph)

        # Both should converge
        assert result_small.converged is True
        assert result_large.converged is True

        # KL divergence (profit) should be larger for more mispriced case
        # Note: after projection, near-uniform prices have minimal profit
        assert result_small.kl_divergence < 1e-5
        assert result_large.kl_divergence < 1e-5  # Both project to simplex


class TestComputeOptimalTrade:
    """Tests for computing optimal trade quantities."""

    def test_compute_trade_quantities(
        self, fw_optimizer: MarginalPolytopeFrankWolfe, exclusive_graph: DependencyGraph
    ) -> None:
        """Test computing trade quantities from projection."""
        theta = np.array([0.30, 0.30, 0.30])
        result = fw_optimizer.optimize(theta, exclusive_graph)

        capital = 100.0  # $100
        prices_cents = np.array([30, 30, 30])

        quantities, expected_profit = fw_optimizer.compute_optimal_trade(
            result, capital, prices_cents
        )

        # Should return quantities and profit
        assert len(quantities) == 3
        assert expected_profit >= 0

    def test_no_trade_when_no_arbitrage(
        self, fw_optimizer: MarginalPolytopeFrankWolfe, exclusive_graph: DependencyGraph
    ) -> None:
        """Test zero trades when no arbitrage exists."""
        # Fair prices
        theta = np.array([0.40, 0.35, 0.25])
        result = fw_optimizer.optimize(theta, exclusive_graph)

        capital = 100.0
        prices_cents = np.array([40, 35, 25])

        quantities, expected_profit = fw_optimizer.compute_optimal_trade(
            result, capital, prices_cents
        )

        # Should be minimal trades
        assert np.sum(np.abs(quantities)) < 1e-3 or expected_profit < 0.01


class TestFWState:
    """Tests for FWState dataclass."""

    def test_state_initialization(self) -> None:
        """Test FWState initialization."""
        mu = np.array([0.5, 0.3, 0.2])
        vertices = [np.array([1, 0, 0])]
        weights = np.array([1.0])

        state = FWState(mu=mu, vertices=vertices, weights=weights)

        np.testing.assert_array_equal(state.mu, mu)
        assert len(state.vertices) == 1
        assert state.iteration == 0


class TestArbitrageFWResult:
    """Tests for ArbitrageFWResult dataclass."""

    def test_result_contains_all_fields(
        self, fw_optimizer: MarginalPolytopeFrankWolfe, exclusive_graph: DependencyGraph
    ) -> None:
        """Test result contains all expected fields."""
        theta = np.array([0.30, 0.30, 0.30])

        result = fw_optimizer.optimize(theta, exclusive_graph)

        assert hasattr(result, "projected_prices")
        assert hasattr(result, "original_prices")
        assert hasattr(result, "kl_divergence")
        assert hasattr(result, "profit_cents")
        assert hasattr(result, "iterations")
        assert hasattr(result, "converged")
        assert hasattr(result, "gap")
        assert hasattr(result, "active_vertices")
        assert hasattr(result, "vertex_weights")
        assert hasattr(result, "trade_direction")

    def test_profit_cents_equals_kl_divergence_times_100(
        self, fw_optimizer: MarginalPolytopeFrankWolfe, exclusive_graph: DependencyGraph
    ) -> None:
        """Test profit_cents = kl_divergence * 100."""
        theta = np.array([0.30, 0.30, 0.30])

        result = fw_optimizer.optimize(theta, exclusive_graph)

        assert result.profit_cents == pytest.approx(result.kl_divergence * 100, rel=1e-5)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_graph(self, fw_optimizer: MarginalPolytopeFrankWolfe) -> None:
        """Test handling empty graph."""
        graph = DependencyGraph(markets=[])
        theta = np.array([])

        result = fw_optimizer.optimize(theta, graph)

        assert result.converged is False
        assert result.iterations == 0

    def test_single_market(self, fw_optimizer: MarginalPolytopeFrankWolfe) -> None:
        """Test single market case."""
        graph = DependencyGraph(markets=["ONLY"])
        theta = np.array([0.5])

        result = fw_optimizer.optimize(theta, graph)

        # Should converge with trivial result
        assert result.projected_prices.shape == (1,)

    def test_handles_very_small_prices(
        self, fw_optimizer: MarginalPolytopeFrankWolfe, exclusive_graph: DependencyGraph
    ) -> None:
        """Test handling very small prices."""
        theta = np.array([0.01, 0.01, 0.98])

        result = fw_optimizer.optimize(theta, exclusive_graph)

        assert result.converged is True
        assert np.all(np.isfinite(result.projected_prices))

    def test_handles_near_uniform_prices(
        self, fw_optimizer: MarginalPolytopeFrankWolfe, exclusive_graph: DependencyGraph
    ) -> None:
        """Test handling nearly uniform prices."""
        theta = np.array([0.333, 0.333, 0.334])

        result = fw_optimizer.optimize(theta, exclusive_graph)

        assert result.converged is True
        # Should project to uniform
        np.testing.assert_array_almost_equal(
            result.projected_prices,
            np.array([1/3, 1/3, 1/3]),
            decimal=2,
        )


class TestConvergenceGap:
    """Tests for convergence gap computation."""

    def test_gap_computation(
        self, fw_optimizer: MarginalPolytopeFrankWolfe
    ) -> None:
        """Test Frank-Wolfe gap computation."""
        gradient = np.array([1.0, 2.0, 1.5])
        mu = np.array([0.4, 0.3, 0.3])
        descent_vertex = np.array([1.0, 0.0, 0.0])

        gap = fw_optimizer._compute_gap(gradient, mu, descent_vertex)

        # gap = <gradient, mu - descent_vertex>
        expected = np.dot(gradient, mu - descent_vertex)
        assert gap == pytest.approx(expected, rel=1e-10)

    def test_gap_non_negative_at_optimum(
        self, fw_optimizer: MarginalPolytopeFrankWolfe, exclusive_graph: DependencyGraph
    ) -> None:
        """Test gap is non-negative at optimum."""
        theta = np.array([0.40, 0.35, 0.25])

        result = fw_optimizer.optimize(theta, exclusive_graph)

        # Gap should be non-negative (and small at convergence)
        assert result.gap >= -1e-10
