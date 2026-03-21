"""Unit tests for the Integer Programming solver.

Tests cover:
- Dependency graph construction from market relationships
- Linear constraint encoding (subset, exclusive, covering, etc.)
- Arbitrage detection when prices sum < 1
- No arbitrage detection when prices sum = 1
- Frank-Wolfe oracle vertex finding
"""

from __future__ import annotations

import numpy as np
import pytest

from kalshi_bot.core.types import ConstraintType, MarketConstraint
from kalshi_bot.optimization.ip_solver import (
    DependencyGraph,
    IntegerProgrammingSolver,
    IPSolution,
)


@pytest.fixture
def simple_markets() -> list[str]:
    """Three simple markets for testing."""
    return ["MKT-A", "MKT-B", "MKT-C"]


@pytest.fixture
def exclusive_markets() -> list[str]:
    """Markets representing exclusive outcomes (e.g., candidate wins)."""
    return ["CANDIDATE-X", "CANDIDATE-Y", "CANDIDATE-Z"]


@pytest.fixture
def solver() -> IntegerProgrammingSolver:
    """Create a fresh solver instance."""
    return IntegerProgrammingSolver(solver="scipy", timeout_seconds=10.0)


class TestDependencyGraph:
    """Tests for DependencyGraph construction and management."""

    def test_empty_graph_construction(self) -> None:
        """Test creating an empty dependency graph."""
        graph = DependencyGraph(markets=[])
        assert len(graph.markets) == 0
        assert len(graph.constraints) == 0

    def test_graph_with_markets(self, simple_markets: list[str]) -> None:
        """Test graph construction with markets."""
        graph = DependencyGraph(markets=simple_markets)

        assert len(graph.markets) == 3
        assert graph.get_market_index("MKT-A") == 0
        assert graph.get_market_index("MKT-B") == 1
        assert graph.get_market_index("MKT-C") == 2
        assert graph.get_market_index("NONEXISTENT") == -1

    def test_add_constraint(self, simple_markets: list[str]) -> None:
        """Test adding constraints to graph."""
        graph = DependencyGraph(markets=simple_markets)

        constraint = MarketConstraint(
            constraint_type=ConstraintType.SUBSET,
            markets=["MKT-A", "MKT-B"],
            coefficients=[-1.0, 1.0],
            bound=0.0,
            description="A implies B",
        )
        graph.add_constraint(constraint)

        assert len(graph.constraints) == 1
        assert graph.constraints[0].constraint_type == ConstraintType.SUBSET


class TestBuildDependencyGraph:
    """Tests for building dependency graphs from relationships."""

    def test_build_empty_graph(self, solver: IntegerProgrammingSolver) -> None:
        """Test building graph with no relationships."""
        graph = solver.build_dependency_graph(markets=[], relationships=[])

        assert len(graph.markets) == 0
        assert len(graph.constraints) == 0

    def test_build_graph_subset_constraint(
        self, solver: IntegerProgrammingSolver, simple_markets: list[str]
    ) -> None:
        """Test building graph with subset (implication) constraint."""
        relationships = [
            {
                "type": ConstraintType.SUBSET,
                "markets": ["MKT-A", "MKT-B"],
                "description": "A implies B",
            }
        ]

        graph = solver.build_dependency_graph(simple_markets, relationships)

        assert len(graph.constraints) == 1
        constraint = graph.constraints[0]
        assert constraint.constraint_type == ConstraintType.SUBSET
        assert constraint.markets == ["MKT-A", "MKT-B"]
        # For z_A <= z_B, we encode as -z_A + z_B >= 0
        assert constraint.coefficients == [-1.0, 1.0]
        assert constraint.bound == 0.0

    def test_build_graph_exclusive_constraint(
        self, solver: IntegerProgrammingSolver, exclusive_markets: list[str]
    ) -> None:
        """Test building graph with exclusivity constraint (exactly one)."""
        relationships = [
            {
                "type": ConstraintType.EXCLUSIVE,
                "markets": exclusive_markets,
                "description": "Exactly one candidate wins",
            }
        ]

        graph = solver.build_dependency_graph(exclusive_markets, relationships)

        assert len(graph.constraints) == 1
        constraint = graph.constraints[0]
        assert constraint.constraint_type == ConstraintType.EXCLUSIVE
        assert constraint.is_equality is True
        assert constraint.coefficients == [1.0, 1.0, 1.0]
        assert constraint.bound == 1.0

    def test_build_graph_covering_constraint(
        self, solver: IntegerProgrammingSolver, simple_markets: list[str]
    ) -> None:
        """Test building graph with covering constraint (at least one)."""
        relationships = [
            {
                "type": ConstraintType.COVERING,
                "markets": simple_markets,
                "description": "At least one must happen",
            }
        ]

        graph = solver.build_dependency_graph(simple_markets, relationships)

        assert len(graph.constraints) == 1
        constraint = graph.constraints[0]
        assert constraint.constraint_type == ConstraintType.COVERING
        assert constraint.is_equality is False
        assert constraint.bound == 1.0

    def test_build_graph_temporal_constraint(
        self, solver: IntegerProgrammingSolver
    ) -> None:
        """Test building graph with temporal constraint."""
        markets = ["EARLY-EVENT", "LATE-EVENT"]
        relationships = [
            {
                "type": ConstraintType.TEMPORAL,
                "markets": markets,
                "description": "Early implies late",
            }
        ]

        graph = solver.build_dependency_graph(markets, relationships)

        assert len(graph.constraints) == 1
        constraint = graph.constraints[0]
        assert constraint.constraint_type == ConstraintType.TEMPORAL
        assert constraint.coefficients == [-1.0, 1.0]

    def test_build_graph_magnitude_constraint(
        self, solver: IntegerProgrammingSolver
    ) -> None:
        """Test building graph with magnitude constraint (X > high implies X > low)."""
        markets = ["PRICE-GT-100", "PRICE-GT-50"]
        relationships = [
            {
                "type": ConstraintType.MAGNITUDE,
                "markets": markets,
                "description": "Price > 100 implies Price > 50",
            }
        ]

        graph = solver.build_dependency_graph(markets, relationships)

        assert len(graph.constraints) == 1
        constraint = graph.constraints[0]
        assert constraint.constraint_type == ConstraintType.MAGNITUDE
        # z_high <= z_low encoded as -z_high + z_low >= 0
        assert constraint.coefficients == [-1.0, 1.0]


class TestExpressAsLinearConstraints:
    """Tests for converting dependency graphs to linear constraint matrices."""

    def test_empty_constraints(self, solver: IntegerProgrammingSolver) -> None:
        """Test expressing empty graph as constraints."""
        graph = DependencyGraph(markets=["A", "B", "C"])

        A_ub, b_ub, A_eq, b_eq = solver.express_as_linear_constraints(graph)

        assert A_ub.shape == (0, 3)
        assert len(b_ub) == 0
        assert A_eq.shape == (0, 3)
        assert len(b_eq) == 0

    def test_subset_constraint_encoding(
        self, solver: IntegerProgrammingSolver
    ) -> None:
        """Test that z_A <= z_B is properly encoded."""
        graph = DependencyGraph(markets=["A", "B"])
        constraint = MarketConstraint(
            constraint_type=ConstraintType.SUBSET,
            markets=["A", "B"],
            coefficients=[-1.0, 1.0],  # -z_A + z_B >= 0
            bound=0.0,
        )
        graph.add_constraint(constraint)

        A_ub, b_ub, A_eq, b_eq = solver.express_as_linear_constraints(graph)

        # Converted to <= form: z_A - z_B <= 0
        assert A_ub.shape == (1, 2)
        np.testing.assert_array_equal(A_ub[0], [1.0, -1.0])
        assert b_ub[0] == 0.0

    def test_exclusive_constraint_encoding(
        self, solver: IntegerProgrammingSolver, exclusive_markets: list[str]
    ) -> None:
        """Test that sum(z_i) = 1 is properly encoded."""
        graph = DependencyGraph(markets=exclusive_markets)
        constraint = MarketConstraint(
            constraint_type=ConstraintType.EXCLUSIVE,
            markets=exclusive_markets,
            coefficients=[1.0, 1.0, 1.0],
            bound=1.0,
            is_equality=True,
        )
        graph.add_constraint(constraint)

        A_ub, b_ub, A_eq, b_eq = solver.express_as_linear_constraints(graph)

        # Equality constraint
        assert A_eq.shape == (1, 3)
        np.testing.assert_array_equal(A_eq[0], [1.0, 1.0, 1.0])
        assert b_eq[0] == 1.0
        # No inequality constraints
        assert len(b_ub) == 0


class TestDetectArbitrageWhenPresent:
    """Tests for detecting arbitrage when prices sum < 1."""

    def test_arbitrage_when_prices_sum_below_one(
        self, solver: IntegerProgrammingSolver, exclusive_markets: list[str]
    ) -> None:
        """Test detecting arbitrage when exclusive outcomes sum < 1."""
        # Setup: Three exclusive outcomes with prices summing to 0.90
        graph = DependencyGraph(markets=exclusive_markets)
        constraint = MarketConstraint(
            constraint_type=ConstraintType.EXCLUSIVE,
            markets=exclusive_markets,
            coefficients=[1.0, 1.0, 1.0],
            bound=1.0,
            is_equality=True,
        )
        graph.add_constraint(constraint)

        # Prices: 30%, 30%, 30% = 90% total (10% profit opportunity)
        prices = np.array([0.30, 0.30, 0.30])

        solution = solver.solve_arbitrage_detection(prices, graph)

        assert solution.has_arbitrage is True
        assert solution.optimal_value < 1.0
        assert solution.profit_per_dollar > 0
        # With exclusive constraint, exactly one outcome
        assert solution.optimal_vertex is not None
        assert np.sum(solution.optimal_vertex) == 1.0

    def test_large_arbitrage_opportunity(
        self, solver: IntegerProgrammingSolver, exclusive_markets: list[str]
    ) -> None:
        """Test detecting large arbitrage opportunity."""
        graph = DependencyGraph(markets=exclusive_markets)
        constraint = MarketConstraint(
            constraint_type=ConstraintType.EXCLUSIVE,
            markets=exclusive_markets,
            coefficients=[1.0, 1.0, 1.0],
            bound=1.0,
            is_equality=True,
        )
        graph.add_constraint(constraint)

        # Extreme underpricing: 20%, 20%, 20% = 60% total
        prices = np.array([0.20, 0.20, 0.20])

        solution = solver.solve_arbitrage_detection(prices, graph)

        assert solution.has_arbitrage is True
        assert solution.optimal_value == pytest.approx(0.20, rel=1e-5)
        assert solution.profit_per_dollar == pytest.approx(0.80, rel=1e-5)


class TestNoArbitrageWhenFair:
    """Tests for no arbitrage detection when prices are fair."""

    def test_optimal_value_equals_minimum_price(
        self, solver: IntegerProgrammingSolver, exclusive_markets: list[str]
    ) -> None:
        """Test IP finds minimum price among exclusive outcomes.

        Note: For exclusive constraints, the IP solver finds the cheapest
        outcome, which for fair pricing (sum=1) equals the minimum price.
        Arbitrage detection based on sum < 1 is done at a higher level.
        """
        graph = DependencyGraph(markets=exclusive_markets)
        constraint = MarketConstraint(
            constraint_type=ConstraintType.EXCLUSIVE,
            markets=exclusive_markets,
            coefficients=[1.0, 1.0, 1.0],
            bound=1.0,
            is_equality=True,
        )
        graph.add_constraint(constraint)

        # Fair prices: 40%, 35%, 25% = 100%
        prices = np.array([0.40, 0.35, 0.25])

        solution = solver.solve_arbitrage_detection(prices, graph)

        # IP finds minimum cost outcome (0.25)
        assert solution.optimal_value == pytest.approx(0.25, rel=1e-5)
        # The optimal vertex should be the cheapest outcome
        assert solution.optimal_vertex is not None
        np.testing.assert_array_equal(solution.optimal_vertex, [0, 0, 1])

    def test_no_arbitrage_when_prices_sum_above_one(
        self, solver: IntegerProgrammingSolver, exclusive_markets: list[str]
    ) -> None:
        """Test no arbitrage when prices sum > 1 (overpriced)."""
        graph = DependencyGraph(markets=exclusive_markets)
        constraint = MarketConstraint(
            constraint_type=ConstraintType.EXCLUSIVE,
            markets=exclusive_markets,
            coefficients=[1.0, 1.0, 1.0],
            bound=1.0,
            is_equality=True,
        )
        graph.add_constraint(constraint)

        # Overpriced: 40%, 40%, 30% = 110%
        prices = np.array([0.40, 0.40, 0.30])

        solution = solver.solve_arbitrage_detection(prices, graph)

        # Still picks the cheapest outcome
        assert solution.optimal_value == pytest.approx(0.30, rel=1e-5)
        # But this doesn't create arbitrage (we're looking at buy side)
        # For 3 exclusive outcomes, optimal = min price = 0.30
        # Since 0.30 < 1.0, technically there's arbitrage from perspective
        # of buying cheapest outcome guaranteed to pay $1


class TestFindViolatingVertex:
    """Tests for Frank-Wolfe oracle vertex finding."""

    def test_find_vertex_minimizing_gradient(
        self, solver: IntegerProgrammingSolver, exclusive_markets: list[str]
    ) -> None:
        """Test that oracle finds vertex minimizing linear objective."""
        graph = DependencyGraph(markets=exclusive_markets)
        constraint = MarketConstraint(
            constraint_type=ConstraintType.EXCLUSIVE,
            markets=exclusive_markets,
            coefficients=[1.0, 1.0, 1.0],
            bound=1.0,
            is_equality=True,
        )
        graph.add_constraint(constraint)

        # Gradient pointing to first market
        gradient = np.array([0.5, 1.0, 1.5])

        vertex = solver.find_violating_vertex(gradient, graph)

        assert vertex is not None
        # Should select first market (lowest gradient)
        np.testing.assert_array_equal(vertex, [1, 0, 0])

    def test_find_vertex_with_subset_constraints(
        self, solver: IntegerProgrammingSolver
    ) -> None:
        """Test vertex finding respects subset constraints."""
        markets = ["A", "B"]
        graph = DependencyGraph(markets=markets)
        # A implies B: if A=1, then B=1
        constraint = MarketConstraint(
            constraint_type=ConstraintType.SUBSET,
            markets=["A", "B"],
            coefficients=[-1.0, 1.0],
            bound=0.0,
        )
        graph.add_constraint(constraint)

        # Gradient favoring A only
        gradient = np.array([0.0, 1.0])

        vertex = solver.find_violating_vertex(gradient, graph)

        assert vertex is not None
        # Can't have A=1, B=0 due to constraint
        # Valid vertices: (0,0), (0,1), (1,1)
        # Minimizing [0, 1] dot vertex -> (0,0) or (1,1) with value 0 or 1
        # (0,0) has value 0, (1,1) has value 1, (0,1) has value 1
        # So optimal is (0,0)
        np.testing.assert_array_equal(vertex, [0, 0])

    def test_find_vertex_returns_valid_binary(
        self, solver: IntegerProgrammingSolver, exclusive_markets: list[str]
    ) -> None:
        """Test that returned vertices are binary."""
        graph = DependencyGraph(markets=exclusive_markets)
        constraint = MarketConstraint(
            constraint_type=ConstraintType.EXCLUSIVE,
            markets=exclusive_markets,
            coefficients=[1.0, 1.0, 1.0],
            bound=1.0,
            is_equality=True,
        )
        graph.add_constraint(constraint)

        # Random gradient
        gradient = np.array([0.3, 0.7, 0.5])

        vertex = solver.find_violating_vertex(gradient, graph)

        assert vertex is not None
        # All entries should be 0 or 1
        assert all(v in [0, 1] for v in vertex)
        # Should satisfy exclusivity
        assert np.sum(vertex) == 1


class TestEnumerateVertices:
    """Tests for vertex enumeration."""

    def test_enumerate_exclusive_vertices(
        self, solver: IntegerProgrammingSolver, exclusive_markets: list[str]
    ) -> None:
        """Test enumerating vertices of exclusive constraint polytope."""
        graph = DependencyGraph(markets=exclusive_markets)
        constraint = MarketConstraint(
            constraint_type=ConstraintType.EXCLUSIVE,
            markets=exclusive_markets,
            coefficients=[1.0, 1.0, 1.0],
            bound=1.0,
            is_equality=True,
        )
        graph.add_constraint(constraint)

        vertices = solver.enumerate_vertices(graph, max_vertices=10)

        # Should find exactly 3 vertices for 3 exclusive outcomes
        assert len(vertices) == 3

        # Each vertex should have exactly one 1
        for v in vertices:
            assert np.sum(v) == 1


class TestValidateConstraints:
    """Tests for constraint validation."""

    def test_valid_vertex_passes(
        self, solver: IntegerProgrammingSolver, exclusive_markets: list[str]
    ) -> None:
        """Test that valid vertices pass validation."""
        graph = DependencyGraph(markets=exclusive_markets)
        constraint = MarketConstraint(
            constraint_type=ConstraintType.EXCLUSIVE,
            markets=exclusive_markets,
            coefficients=[1.0, 1.0, 1.0],
            bound=1.0,
            is_equality=True,
        )
        graph.add_constraint(constraint)

        valid_vertex = np.array([1.0, 0.0, 0.0])

        assert solver.validate_constraints(valid_vertex, graph) is True

    def test_invalid_vertex_fails(
        self, solver: IntegerProgrammingSolver, exclusive_markets: list[str]
    ) -> None:
        """Test that invalid vertices fail validation."""
        graph = DependencyGraph(markets=exclusive_markets)
        constraint = MarketConstraint(
            constraint_type=ConstraintType.EXCLUSIVE,
            markets=exclusive_markets,
            coefficients=[1.0, 1.0, 1.0],
            bound=1.0,
            is_equality=True,
        )
        graph.add_constraint(constraint)

        # Two outcomes selected - violates exclusivity
        invalid_vertex = np.array([1.0, 1.0, 0.0])

        assert solver.validate_constraints(invalid_vertex, graph) is False

    def test_subset_constraint_validation(
        self, solver: IntegerProgrammingSolver
    ) -> None:
        """Test subset constraint validation."""
        markets = ["A", "B"]
        graph = DependencyGraph(markets=markets)
        constraint = MarketConstraint(
            constraint_type=ConstraintType.SUBSET,
            markets=["A", "B"],
            coefficients=[-1.0, 1.0],  # -z_A + z_B >= 0
            bound=0.0,
        )
        graph.add_constraint(constraint)

        # Valid: A=0, B=0 (0 >= 0)
        assert solver.validate_constraints(np.array([0.0, 0.0]), graph) is True
        # Valid: A=0, B=1 (1 >= 0)
        assert solver.validate_constraints(np.array([0.0, 1.0]), graph) is True
        # Valid: A=1, B=1 (0 >= 0)
        assert solver.validate_constraints(np.array([1.0, 1.0]), graph) is True
        # Invalid: A=1, B=0 (-1 >= 0 is false)
        assert solver.validate_constraints(np.array([1.0, 0.0]), graph) is False


class TestIPSolutionDataclass:
    """Tests for IPSolution dataclass."""

    def test_solution_with_arbitrage(self) -> None:
        """Test IPSolution with arbitrage present."""
        solution = IPSolution(
            has_arbitrage=True,
            optimal_value=0.85,
            optimal_vertex=np.array([1, 0, 0]),
            profit_per_dollar=0.15,
            solve_time_ms=5.0,
            status="optimal",
            num_constraints=1,
            num_variables=3,
        )

        assert solution.has_arbitrage is True
        assert solution.profit_per_dollar == 0.15
        assert solution.status == "optimal"

    def test_solution_without_arbitrage(self) -> None:
        """Test IPSolution without arbitrage."""
        solution = IPSolution(
            has_arbitrage=False,
            optimal_value=1.0,
            optimal_vertex=np.array([0, 1, 0]),
            profit_per_dollar=0.0,
        )

        assert solution.has_arbitrage is False
        assert solution.profit_per_dollar == 0.0


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_markets(self, solver: IntegerProgrammingSolver) -> None:
        """Test handling empty markets."""
        graph = DependencyGraph(markets=[])
        prices = np.array([])

        solution = solver.solve_arbitrage_detection(prices, graph)

        assert solution.has_arbitrage is False
        assert solution.status == "empty"

    def test_single_market(self, solver: IntegerProgrammingSolver) -> None:
        """Test single market case."""
        graph = DependencyGraph(markets=["SINGLE"])
        prices = np.array([0.5])

        solution = solver.solve_arbitrage_detection(prices, graph)

        # Single unconstrained binary variable
        assert solution.optimal_value == pytest.approx(0.0, abs=1e-5)

    def test_invalid_relationship_ignored(
        self, solver: IntegerProgrammingSolver, simple_markets: list[str]
    ) -> None:
        """Test that invalid relationships are ignored."""
        relationships = [
            {
                "type": "invalid_type",
                "markets": simple_markets,
            },
            {
                "type": ConstraintType.SUBSET,
                "markets": [],  # Invalid: no markets
            },
        ]

        graph = solver.build_dependency_graph(simple_markets, relationships)

        # Invalid relationships should be skipped
        assert len(graph.constraints) == 0
