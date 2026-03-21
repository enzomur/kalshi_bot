"""Integer Programming solver for dependency-based arbitrage detection.

Based on: "Unravelling the Probabilistic Forest: Arbitrage in Prediction Markets"
(arXiv:2508.03474v1)

The key insight is that valid outcomes form a polytope Z = {z ∈ {0,1}^I : A^T z >= b},
and arbitrage exists iff min_{z∈Z} θ·z < 1 where θ are the market prices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import milp, LinearConstraint, Bounds

from kalshi_bot.core.types import ConstraintType, MarketConstraint
from kalshi_bot.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DependencyGraph:
    """
    Graph representing logical dependencies between markets.

    Nodes are markets, edges represent constraints like:
    - Subset: A implies B
    - Temporal: Earlier event implies later event
    - Magnitude: Higher threshold implies lower threshold
    """

    markets: list[str]
    constraints: list[MarketConstraint] = field(default_factory=list)
    market_to_index: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Build market to index mapping."""
        if not self.market_to_index:
            self.market_to_index = {m: i for i, m in enumerate(self.markets)}

    def add_constraint(self, constraint: MarketConstraint) -> None:
        """Add a constraint to the graph."""
        self.constraints.append(constraint)

    def get_market_index(self, ticker: str) -> int:
        """Get index for a market ticker."""
        return self.market_to_index.get(ticker, -1)


@dataclass
class IPSolution:
    """Result of solving the Integer Program."""

    has_arbitrage: bool
    optimal_value: float  # min θ·z value
    optimal_vertex: NDArray[np.float64] | None  # z* that achieves minimum
    profit_per_dollar: float  # 1 - optimal_value (profit if < 0 means no arb)
    solve_time_ms: float = 0.0
    status: str = ""
    num_constraints: int = 0
    num_variables: int = 0


class IntegerProgrammingSolver:
    """
    Detects arbitrage using Integer Programming.

    Instead of enumerating 2^n possible outcomes, uses IP to efficiently
    find if any valid outcome assignment makes the total cost < 1.

    Key methods:
    - build_dependency_graph: Create graph from market relationships
    - express_as_linear_constraints: Convert graph to A^T z >= b form
    - solve_arbitrage_detection: Check if arbitrage exists
    - find_violating_vertex: Frank-Wolfe oracle for Bregman projection
    """

    def __init__(
        self,
        solver: str = "scipy",
        timeout_seconds: float = 10.0,
    ) -> None:
        """
        Initialize solver.

        Args:
            solver: Backend to use ("scipy" or "pulp")
            timeout_seconds: Maximum solve time
        """
        self.solver = solver
        self.timeout_seconds = timeout_seconds

    def build_dependency_graph(
        self,
        markets: list[str],
        relationships: list[dict[str, Any]],
    ) -> DependencyGraph:
        """
        Build dependency graph from market relationships.

        Args:
            markets: List of market tickers
            relationships: List of relationship definitions with:
                - type: ConstraintType
                - markets: List of involved tickers
                - description: Human-readable description

        Returns:
            DependencyGraph with all constraints
        """
        graph = DependencyGraph(markets=markets)

        for rel in relationships:
            constraint = self._relationship_to_constraint(rel, graph)
            if constraint:
                graph.add_constraint(constraint)

        logger.debug(
            f"Built dependency graph: {len(markets)} markets, "
            f"{len(graph.constraints)} constraints"
        )

        return graph

    def _relationship_to_constraint(
        self,
        relationship: dict[str, Any],
        graph: DependencyGraph,
    ) -> MarketConstraint | None:
        """Convert a relationship definition to a MarketConstraint."""
        rel_type = relationship.get("type")
        rel_markets = relationship.get("markets", [])
        description = relationship.get("description", "")

        if not rel_markets:
            return None

        if rel_type == ConstraintType.SUBSET or rel_type == "subset":
            # A implies B: z_A - z_B <= 0, equivalently -z_A + z_B >= 0
            if len(rel_markets) != 2:
                return None
            return MarketConstraint(
                constraint_type=ConstraintType.SUBSET,
                markets=rel_markets,
                coefficients=[-1.0, 1.0],  # -z_A + z_B >= 0
                bound=0.0,
                description=description or f"{rel_markets[0]} implies {rel_markets[1]}",
            )

        elif rel_type == ConstraintType.EXCLUSIVE or rel_type == "exclusive":
            # Exactly one: sum(z_i) = 1
            n = len(rel_markets)
            return MarketConstraint(
                constraint_type=ConstraintType.EXCLUSIVE,
                markets=rel_markets,
                coefficients=[1.0] * n,
                bound=1.0,
                is_equality=True,
                description=description or f"Exactly one of {rel_markets}",
            )

        elif rel_type == ConstraintType.COVERING or rel_type == "covering":
            # At least one: sum(z_i) >= 1
            n = len(rel_markets)
            return MarketConstraint(
                constraint_type=ConstraintType.COVERING,
                markets=rel_markets,
                coefficients=[1.0] * n,
                bound=1.0,
                description=description or f"At least one of {rel_markets}",
            )

        elif rel_type == ConstraintType.TEMPORAL or rel_type == "temporal":
            # Earlier implies later: z_earlier - z_later <= 0
            if len(rel_markets) != 2:
                return None
            return MarketConstraint(
                constraint_type=ConstraintType.TEMPORAL,
                markets=rel_markets,
                coefficients=[-1.0, 1.0],
                bound=0.0,
                description=description or f"{rel_markets[0]} (earlier) implies {rel_markets[1]} (later)",
            )

        elif rel_type == ConstraintType.MAGNITUDE or rel_type == "magnitude":
            # Higher threshold implies lower: z_high - z_low <= 0
            if len(rel_markets) != 2:
                return None
            return MarketConstraint(
                constraint_type=ConstraintType.MAGNITUDE,
                markets=rel_markets,
                coefficients=[-1.0, 1.0],
                bound=0.0,
                description=description or f"{rel_markets[0]} (high) implies {rel_markets[1]} (low)",
            )

        return None

    def express_as_linear_constraints(
        self,
        graph: DependencyGraph,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """
        Express dependency graph as linear constraints.

        Converts to standard form: A_ub @ z <= b_ub, A_eq @ z = b_eq

        Args:
            graph: Dependency graph with constraints

        Returns:
            Tuple of (A_ub, b_ub, A_eq, b_eq) matrices
            For scipy.milp: inequality constraints A_ub @ x <= b_ub
        """
        n = len(graph.markets)

        inequality_rows = []
        inequality_bounds = []
        equality_rows = []
        equality_bounds = []

        for constraint in graph.constraints:
            row = np.zeros(n)

            for ticker, coef in zip(constraint.markets, constraint.coefficients):
                idx = graph.get_market_index(ticker)
                if idx >= 0:
                    row[idx] = coef

            if constraint.is_equality:
                equality_rows.append(row)
                equality_bounds.append(constraint.bound)
            else:
                # Convert >= to <= by negating
                inequality_rows.append(-row)
                inequality_bounds.append(-constraint.bound)

        A_ub = np.array(inequality_rows) if inequality_rows else np.zeros((0, n))
        b_ub = np.array(inequality_bounds) if inequality_bounds else np.zeros(0)
        A_eq = np.array(equality_rows) if equality_rows else np.zeros((0, n))
        b_eq = np.array(equality_bounds) if equality_bounds else np.zeros(0)

        return A_ub, b_ub, A_eq, b_eq

    def solve_arbitrage_detection(
        self,
        prices: NDArray[np.float64],
        graph: DependencyGraph,
    ) -> IPSolution:
        """
        Solve IP to detect if arbitrage exists.

        Solves: min_{z ∈ Z} θ·z
        Where Z = {z ∈ {0,1}^n : A^T z >= b}

        Arbitrage exists iff optimal value < 1.

        Args:
            prices: Market prices θ (normalized to sum to ~1 for probability)
            graph: Dependency graph defining valid outcomes

        Returns:
            IPSolution with arbitrage detection result
        """
        import time
        start_time = time.time()

        n = len(graph.markets)

        if n == 0:
            return IPSolution(
                has_arbitrage=False,
                optimal_value=1.0,
                optimal_vertex=None,
                profit_per_dollar=0.0,
                status="empty",
            )

        # Get constraint matrices
        A_ub, b_ub, A_eq, b_eq = self.express_as_linear_constraints(graph)

        # Build constraints for scipy.milp
        constraints = []

        if len(A_ub) > 0:
            constraints.append(LinearConstraint(A_ub, -np.inf, b_ub))

        if len(A_eq) > 0:
            constraints.append(LinearConstraint(A_eq, b_eq, b_eq))

        # Variable bounds: z ∈ {0, 1}
        bounds = Bounds(lb=np.zeros(n), ub=np.ones(n))

        # Integrality: all variables are binary
        integrality = np.ones(n)

        try:
            result = milp(
                c=prices,
                constraints=constraints if constraints else None,
                integrality=integrality,
                bounds=bounds,
            )

            solve_time = (time.time() - start_time) * 1000

            if result.success:
                optimal_value = result.fun
                optimal_vertex = result.x

                # Arbitrage exists if we can buy all outcomes for < $1
                has_arbitrage = optimal_value < 1.0 - 1e-6
                profit_per_dollar = 1.0 - optimal_value

                return IPSolution(
                    has_arbitrage=has_arbitrage,
                    optimal_value=optimal_value,
                    optimal_vertex=optimal_vertex,
                    profit_per_dollar=profit_per_dollar,
                    solve_time_ms=solve_time,
                    status="optimal",
                    num_constraints=len(A_ub) + len(A_eq),
                    num_variables=n,
                )
            else:
                logger.warning(f"IP solve failed: {result.message}")
                return IPSolution(
                    has_arbitrage=False,
                    optimal_value=1.0,
                    optimal_vertex=None,
                    profit_per_dollar=0.0,
                    solve_time_ms=solve_time,
                    status=f"failed: {result.message}",
                    num_constraints=len(A_ub) + len(A_eq),
                    num_variables=n,
                )

        except Exception as e:
            solve_time = (time.time() - start_time) * 1000
            logger.error(f"IP solver error: {e}")
            return IPSolution(
                has_arbitrage=False,
                optimal_value=1.0,
                optimal_vertex=None,
                profit_per_dollar=0.0,
                solve_time_ms=solve_time,
                status=f"error: {str(e)}",
            )

    def find_violating_vertex(
        self,
        gradient: NDArray[np.float64],
        graph: DependencyGraph,
    ) -> NDArray[np.float64] | None:
        """
        Find vertex minimizing linear objective over constraint polytope.

        This is the Frank-Wolfe "linear oracle" for Bregman projection.
        Solves: z* = argmin_{z ∈ Z} <gradient, z>

        Args:
            gradient: Gradient vector (typically ∇D_KL)
            graph: Dependency graph defining Z

        Returns:
            Optimal vertex z*, or None if infeasible
        """
        solution = self.solve_arbitrage_detection(gradient, graph)

        if solution.optimal_vertex is not None:
            # Round to binary (should already be binary from IP)
            vertex = np.round(solution.optimal_vertex)
            return vertex

        return None

    def enumerate_vertices(
        self,
        graph: DependencyGraph,
        max_vertices: int = 1000,
    ) -> list[NDArray[np.float64]]:
        """
        Enumerate vertices of the constraint polytope (for small problems).

        Uses randomized objective vectors to find diverse vertices.

        Args:
            graph: Dependency graph
            max_vertices: Maximum number of vertices to find

        Returns:
            List of unique vertices
        """
        vertices: list[NDArray[np.float64]] = []
        seen: set[tuple[int, ...]] = set()

        n = len(graph.markets)
        if n == 0:
            return vertices

        # Try random objective directions
        for _ in range(max_vertices * 2):
            c = np.random.randn(n)
            vertex = self.find_violating_vertex(c, graph)

            if vertex is not None:
                key = tuple(int(v) for v in vertex)
                if key not in seen:
                    seen.add(key)
                    vertices.append(vertex)

                    if len(vertices) >= max_vertices:
                        break

        logger.debug(f"Enumerated {len(vertices)} vertices")
        return vertices

    def validate_constraints(
        self,
        vertex: NDArray[np.float64],
        graph: DependencyGraph,
        tolerance: float = 1e-6,
    ) -> bool:
        """
        Check if a vertex satisfies all constraints.

        Args:
            vertex: Binary vector to check
            graph: Dependency graph with constraints
            tolerance: Numerical tolerance

        Returns:
            True if all constraints satisfied
        """
        for constraint in graph.constraints:
            value = 0.0
            for ticker, coef in zip(constraint.markets, constraint.coefficients):
                idx = graph.get_market_index(ticker)
                if idx >= 0:
                    value += coef * vertex[idx]

            if constraint.is_equality:
                if abs(value - constraint.bound) > tolerance:
                    return False
            else:
                # >= constraint
                if value < constraint.bound - tolerance:
                    return False

        return True
