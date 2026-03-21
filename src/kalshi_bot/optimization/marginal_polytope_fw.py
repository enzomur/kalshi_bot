"""Frank-Wolfe optimizer for Bregman projection onto arbitrary polytopes.

Based on: "Unravelling the Probabilistic Forest: Arbitrage in Prediction Markets"
(arXiv:2508.03474v1)

This extends the simplex-based Frank-Wolfe to handle arbitrary polytopes
defined by linear constraints (the marginal polytope M). The key difference
is that the linear minimization oracle uses Integer Programming instead of
a closed-form vertex selection.

The projection finds μ* = argmin_{μ∈M} D_KL(μ || θ) where:
- θ is the current (arbitrageable) price vector
- M is the marginal polytope of valid probability distributions
- D_KL is the Kullback-Leibler divergence

The optimal profit equals D_KL(μ* || θ).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from kalshi_bot.optimization.ip_solver import DependencyGraph, IntegerProgrammingSolver
from kalshi_bot.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ArbitrageFWResult:
    """Result of Frank-Wolfe optimization for arbitrage."""

    projected_prices: NDArray[np.float64]  # μ* - arbitrage-free projection
    original_prices: NDArray[np.float64]  # θ - original prices
    kl_divergence: float  # D_KL(μ* || θ) = profit
    profit_cents: float  # Profit in cents per dollar invested
    iterations: int
    converged: bool
    gap: float  # Frank-Wolfe duality gap
    active_vertices: list[NDArray[np.float64]]  # Vertices in convex combination
    vertex_weights: NDArray[np.float64]  # Weights for each vertex
    trade_direction: NDArray[np.float64]  # θ - μ* (positive = sell, negative = buy)


@dataclass
class FWState:
    """Internal state for Frank-Wolfe iterations."""

    mu: NDArray[np.float64]  # Current iterate
    vertices: list[NDArray[np.float64]]  # Active set of vertices
    weights: NDArray[np.float64]  # Convex combination weights
    iteration: int = 0


class MarginalPolytopeFrankWolfe:
    """
    Frank-Wolfe optimizer for projection onto marginal polytopes.

    Uses IP oracle for the linear minimization step, enabling projection
    onto arbitrary polytopes defined by linear constraints (not just simplex).

    The algorithm maintains a convex combination of vertices and iteratively
    adds new vertices that decrease the objective (KL divergence).
    """

    def __init__(
        self,
        ip_solver: IntegerProgrammingSolver | None = None,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        away_step: bool = True,
    ) -> None:
        """
        Initialize optimizer.

        Args:
            ip_solver: Integer programming solver for linear oracle
            max_iterations: Maximum Frank-Wolfe iterations
            tolerance: Convergence tolerance (duality gap)
            away_step: Use away-step variant for faster convergence
        """
        self.ip_solver = ip_solver or IntegerProgrammingSolver()
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.away_step = away_step

    def optimize(
        self,
        theta: NDArray[np.float64],
        graph: DependencyGraph,
        initial_vertices: list[NDArray[np.float64]] | None = None,
    ) -> ArbitrageFWResult:
        """
        Project prices onto marginal polytope using Frank-Wolfe.

        Solves: μ* = argmin_{μ∈M} D_KL(μ || θ)

        Args:
            theta: Current market prices (should be positive, sum ~1)
            graph: Dependency graph defining the marginal polytope
            initial_vertices: Optional initial vertices for warm start

        Returns:
            ArbitrageFWResult with projection and profit
        """
        n = len(theta)

        # Ensure theta is valid (positive, normalized)
        theta = np.maximum(theta, 1e-10)
        theta = theta / np.sum(theta)

        # Initialize with a feasible vertex
        if initial_vertices and len(initial_vertices) > 0:
            vertices = list(initial_vertices)
        else:
            # Find an initial feasible vertex using IP
            init_vertex = self._find_initial_vertex(graph)
            if init_vertex is None:
                logger.warning("Could not find initial feasible vertex")
                return self._empty_result(theta)
            vertices = [init_vertex]

        # Initial point is the first vertex
        weights = np.ones(1)
        mu = vertices[0].copy()

        state = FWState(mu=mu, vertices=vertices, weights=weights)

        gap = float('inf')
        for iteration in range(self.max_iterations):
            state.iteration = iteration

            # Compute gradient: ∇D_KL(μ || θ) = log(μ) - log(θ) + 1
            gradient = self._kl_gradient(state.mu, theta)

            # Find descent vertex using IP oracle
            descent_vertex = self._find_descent_vertex(gradient, graph)
            if descent_vertex is None:
                break

            # Compute Frank-Wolfe gap
            gap = self._compute_gap(gradient, state.mu, descent_vertex)

            if gap < self.tolerance:
                logger.debug(f"FW converged in {iteration + 1} iterations, gap={gap:.2e}")
                break

            # Update iterate
            if self.away_step:
                self._away_step_update(state, descent_vertex, gradient, theta)
            else:
                self._standard_update(state, descent_vertex, iteration)

        # Compute final result
        kl_div = self._kl_divergence(state.mu, theta)
        profit_cents = kl_div * 100  # Convert to cents

        # Trade direction: positive means sell (price should decrease)
        trade_direction = theta - state.mu

        return ArbitrageFWResult(
            projected_prices=state.mu,
            original_prices=theta,
            kl_divergence=kl_div,
            profit_cents=profit_cents,
            iterations=state.iteration + 1,
            converged=gap < self.tolerance,
            gap=gap,
            active_vertices=state.vertices,
            vertex_weights=state.weights,
            trade_direction=trade_direction,
        )

    def _find_initial_vertex(
        self,
        graph: DependencyGraph,
    ) -> NDArray[np.float64] | None:
        """Find an initial feasible vertex."""
        # Use uniform objective to find any feasible vertex
        n = len(graph.markets)
        if n == 0:
            return None

        c = np.ones(n)
        return self.ip_solver.find_violating_vertex(c, graph)

    def _find_descent_vertex(
        self,
        gradient: NDArray[np.float64],
        graph: DependencyGraph,
    ) -> NDArray[np.float64] | None:
        """
        Find vertex minimizing linear approximation (FW oracle).

        z_t = argmin_{z ∈ Z} <gradient, z>
        """
        return self.ip_solver.find_violating_vertex(gradient, graph)

    def _kl_gradient(
        self,
        mu: NDArray[np.float64],
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Compute gradient of KL divergence D_KL(μ || θ).

        ∇_μ D_KL(μ || θ) = log(μ) - log(θ) + 1
        """
        mu_safe = np.maximum(mu, 1e-10)
        theta_safe = np.maximum(theta, 1e-10)
        return np.log(mu_safe) - np.log(theta_safe) + 1

    def _kl_divergence(
        self,
        mu: NDArray[np.float64],
        theta: NDArray[np.float64],
    ) -> float:
        """
        Compute KL divergence D_KL(μ || θ).

        D_KL(μ || θ) = Σ μ_i log(μ_i / θ_i)
        """
        mu_safe = np.maximum(mu, 1e-10)
        theta_safe = np.maximum(theta, 1e-10)

        # Only compute for non-zero μ entries
        mask = mu_safe > 1e-9
        if not np.any(mask):
            return 0.0

        return float(np.sum(mu_safe[mask] * np.log(mu_safe[mask] / theta_safe[mask])))

    def _compute_gap(
        self,
        gradient: NDArray[np.float64],
        mu: NDArray[np.float64],
        descent_vertex: NDArray[np.float64],
    ) -> float:
        """
        Compute Frank-Wolfe duality gap.

        gap = <gradient, μ - z>
        """
        return float(np.dot(gradient, mu - descent_vertex))

    def _standard_update(
        self,
        state: FWState,
        descent_vertex: NDArray[np.float64],
        iteration: int,
    ) -> None:
        """Standard Frank-Wolfe update with decaying step size."""
        # Step size: 2/(t+2) for O(1/t) convergence
        gamma = 2.0 / (iteration + 2)

        # Update iterate
        state.mu = (1 - gamma) * state.mu + gamma * descent_vertex

        # Update vertex weights
        state.weights = (1 - gamma) * state.weights

        # Add new vertex if not already present
        is_new = True
        for i, v in enumerate(state.vertices):
            if np.allclose(v, descent_vertex):
                state.weights[i] += gamma
                is_new = False
                break

        if is_new:
            state.vertices.append(descent_vertex.copy())
            state.weights = np.append(state.weights, gamma)

        # Prune zero-weight vertices
        self._prune_vertices(state)

    def _away_step_update(
        self,
        state: FWState,
        descent_vertex: NDArray[np.float64],
        gradient: NDArray[np.float64],
        theta: NDArray[np.float64],
    ) -> None:
        """
        Away-step Frank-Wolfe for faster convergence.

        Chooses between FW direction and away direction based on
        which provides better descent.
        """
        # FW direction
        d_fw = descent_vertex - state.mu
        gap_fw = -np.dot(gradient, d_fw)

        # Away direction: find vertex with maximum gradient
        if len(state.vertices) > 1:
            vertex_gradients = [np.dot(gradient, v) for v in state.vertices]
            away_idx = np.argmax(vertex_gradients)
            away_vertex = state.vertices[away_idx]
            d_away = state.mu - away_vertex
            gap_away = np.dot(gradient, d_away)
        else:
            gap_away = 0
            away_idx = 0

        # Choose direction
        if gap_fw >= gap_away:
            # Use FW direction
            direction = d_fw
            gamma_max = 1.0

            # Line search
            gamma = self._line_search_kl(state.mu, direction, theta, gamma_max)

            # Update
            state.mu = state.mu + gamma * direction

            # Update weights
            state.weights = (1 - gamma) * state.weights

            # Add new vertex
            is_new = True
            for i, v in enumerate(state.vertices):
                if np.allclose(v, descent_vertex):
                    state.weights[i] += gamma
                    is_new = False
                    break

            if is_new:
                state.vertices.append(descent_vertex.copy())
                state.weights = np.append(state.weights, gamma)

        else:
            # Use away direction
            direction = d_away
            gamma_max = state.weights[away_idx] / (1 - state.weights[away_idx] + 1e-10)
            gamma_max = min(gamma_max, 1.0)

            # Line search
            gamma = self._line_search_kl(state.mu, direction, theta, gamma_max)

            # Update
            state.mu = state.mu + gamma * direction

            # Update weights
            state.weights = (1 + gamma) * state.weights
            state.weights[away_idx] -= gamma

        # Prune zero-weight vertices
        self._prune_vertices(state)

    def _line_search_kl(
        self,
        mu: NDArray[np.float64],
        direction: NDArray[np.float64],
        theta: NDArray[np.float64],
        gamma_max: float,
    ) -> float:
        """
        Line search for KL divergence minimization.

        Uses Armijo backtracking.
        """
        alpha = 0.3
        beta = 0.5
        gamma = gamma_max

        f_current = self._kl_divergence(mu, theta)
        grad = self._kl_gradient(mu, theta)
        slope = np.dot(grad, direction)

        for _ in range(20):
            mu_new = mu + gamma * direction

            # Ensure non-negative
            if np.any(mu_new < 0):
                gamma *= beta
                continue

            f_new = self._kl_divergence(mu_new, theta)

            if f_new <= f_current + alpha * gamma * slope:
                return gamma

            gamma *= beta

        return gamma

    def _prune_vertices(self, state: FWState, threshold: float = 1e-10) -> None:
        """Remove vertices with negligible weight."""
        mask = state.weights > threshold
        if np.sum(mask) < len(state.weights):
            state.vertices = [v for v, m in zip(state.vertices, mask) if m]
            state.weights = state.weights[mask]

            # Renormalize weights
            if len(state.weights) > 0:
                state.weights = state.weights / np.sum(state.weights)

    def _empty_result(self, theta: NDArray[np.float64]) -> ArbitrageFWResult:
        """Return empty result when optimization cannot proceed."""
        return ArbitrageFWResult(
            projected_prices=theta,
            original_prices=theta,
            kl_divergence=0.0,
            profit_cents=0.0,
            iterations=0,
            converged=False,
            gap=float('inf'),
            active_vertices=[],
            vertex_weights=np.array([]),
            trade_direction=np.zeros_like(theta),
        )

    def compute_optimal_trade(
        self,
        result: ArbitrageFWResult,
        capital: float,
        prices_cents: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], float]:
        """
        Compute optimal trade quantities from projection result.

        Args:
            result: FW optimization result
            capital: Available capital in dollars
            prices_cents: Current prices in cents

        Returns:
            Tuple of (quantities, expected_profit)
            Positive quantities = buy, negative = sell
        """
        if result.kl_divergence <= 0:
            return np.zeros_like(result.trade_direction), 0.0

        # Trade direction indicates price movement needed
        # If θ_i > μ*_i, price should decrease, so we sell
        # If θ_i < μ*_i, price should increase, so we buy

        direction = result.trade_direction
        magnitude = np.abs(direction)

        # Scale by capital
        total_magnitude = np.sum(magnitude)
        if total_magnitude < 1e-10:
            return np.zeros_like(direction), 0.0

        # Allocate capital proportionally to magnitude
        allocation = capital * magnitude / total_magnitude

        # Convert to quantities (contracts)
        quantities = np.zeros_like(direction)
        for i in range(len(direction)):
            if prices_cents[i] > 0:
                contracts = allocation[i] / (prices_cents[i] / 100)
                # Negative direction means buy (price should increase)
                quantities[i] = -np.sign(direction[i]) * contracts

        expected_profit = result.kl_divergence * capital

        return quantities, expected_profit
