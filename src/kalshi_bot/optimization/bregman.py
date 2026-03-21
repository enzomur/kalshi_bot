"""Bregman projection for probability simplex constraints."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from kalshi_bot.utils.logging import get_logger

logger = get_logger(__name__)


class BregmanProjection:
    """
    Implements Bregman projection onto the probability simplex.

    Bregman projection is used to project a point onto a convex set
    while minimizing a Bregman divergence (e.g., KL divergence).

    For portfolio optimization, this ensures allocations sum to 1
    while minimizing distance from the target allocation.

    The KL-divergence projection onto the simplex has closed-form solution:
        x_i* = x_i * exp(-lambda) / Z

    Where lambda is chosen so that sum(x_i*) = 1.
    """

    def __init__(
        self,
        max_iterations: int = 100,
        tolerance: float = 1e-8,
    ) -> None:
        """
        Initialize Bregman projection.

        Args:
            max_iterations: Maximum iterations for iterative methods
            tolerance: Convergence tolerance
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def project_simplex(
        self,
        x: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Project point onto the probability simplex.

        Uses the efficient O(n log n) algorithm by Duchi et al.

        Args:
            x: Input vector to project

        Returns:
            Projection onto simplex (sum = 1, all >= 0)
        """
        n = len(x)
        sorted_x = np.sort(x)[::-1]

        cumsum = np.cumsum(sorted_x)
        indices = np.arange(1, n + 1)
        threshold_candidates = (cumsum - 1) / indices

        rho = np.max(np.where(sorted_x > threshold_candidates)[0]) + 1
        theta = (cumsum[rho - 1] - 1) / rho

        projected = np.maximum(x - theta, 0)

        return projected

    def project_simplex_kl(
        self,
        x: NDArray[np.float64],
        reference: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """
        Project onto simplex using KL-divergence.

        Minimizes KL(p || reference) subject to p being on simplex.
        If reference is None, uses uniform distribution.

        Args:
            x: Input vector (can be log-probabilities)
            reference: Reference distribution for KL divergence

        Returns:
            Projection minimizing KL divergence
        """
        if reference is None:
            reference = np.ones(len(x)) / len(x)

        exp_x = np.exp(x - np.max(x))

        result = reference * exp_x
        result = result / np.sum(result)

        result = np.maximum(result, 1e-10)
        result = result / np.sum(result)

        return result

    def project_with_bounds(
        self,
        x: NDArray[np.float64],
        lower: NDArray[np.float64] | None = None,
        upper: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """
        Project onto simplex with additional box constraints.

        Uses iterative Bregman projection (Dykstra's algorithm variant).

        Args:
            x: Input vector
            lower: Lower bounds (default: 0)
            upper: Upper bounds (default: 1)

        Returns:
            Projection satisfying simplex and box constraints
        """
        n = len(x)

        if lower is None:
            lower = np.zeros(n)
        if upper is None:
            upper = np.ones(n)

        y = x.copy()

        for iteration in range(self.max_iterations):
            y_prev = y.copy()

            y = np.clip(y, lower, upper)

            y = self.project_simplex(y)

            if np.max(np.abs(y - y_prev)) < self.tolerance:
                logger.debug(f"Converged in {iteration + 1} iterations")
                break

        return y

    def project_scaled_simplex(
        self,
        x: NDArray[np.float64],
        budget: float,
    ) -> NDArray[np.float64]:
        """
        Project onto scaled simplex (sum = budget, all >= 0).

        Args:
            x: Input vector
            budget: Target sum

        Returns:
            Projection with sum = budget
        """
        normalized = self.project_simplex(x / budget)
        return normalized * budget

    def kl_divergence(
        self,
        p: NDArray[np.float64],
        q: NDArray[np.float64],
    ) -> float:
        """
        Calculate KL divergence D_KL(p || q).

        Args:
            p: First distribution
            q: Second distribution

        Returns:
            KL divergence value
        """
        p = np.maximum(p, 1e-10)
        q = np.maximum(q, 1e-10)

        return float(np.sum(p * np.log(p / q)))

    def project_onto_marginal_polytope(
        self,
        theta: NDArray[np.float64],
        valid_vertices: list[NDArray[np.float64]],
    ) -> tuple[NDArray[np.float64], float]:
        """
        Project prices onto marginal polytope defined by vertices.

        Finds μ* = argmin_{μ∈M} D_KL(μ || θ) where M is the convex hull
        of the valid vertices.

        This is used for dependency-based arbitrage detection where the
        valid outcomes form a polytope (not necessarily a simplex).

        Args:
            theta: Current market prices (normalized)
            valid_vertices: List of valid outcome vertices (binary vectors)

        Returns:
            Tuple of (projected_prices, kl_divergence)
        """
        if not valid_vertices:
            return theta.copy(), 0.0

        theta = np.maximum(theta, 1e-10)
        theta = theta / np.sum(theta)

        # Simple projection: find convex combination minimizing KL
        # For small vertex sets, use direct optimization
        n_vertices = len(valid_vertices)

        if n_vertices == 1:
            mu = np.array(valid_vertices[0], dtype=np.float64)
            mu = np.maximum(mu, 1e-10)
            mu = mu / np.sum(mu)
            return mu, self.kl_divergence(mu, theta)

        # Use iterative Bregman projection onto convex hull
        from scipy.optimize import minimize

        def objective(weights: NDArray[np.float64]) -> float:
            """KL divergence of weighted combination."""
            # Compute convex combination
            mu = np.zeros_like(theta)
            for i, v in enumerate(valid_vertices):
                mu += weights[i] * np.array(v, dtype=np.float64)

            mu = np.maximum(mu, 1e-10)
            mu_sum = np.sum(mu)
            if mu_sum > 0:
                mu = mu / mu_sum

            return self.kl_divergence(mu, theta)

        # Initial weights: uniform
        w0 = np.ones(n_vertices) / n_vertices

        # Constraints: weights sum to 1, weights >= 0
        from scipy.optimize import LinearConstraint, Bounds

        constraints = LinearConstraint(
            np.ones((1, n_vertices)),
            lb=1.0,
            ub=1.0,
        )
        bounds = Bounds(lb=np.zeros(n_vertices), ub=np.ones(n_vertices))

        result = minimize(
            objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            options={'maxiter': 100},
        )

        # Compute optimal mu
        mu = np.zeros_like(theta)
        for i, v in enumerate(valid_vertices):
            mu += result.x[i] * np.array(v, dtype=np.float64)

        mu = np.maximum(mu, 1e-10)
        mu = mu / np.sum(mu)

        kl_div = self.kl_divergence(mu, theta)

        return mu, kl_div

    def compute_optimal_trade_direction(
        self,
        current_prices: NDArray[np.float64],
        projected_prices: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Compute the direction to trade based on Bregman projection.

        The trade direction is θ - μ* where:
        - θ is current (arbitrageable) prices
        - μ* is the projected (arbitrage-free) prices

        Positive values indicate the price should decrease (sell pressure).
        Negative values indicate the price should increase (buy pressure).

        Args:
            current_prices: θ - current market prices
            projected_prices: μ* - arbitrage-free projection

        Returns:
            Trade direction vector
        """
        return current_prices - projected_prices

    def compute_trade_quantities(
        self,
        direction: NDArray[np.float64],
        capital: float,
        prices_cents: NDArray[np.float64],
        liquidity: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """
        Compute trade quantities from trade direction.

        Args:
            direction: Trade direction (from compute_optimal_trade_direction)
            capital: Available capital in dollars
            prices_cents: Current prices in cents
            liquidity: Available liquidity per market (optional constraint)

        Returns:
            Quantities to trade (positive = buy, negative = sell)
        """
        magnitude = np.abs(direction)
        total_magnitude = np.sum(magnitude)

        if total_magnitude < 1e-10:
            return np.zeros_like(direction)

        # Allocate capital proportionally
        allocation = capital * magnitude / total_magnitude

        # Convert to contract quantities
        quantities = np.zeros_like(direction)
        for i in range(len(direction)):
            if prices_cents[i] > 0:
                contracts = allocation[i] / (prices_cents[i] / 100)
                # Negative direction = price should go up = buy
                quantities[i] = -np.sign(direction[i]) * contracts

                # Apply liquidity constraint if provided
                if liquidity is not None:
                    max_qty = liquidity[i]
                    quantities[i] = np.clip(quantities[i], -max_qty, max_qty)

        return quantities

    def compute_arbitrage_profit(
        self,
        theta: NDArray[np.float64],
        mu_star: NDArray[np.float64],
    ) -> float:
        """
        Compute the arbitrage profit from Bregman projection.

        The profit equals D_KL(μ* || θ) per dollar invested.

        Args:
            theta: Original (arbitrageable) prices
            mu_star: Projected (arbitrage-free) prices

        Returns:
            Profit in dollars per dollar invested
        """
        theta = np.maximum(theta, 1e-10)
        mu_star = np.maximum(mu_star, 1e-10)

        # Normalize
        theta = theta / np.sum(theta)
        mu_star = mu_star / np.sum(mu_star)

        return self.kl_divergence(mu_star, theta)
