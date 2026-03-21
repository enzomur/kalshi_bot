"""Frank-Wolfe (Conditional Gradient) optimizer for portfolio optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from kalshi_bot.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class OptimizationResult:
    """Result of Frank-Wolfe optimization."""

    solution: NDArray[np.float64]
    objective_value: float
    iterations: int
    converged: bool
    gap: float


class FrankWolfeOptimizer:
    """
    Implements the Frank-Wolfe (Conditional Gradient) algorithm.

    Frank-Wolfe is a projection-free algorithm for constrained optimization.
    It's particularly useful when:
    - The constraint set is a polytope (like the simplex)
    - Linear optimization over the constraint set is cheap
    - Projections are expensive

    The algorithm iteratively:
    1. Computes gradient at current point
    2. Finds direction minimizing linear approximation over constraints
    3. Takes a step in that direction

    For portfolio optimization on the simplex, the linear minimization
    step has a closed-form solution: move to the vertex with minimum gradient.
    """

    def __init__(
        self,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        line_search: bool = True,
    ) -> None:
        """
        Initialize optimizer.

        Args:
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance (duality gap)
            line_search: Use line search for step size
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.line_search = line_search

    def optimize_simplex(
        self,
        objective: Callable[[NDArray[np.float64]], float],
        gradient: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        n: int,
        x0: NDArray[np.float64] | None = None,
    ) -> OptimizationResult:
        """
        Minimize objective over the probability simplex.

        Args:
            objective: Objective function f(x)
            gradient: Gradient function grad_f(x)
            n: Dimension of the simplex
            x0: Initial point (default: uniform)

        Returns:
            OptimizationResult with solution
        """
        if x0 is None:
            x = np.ones(n) / n
        else:
            x = x0.copy()

        for iteration in range(self.max_iterations):
            grad = gradient(x)

            min_idx = np.argmin(grad)
            s = np.zeros(n)
            s[min_idx] = 1.0

            direction = s - x

            gap = -np.dot(grad, direction)

            if gap < self.tolerance:
                return OptimizationResult(
                    solution=x,
                    objective_value=objective(x),
                    iterations=iteration + 1,
                    converged=True,
                    gap=gap,
                )

            if self.line_search:
                step = self._line_search(objective, x, direction)
            else:
                step = 2.0 / (iteration + 2)

            x = x + step * direction

        return OptimizationResult(
            solution=x,
            objective_value=objective(x),
            iterations=self.max_iterations,
            converged=False,
            gap=gap,
        )

    def optimize_box_simplex(
        self,
        objective: Callable[[NDArray[np.float64]], float],
        gradient: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        n: int,
        lower: NDArray[np.float64],
        upper: NDArray[np.float64],
        x0: NDArray[np.float64] | None = None,
    ) -> OptimizationResult:
        """
        Minimize objective over simplex with box constraints.

        Constraints: sum(x) = 1, lower <= x <= upper

        Args:
            objective: Objective function
            gradient: Gradient function
            n: Dimension
            lower: Lower bounds
            upper: Upper bounds
            x0: Initial point

        Returns:
            OptimizationResult with solution
        """
        if x0 is None:
            x = np.clip(np.ones(n) / n, lower, upper)
            x = x / np.sum(x)
        else:
            x = x0.copy()

        for iteration in range(self.max_iterations):
            grad = gradient(x)

            s = self._linear_oracle_box_simplex(grad, lower, upper)

            direction = s - x

            gap = -np.dot(grad, direction)

            if gap < self.tolerance:
                return OptimizationResult(
                    solution=x,
                    objective_value=objective(x),
                    iterations=iteration + 1,
                    converged=True,
                    gap=gap,
                )

            if self.line_search:
                step = self._line_search_constrained(
                    objective, x, direction, lower, upper
                )
            else:
                step = 2.0 / (iteration + 2)

            x = x + step * direction

        return OptimizationResult(
            solution=x,
            objective_value=objective(x),
            iterations=self.max_iterations,
            converged=False,
            gap=gap,
        )

    def _linear_oracle_box_simplex(
        self,
        c: NDArray[np.float64],
        lower: NDArray[np.float64],
        upper: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Solve linear program: min c'x s.t. sum(x)=1, lower<=x<=upper

        Uses a greedy approach: allocate to variables with smallest
        cost coefficients, respecting bounds.
        """
        n = len(c)
        x = lower.copy()
        remaining = 1.0 - np.sum(x)

        indices = np.argsort(c)

        for i in indices:
            if remaining <= 0:
                break

            space = upper[i] - x[i]
            allocation = min(space, remaining)
            x[i] += allocation
            remaining -= allocation

        return x

    def _line_search(
        self,
        objective: Callable[[NDArray[np.float64]], float],
        x: NDArray[np.float64],
        direction: NDArray[np.float64],
        max_step: float = 1.0,
    ) -> float:
        """Backtracking line search."""
        step = max_step
        alpha = 0.3
        beta = 0.5

        f_x = objective(x)

        for _ in range(20):
            x_new = x + step * direction
            if objective(x_new) < f_x - alpha * step * np.dot(-direction, direction):
                return step
            step *= beta

        return step

    def _line_search_constrained(
        self,
        objective: Callable[[NDArray[np.float64]], float],
        x: NDArray[np.float64],
        direction: NDArray[np.float64],
        lower: NDArray[np.float64],
        upper: NDArray[np.float64],
    ) -> float:
        """Line search respecting box constraints."""
        max_step = 1.0

        for i in range(len(x)):
            if direction[i] > 0:
                max_step = min(max_step, (upper[i] - x[i]) / direction[i])
            elif direction[i] < 0:
                max_step = min(max_step, (lower[i] - x[i]) / direction[i])

        return self._line_search(objective, x, direction, max_step)

    def maximize_portfolio_return(
        self,
        expected_returns: NDArray[np.float64],
        position_limits: NDArray[np.float64] | None = None,
    ) -> OptimizationResult:
        """
        Maximize expected return subject to simplex constraint.

        Args:
            expected_returns: Expected return for each asset
            position_limits: Maximum allocation per asset

        Returns:
            Optimal allocation
        """
        n = len(expected_returns)

        def objective(x: NDArray[np.float64]) -> float:
            return -np.dot(expected_returns, x)

        def gradient(x: NDArray[np.float64]) -> NDArray[np.float64]:
            return -expected_returns

        if position_limits is not None:
            return self.optimize_box_simplex(
                objective,
                gradient,
                n,
                np.zeros(n),
                position_limits,
            )
        else:
            return self.optimize_simplex(objective, gradient, n)
