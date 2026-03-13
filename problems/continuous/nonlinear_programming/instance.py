"""Nonlinear Programming (NLP) instance and solution definitions.

Problem: Minimize (or maximize) a nonlinear objective function subject to
nonlinear equality/inequality constraints and variable bounds.

    min f(x)
    s.t. g_i(x) <= 0   (inequality constraints)
         h_j(x) = 0    (equality constraints)
         lb <= x <= ub  (bounds)

References:
    Nocedal, J., & Wright, S. J. (2006). Numerical Optimization.
    Springer. 2nd edition.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class NLPInstance:
    """Nonlinear programming problem instance.

    Attributes:
        objective: Callable f(x) -> float.
        n_vars: Number of decision variables.
        x0: Initial guess.
        bounds: Optional list of (lb, ub) for each variable.
        ineq_constraints: List of callables g_i(x) <= 0.
        eq_constraints: List of callables h_j(x) = 0.
        gradient: Optional gradient of the objective.
        name: Problem name.
    """
    objective: Callable[[np.ndarray], float]
    n_vars: int
    x0: np.ndarray
    bounds: list[tuple[float | None, float | None]] | None = None
    ineq_constraints: list[Callable[[np.ndarray], float]] = field(default_factory=list)
    eq_constraints: list[Callable[[np.ndarray], float]] = field(default_factory=list)
    gradient: Callable[[np.ndarray], np.ndarray] | None = None
    name: str = "NLP"

    @classmethod
    def rosenbrock(cls, n: int = 2) -> NLPInstance:
        """Create a Rosenbrock function instance.

        f(x) = sum_{i=0}^{n-2} [100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]
        Global minimum at x* = (1, 1, ..., 1), f* = 0.

        Args:
            n: Number of dimensions.

        Returns:
            NLPInstance for the Rosenbrock problem.
        """
        def objective(x: np.ndarray) -> float:
            return float(np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2))

        def gradient(x: np.ndarray) -> np.ndarray:
            g = np.zeros_like(x)
            g[:-1] += -400.0 * x[:-1] * (x[1:] - x[:-1]**2) + 2.0 * (x[:-1] - 1)
            g[1:] += 200.0 * (x[1:] - x[:-1]**2)
            return g

        x0 = np.full(n, -1.0)
        return cls(objective=objective, n_vars=n, x0=x0,
                   gradient=gradient, name="Rosenbrock")

    @classmethod
    def sphere(cls, n: int = 3) -> NLPInstance:
        """Create a sphere function instance.

        f(x) = sum(x_i^2), global minimum at origin, f* = 0.

        Args:
            n: Number of dimensions.

        Returns:
            NLPInstance for the sphere problem.
        """
        def objective(x: np.ndarray) -> float:
            return float(np.sum(x**2))

        def gradient(x: np.ndarray) -> np.ndarray:
            return 2.0 * x

        x0 = np.ones(n) * 5.0
        return cls(objective=objective, n_vars=n, x0=x0,
                   gradient=gradient, name="Sphere")

    @classmethod
    def constrained_quadratic(cls) -> NLPInstance:
        """Create a constrained quadratic instance.

        min (x-1)^2 + (y-2)^2
        s.t. x + y <= 3
             x >= 0, y >= 0

        Optimal: x=1, y=2 (constraint inactive).

        Returns:
            NLPInstance for the constrained quadratic problem.
        """
        def objective(x: np.ndarray) -> float:
            return float((x[0] - 1)**2 + (x[1] - 2)**2)

        def ineq1(x: np.ndarray) -> float:
            return float(x[0] + x[1] - 3)  # x + y <= 3

        x0 = np.array([0.5, 0.5])
        bounds = [(0.0, None), (0.0, None)]
        return cls(objective=objective, n_vars=2, x0=x0,
                   bounds=bounds, ineq_constraints=[ineq1],
                   name="ConstrainedQuadratic")


@dataclass
class NLPSolution:
    """Solution to a nonlinear programming problem.

    Attributes:
        x: Optimal decision variable values.
        objective_value: Objective function value at x.
        success: Whether the solver converged.
        method: Solver method used.
        n_iterations: Number of iterations.
        message: Solver message.
    """
    x: np.ndarray
    objective_value: float
    success: bool
    method: str
    n_iterations: int
    message: str

    def __repr__(self) -> str:
        return (f"NLPSolution(obj={self.objective_value:.6f}, "
                f"method={self.method}, success={self.success})")
