"""Linear Programming with Sensitivity Analysis.

Minimize c^T x subject to A_ub x <= b_ub, A_eq x = b_eq, bounds.

Sensitivity analysis provides shadow prices (dual variables),
reduced costs, and allowable ranges for RHS and objective coefficients.

Complexity: Polynomial (interior point), exponential worst-case (simplex).

References:
    Dantzig, G. B. (1963). Linear Programming and Extensions. Princeton
    University Press.
    Bertsimas, D., & Tsitsiklis, J. N. (1997). Introduction to Linear
    Optimization. Athena Scientific.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LPInstance:
    """Linear programming instance.

    Minimize c^T x subject to A_ub x <= b_ub, A_eq x = b_eq, bounds.

    Attributes:
        n: Number of variables.
        c: Objective coefficients, shape (n,).
        A_ub: Inequality constraint matrix, shape (m_ub, n) or None.
        b_ub: Inequality RHS, shape (m_ub,) or None.
        A_eq: Equality constraint matrix, shape (m_eq, n) or None.
        b_eq: Equality RHS, shape (m_eq,) or None.
        bounds: List of (lb, ub) tuples for each variable, or None.
    """

    n: int
    c: np.ndarray
    A_ub: np.ndarray | None = None
    b_ub: np.ndarray | None = None
    A_eq: np.ndarray | None = None
    b_eq: np.ndarray | None = None
    bounds: list[tuple[float | None, float | None]] | None = None

    @classmethod
    def random(cls, n: int = 4, m_ub: int = 3,
               seed: int | None = None) -> LPInstance:
        """Generate a random bounded feasible LP instance.

        Args:
            n: Number of variables.
            m_ub: Number of inequality constraints.
            seed: Random seed.

        Returns:
            A random LPInstance.
        """
        rng = np.random.default_rng(seed)
        c = rng.uniform(-5, 5, size=n)
        # Use non-negative coefficients so that x >= 0 ensures feasibility
        A_ub = rng.uniform(0.5, 3, size=(m_ub, n))
        # RHS chosen so feasible region is bounded and non-empty
        b_ub = rng.uniform(5, 20, size=m_ub)
        bounds = [(0.0, None) for _ in range(n)]
        return cls(n=n, c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)

    def objective(self, x: np.ndarray) -> float:
        """Evaluate objective at x.

        Args:
            x: Variable vector.

        Returns:
            c^T x.
        """
        return float(self.c @ x)


@dataclass
class LPSolution:
    """Solution to a linear program with sensitivity information.

    Attributes:
        x: Optimal variable values, shape (n,).
        objective: Optimal objective value.
        shadow_prices: Dual variables for inequality constraints.
        reduced_costs: Reduced costs for each variable.
        success: Whether the solver found an optimal solution.
        message: Solver status message.
    """

    x: np.ndarray
    objective: float
    shadow_prices: np.ndarray | None
    reduced_costs: np.ndarray | None
    success: bool
    message: str

    def __repr__(self) -> str:
        return (f"LPSolution(objective={self.objective:.6f}, "
                f"success={self.success})")
