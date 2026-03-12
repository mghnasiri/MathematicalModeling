"""Quadratic Programming (QP).

Minimize (1/2) x^T Q x + c^T x
subject to A_eq x = b_eq, A_ub x <= b_ub, lb <= x <= ub.

Q is a symmetric positive semi-definite matrix for convex QP.

Complexity: Polynomial for convex QP (interior point methods);
NP-hard for non-convex QP.

References:
    Nocedal, J., & Wright, S. J. (2006). Numerical Optimization (2nd ed.).
    Springer.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class QPInstance:
    """Quadratic programming instance.

    Minimize (1/2) x^T Q x + c^T x
    subject to A_ub @ x <= b_ub, A_eq @ x = b_eq, bounds.

    Attributes:
        n: Number of variables.
        Q: Quadratic cost matrix, shape (n, n), symmetric PSD.
        c: Linear cost vector, shape (n,).
        A_ub: Inequality constraint matrix, shape (m_ub, n) or None.
        b_ub: Inequality RHS, shape (m_ub,) or None.
        A_eq: Equality constraint matrix, shape (m_eq, n) or None.
        b_eq: Equality RHS, shape (m_eq,) or None.
        bounds: List of (lb, ub) tuples for each variable, or None.
    """

    n: int
    Q: np.ndarray
    c: np.ndarray
    A_ub: np.ndarray | None = None
    b_ub: np.ndarray | None = None
    A_eq: np.ndarray | None = None
    b_eq: np.ndarray | None = None
    bounds: list[tuple[float | None, float | None]] | None = None

    @classmethod
    def random(cls, n: int = 5, m_ub: int = 3,
               seed: int | None = None) -> QPInstance:
        """Generate a random convex QP instance.

        Args:
            n: Number of variables.
            m_ub: Number of inequality constraints.
            seed: Random seed.

        Returns:
            A random QPInstance with PSD Q.
        """
        rng = np.random.default_rng(seed)
        # Generate PSD Q = A^T A + small diagonal
        A = rng.standard_normal((n, n))
        Q = A.T @ A + 0.1 * np.eye(n)
        Q = (Q + Q.T) / 2  # Ensure symmetry

        c = rng.standard_normal(n)

        A_ub = rng.standard_normal((m_ub, n))
        # Choose b_ub so that origin is feasible
        b_ub = np.abs(A_ub @ np.zeros(n)) + rng.uniform(1, 5, size=m_ub)

        bounds = [(0.0, None) for _ in range(n)]

        return cls(n=n, Q=Q, c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)

    def objective(self, x: np.ndarray) -> float:
        """Evaluate objective at x.

        Args:
            x: Decision variable vector.

        Returns:
            Objective value.
        """
        return float(0.5 * x @ self.Q @ x + self.c @ x)


@dataclass
class QPSolution:
    """Solution to a quadratic program.

    Attributes:
        x: Optimal variable values, shape (n,).
        objective: Optimal objective value.
        success: Whether the solver converged.
        message: Solver status message.
    """

    x: np.ndarray
    objective: float
    success: bool
    message: str

    def __repr__(self) -> str:
        return (f"QPSolution(objective={self.objective:.6f}, "
                f"success={self.success})")
