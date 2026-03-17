"""
Distributionally Robust Optimization (DRO)

Instead of assuming a known probability distribution, DRO optimizes
over a family (ambiguity set) of distributions. The decision maker
minimizes the worst-case expected cost over all distributions in
the ambiguity set.

    min_x  max_{P in A}  E_P[f(x, xi)]

Common ambiguity sets:
    - Moment-based: distributions with given mean and covariance.
    - Wasserstein ball: distributions within Wasserstein distance
      epsilon of an empirical distribution.

This module implements a linear DRO with discrete support and
moment constraints (mean matching + bounded covariance).

Complexity: The inner maximization is a linear program over
            probability weights. The full DRO can be reformulated
            as a finite-dimensional convex problem.

References:
    - Delage, E. & Ye, Y. (2010). Distributionally robust optimization
      under moment uncertainty with application to data-driven problems.
      Oper. Res., 58(3), 595-612.
      https://doi.org/10.1287/opre.1090.0741
    - Esfahani, P.M. & Kuhn, D. (2018). Data-driven distributionally robust
      optimization using the Wasserstein metric. Math. Program., 171(1-2),
      115-166. https://doi.org/10.1007/s10107-017-1172-1
    - Rahimian, H. & Mehrotra, S. (2022). Frameworks and results in
      distributionally robust optimization. Open J. Math. Optim., 3(4),
      1-85. https://doi.org/10.5802/ojmo.15
"""
from __future__ import annotations

import sys
import os
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


@dataclass
class DROInstance:
    """Distributionally robust optimization instance.

    A linear cost function f(x, xi) = (c + xi)^T x, where xi is a
    random perturbation vector drawn from an ambiguity set.

    Args:
        n: Decision dimension.
        c: Nominal cost vector (n,).
        support_points: Possible values of xi (K, n) — the discrete support.
        nominal_probs: Nominal (empirical) distribution over support (K,).
        A_ub: Inequality constraint matrix (m, n). A_ub x <= b_ub.
        b_ub: Inequality RHS (m,).
        wasserstein_radius: Epsilon for Wasserstein ambiguity set.
        mean_target: Target mean for moment-based ambiguity (n,). Optional.
    """
    n: int
    c: np.ndarray
    support_points: np.ndarray
    nominal_probs: np.ndarray | None = None
    A_ub: np.ndarray | None = None
    b_ub: np.ndarray | None = None
    wasserstein_radius: float = 0.1
    mean_target: np.ndarray | None = None

    def __post_init__(self):
        self.c = np.asarray(self.c, dtype=float)
        self.support_points = np.asarray(self.support_points, dtype=float)
        if self.nominal_probs is None:
            K = len(self.support_points)
            self.nominal_probs = np.full(K, 1.0 / K)
        else:
            self.nominal_probs = np.asarray(self.nominal_probs, dtype=float)
        if self.A_ub is not None:
            self.A_ub = np.asarray(self.A_ub, dtype=float)
            self.b_ub = np.asarray(self.b_ub, dtype=float)
        if self.mean_target is not None:
            self.mean_target = np.asarray(self.mean_target, dtype=float)

    @property
    def n_support(self) -> int:
        return len(self.support_points)

    def cost(self, x: np.ndarray, xi: np.ndarray) -> float:
        """Cost under decision x and perturbation xi."""
        return float(np.dot(self.c + xi, x))

    def nominal_expected_cost(self, x: np.ndarray) -> float:
        """Expected cost under the nominal distribution."""
        costs = np.array([self.cost(x, xi) for xi in self.support_points])
        return float(np.dot(costs, self.nominal_probs))

    def worst_case_cost(self, x: np.ndarray, probs: np.ndarray) -> float:
        """Expected cost under a given distribution over support."""
        costs = np.array([self.cost(x, xi) for xi in self.support_points])
        return float(np.dot(costs, probs))

    @classmethod
    def random(cls, n: int = 3, n_support: int = 10,
               seed: int = 42) -> DROInstance:
        """Generate a random DRO instance."""
        rng = np.random.default_rng(seed)
        c = rng.uniform(1, 10, n)
        support_points = rng.normal(0, 2, (n_support, n))
        A_ub = np.vstack([np.eye(n), -np.eye(n)])
        b_ub = np.concatenate([np.ones(n), np.zeros(n)])  # 0 <= x <= 1
        return cls(
            n=n, c=c, support_points=support_points,
            A_ub=A_ub, b_ub=b_ub, wasserstein_radius=0.5,
        )


@dataclass
class DROSolution:
    """Solution to a DRO problem.

    Args:
        x: Decision vector.
        worst_case_cost: max_{P in A} E_P[cost].
        nominal_cost: E[cost] under nominal distribution.
        worst_case_probs: Distribution achieving worst case.
    """
    x: np.ndarray
    worst_case_cost: float
    nominal_cost: float
    worst_case_probs: np.ndarray | None = None

    def __repr__(self) -> str:
        return (f"DROSolution(x={np.round(self.x, 3)}, "
                f"WC_cost={self.worst_case_cost:.3f}, "
                f"nom_cost={self.nominal_cost:.3f})")
