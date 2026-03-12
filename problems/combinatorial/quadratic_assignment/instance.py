"""
Quadratic Assignment Problem (QAP)

Given n facilities and n locations, a flow matrix F (n x n) and a
distance matrix D (n x n), find an assignment (permutation) pi
minimizing:

    sum_{i,j} F[i,j] * D[pi(i), pi(j)]

Generalizes LAP: QAP reduces to LAP when F or D is diagonal.

Complexity: NP-hard. One of the hardest combinatorial optimization
problems — instances with n > 30 are very challenging.

References:
    - Koopmans, T.C. & Beckmann, M. (1957). Assignment problems and the
      location of economic activities. Econometrica, 25(1), 53-76.
      https://doi.org/10.2307/1907742
    - Burkard, R.E., Dell'Amico, M. & Martello, S. (2012). Assignment
      Problems, Revised Reprint. SIAM.
    - QAPLIB: http://www.opt.math.tugraz.at/qaplib/
"""
from __future__ import annotations

import sys
import os
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


@dataclass
class QAPInstance:
    """Quadratic assignment problem instance.

    Args:
        n: Problem size (number of facilities/locations).
        flow: Flow matrix (n, n) — flow between facilities.
        distance: Distance matrix (n, n) — distance between locations.
    """
    n: int
    flow: np.ndarray
    distance: np.ndarray

    def __post_init__(self):
        self.flow = np.asarray(self.flow, dtype=float)
        self.distance = np.asarray(self.distance, dtype=float)

    def objective(self, permutation: list[int]) -> float:
        """Compute QAP objective: sum F[i,j] * D[pi(i), pi(j)]."""
        total = 0.0
        for i in range(self.n):
            for j in range(self.n):
                total += self.flow[i, j] * self.distance[permutation[i], permutation[j]]
        return total

    def delta_swap(self, permutation: list[int], r: int, s: int) -> float:
        """Compute change in objective when swapping positions r and s."""
        n = self.n
        pi = permutation
        delta = 0.0
        for k in range(n):
            if k != r and k != s:
                delta += (self.flow[r, k] - self.flow[s, k]) * (
                    self.distance[pi[s], pi[k]] - self.distance[pi[r], pi[k]]
                )
                delta += (self.flow[k, r] - self.flow[k, s]) * (
                    self.distance[pi[k], pi[s]] - self.distance[pi[k], pi[r]]
                )
        delta += (self.flow[r, s] - self.flow[s, r]) * (
            self.distance[pi[s], pi[r]] - self.distance[pi[r], pi[s]]
        )
        return delta

    @classmethod
    def random(cls, n: int = 10, seed: int = 42) -> QAPInstance:
        rng = np.random.default_rng(seed)
        flow = rng.integers(0, 20, (n, n)).astype(float)
        np.fill_diagonal(flow, 0)
        coords = rng.uniform(0, 100, (n, 2))
        distance = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                distance[i, j] = np.sqrt(np.sum((coords[i] - coords[j]) ** 2))
        return cls(n=n, flow=flow, distance=distance)

    @classmethod
    def nug5(cls) -> QAPInstance:
        """Nugent 5 benchmark (optimal = 50)."""
        flow = np.array([
            [0,5,2,4,1],[5,0,3,0,2],[2,3,0,0,0],[4,0,0,0,5],[1,2,0,5,0]
        ], dtype=float)
        dist = np.array([
            [0,1,1,2,3],[1,0,2,1,2],[1,2,0,1,2],[2,1,1,0,1],[3,2,2,1,0]
        ], dtype=float)
        return cls(n=5, flow=flow, distance=dist)


@dataclass
class QAPSolution:
    permutation: list[int]
    objective: float

    def __repr__(self) -> str:
        return f"QAPSolution(perm={self.permutation}, obj={self.objective:.0f})"
