"""
Set Covering Problem (SCP)

Given a universe U = {1,...,m} and a collection S = {S_1,...,S_n} of subsets
with costs c_j, find a minimum-cost sub-collection covering all elements.

    min  sum_{j=1}^{n} c_j x_j
    s.t. sum_{j: i in S_j} x_j >= 1   for all i in U
         x_j in {0,1}

Complexity: NP-hard. Greedy achieves ln(m)+1 approximation (best possible
unless P=NP).

References:
    - Chvátal, V. (1979). A greedy heuristic for the set-covering problem.
      Math. Oper. Res., 4(3), 233-235. https://doi.org/10.1287/moor.4.3.233
    - Caprara, A., Fischetti, M. & Toth, P. (1999). A heuristic method for
      the set covering problem. Oper. Res., 47(5), 730-743.
      https://doi.org/10.1287/opre.47.5.730
"""
from __future__ import annotations

import sys
import os
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


@dataclass
class SetCoveringInstance:
    """Set covering problem instance.

    Args:
        m: Number of elements in the universe.
        n: Number of subsets.
        subsets: List of sets, each containing element indices (0-indexed).
        costs: Cost of each subset (n,).
    """
    m: int
    n: int
    subsets: list[set[int]]
    costs: np.ndarray

    def __post_init__(self):
        self.costs = np.asarray(self.costs, dtype=float)

    def is_cover(self, selected: list[int]) -> bool:
        """Check if selected subsets cover the entire universe."""
        covered = set()
        for j in selected:
            covered |= self.subsets[j]
        return len(covered) == self.m

    def total_cost(self, selected: list[int]) -> float:
        """Total cost of selected subsets."""
        return float(sum(self.costs[j] for j in selected))

    @classmethod
    def random(cls, m: int = 20, n: int = 30, density: float = 0.3,
               seed: int = 42) -> SetCoveringInstance:
        rng = np.random.default_rng(seed)
        subsets = []
        for _ in range(n):
            size = max(1, int(m * density * rng.uniform(0.5, 1.5)))
            subsets.append(set(rng.choice(m, size=min(size, m), replace=False).tolist()))
        # Ensure feasibility
        uncovered = set(range(m)) - set().union(*subsets)
        for e in uncovered:
            subsets[rng.integers(n)].add(e)
        costs = rng.uniform(1, 10, n)
        return cls(m=m, n=n, subsets=subsets, costs=costs)


@dataclass
class SetCoveringSolution:
    """Solution to the set covering problem."""
    selected: list[int]
    total_cost: float
    n_selected: int

    def __repr__(self) -> str:
        return (f"SetCoveringSolution(n_selected={self.n_selected}, "
                f"cost={self.total_cost:.2f})")
