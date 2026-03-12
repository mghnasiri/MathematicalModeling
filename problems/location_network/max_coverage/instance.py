"""Maximum Coverage Problem.

Given a universe U of n elements, a collection S of m subsets of U,
and a budget k, select at most k subsets to maximize the number of
covered elements.

Complexity: NP-hard. Greedy achieves (1 - 1/e) approximation (Nemhauser et al., 1978).

References:
    Nemhauser, G. L., Wolsey, L. A., & Fisher, M. L. (1978). An analysis
    of approximations for maximizing submodular set functions. Mathematical
    Programming, 14(1), 265-294.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MaxCoverageInstance:
    """Maximum coverage problem instance.

    Attributes:
        n: Number of elements in the universe.
        m: Number of available subsets.
        k: Budget (max subsets to select).
        subsets: List of sets, subsets[j] is the set of elements covered by subset j.
    """

    n: int
    m: int
    k: int
    subsets: list[set[int]]

    @classmethod
    def random(cls, n: int = 20, m: int = 10, k: int = 3,
               density: float = 0.3,
               seed: int | None = None) -> MaxCoverageInstance:
        """Generate a random max coverage instance.

        Args:
            n: Number of elements.
            m: Number of subsets.
            k: Budget.
            density: Probability each element is in each subset.
            seed: Random seed.

        Returns:
            A random MaxCoverageInstance.
        """
        rng = np.random.default_rng(seed)
        subsets = []
        for _ in range(m):
            mask = rng.random(n) < density
            s = set(np.where(mask)[0].tolist())
            if len(s) == 0:
                # Ensure non-empty
                s.add(int(rng.integers(0, n)))
            subsets.append(s)
        return cls(n=n, m=m, k=k, subsets=subsets)

    def coverage(self, selected: list[int]) -> set[int]:
        """Return the set of elements covered by the selected subsets.

        Args:
            selected: Indices of selected subsets.

        Returns:
            Set of covered element indices.
        """
        covered: set[int] = set()
        for j in selected:
            covered |= self.subsets[j]
        return covered


@dataclass
class MaxCoverageSolution:
    """Solution to a max coverage problem.

    Attributes:
        selected: List of selected subset indices.
        covered: Set of covered elements.
        objective: Number of covered elements.
    """

    selected: list[int]
    covered: set[int]
    objective: int

    def __repr__(self) -> str:
        return (f"MaxCoverageSolution(selected={self.selected}, "
                f"objective={self.objective})")
