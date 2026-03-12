"""Bi-Objective 0-1 Knapsack Problem.

Given n items, each with weight w_i and two value dimensions v1_i and v2_i,
and a knapsack capacity W, find the Pareto-optimal set of solutions
maximizing both objectives subject to capacity.

Complexity: NP-hard (generalizes single-objective knapsack).

References:
    Ehrgott, M. (2005). Multicriteria Optimization (2nd ed.). Springer.
    Bazgan, C., Hugot, H., & Vanderpooten, D. (2009). Solving efficiently
    the 0-1 multi-objective knapsack problem. Computers & Operations Research,
    36(1), 260-279.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BiObjectiveKnapsackInstance:
    """Bi-objective 0-1 knapsack instance.

    Attributes:
        n: Number of items.
        weights: Item weights, shape (n,).
        values1: First objective values, shape (n,).
        values2: Second objective values, shape (n,).
        capacity: Knapsack capacity.
    """

    n: int
    weights: np.ndarray
    values1: np.ndarray
    values2: np.ndarray
    capacity: int

    @classmethod
    def random(cls, n: int = 10, capacity: int | None = None,
               seed: int | None = None) -> BiObjectiveKnapsackInstance:
        """Generate a random bi-objective knapsack instance.

        Args:
            n: Number of items.
            capacity: Knapsack capacity (defaults to ~half total weight).
            seed: Random seed.

        Returns:
            A random BiObjectiveKnapsackInstance.
        """
        rng = np.random.default_rng(seed)
        weights = rng.integers(1, 20, size=n)
        values1 = rng.integers(1, 50, size=n)
        values2 = rng.integers(1, 50, size=n)
        if capacity is None:
            capacity = int(weights.sum() * 0.5)
        return cls(n=n, weights=weights, values1=values1, values2=values2,
                   capacity=capacity)

    def is_feasible(self, x: np.ndarray) -> bool:
        """Check if a binary solution vector is feasible.

        Args:
            x: Binary selection vector.

        Returns:
            True if total weight <= capacity.
        """
        return int(np.dot(self.weights, x)) <= self.capacity

    def evaluate(self, x: np.ndarray) -> tuple[float, float]:
        """Evaluate both objectives.

        Args:
            x: Binary selection vector.

        Returns:
            Tuple (value1, value2).
        """
        return float(np.dot(self.values1, x)), float(np.dot(self.values2, x))


@dataclass
class BiObjectiveKnapsackSolution:
    """Solution to a bi-objective knapsack problem.

    Attributes:
        pareto_front: List of (value1, value2) Pareto-optimal points.
        pareto_solutions: List of binary selection vectors.
        n_solutions: Number of Pareto-optimal solutions found.
    """

    pareto_front: list[tuple[float, float]]
    pareto_solutions: list[np.ndarray]
    n_solutions: int

    def __repr__(self) -> str:
        return (f"BiObjectiveKnapsackSolution(n_solutions={self.n_solutions}, "
                f"front={self.pareto_front})")
