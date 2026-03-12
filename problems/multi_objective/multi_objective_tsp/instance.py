"""Multi-Objective Traveling Salesman Problem (MOTSP).

Given n cities and k distance/cost matrices, find tours (Hamiltonian cycles)
that are Pareto-optimal across all objectives.

Complexity: NP-hard (generalizes single-objective TSP).

References:
    Jaszkiewicz, A. (2002). Genetic local search for multi-objective
    combinatorial optimization. European Journal of Operational Research,
    137(1), 50-71.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MultiObjectiveTSPInstance:
    """Multi-objective TSP instance.

    Attributes:
        n: Number of cities.
        n_objectives: Number of objectives.
        distance_matrices: List of n x n distance matrices, one per objective.
    """

    n: int
    n_objectives: int
    distance_matrices: list[np.ndarray]

    @classmethod
    def random(cls, n: int = 8, n_objectives: int = 2,
               seed: int | None = None) -> MultiObjectiveTSPInstance:
        """Generate a random multi-objective TSP instance.

        Args:
            n: Number of cities.
            n_objectives: Number of objectives.
            seed: Random seed.

        Returns:
            A random MultiObjectiveTSPInstance.
        """
        rng = np.random.default_rng(seed)
        matrices = []
        for _ in range(n_objectives):
            positions = rng.uniform(0, 100, size=(n, 2))
            diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
            dist = np.sqrt((diff ** 2).sum(axis=2))
            np.fill_diagonal(dist, np.inf)
            matrices.append(dist)
        return cls(n=n, n_objectives=n_objectives, distance_matrices=matrices)

    def tour_cost(self, tour: list[int], objective: int) -> float:
        """Compute tour cost for a specific objective.

        Args:
            tour: Permutation of city indices.
            objective: Objective index.

        Returns:
            Total tour cost.
        """
        D = self.distance_matrices[objective]
        cost = 0.0
        for i in range(len(tour)):
            cost += D[tour[i], tour[(i + 1) % len(tour)]]
        return cost

    def evaluate(self, tour: list[int]) -> tuple[float, ...]:
        """Evaluate tour across all objectives.

        Args:
            tour: Permutation of city indices.

        Returns:
            Tuple of costs, one per objective.
        """
        return tuple(self.tour_cost(tour, k)
                     for k in range(self.n_objectives))


@dataclass
class MultiObjectiveTSPSolution:
    """Solution to a multi-objective TSP.

    Attributes:
        pareto_front: List of objective tuples.
        pareto_tours: List of tours (permutations).
        n_solutions: Number of Pareto solutions.
    """

    pareto_front: list[tuple[float, ...]]
    pareto_tours: list[list[int]]
    n_solutions: int

    def __repr__(self) -> str:
        return (f"MultiObjectiveTSPSolution(n_solutions={self.n_solutions})")
