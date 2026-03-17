"""
Asymmetric Traveling Salesman Problem (ATSP) — Instance and Solution.

Given n cities and a directed distance matrix D where d(i,j) ≠ d(j,i)
in general, find the shortest directed Hamiltonian cycle.

Complexity: NP-hard. Harder to approximate than symmetric TSP —
best known is O(log n / log log n) (Asadpour et al., 2017).

References:
    Kanellakis, P.C. & Papadimitriou, C.H. (1980). Local search for the
    asymmetric traveling salesman problem. Operations Research, 28(5),
    1086-1099. https://doi.org/10.1287/opre.28.5.1086

    Asadpour, A., Goemans, M.X., Madry, A., Oveis Gharan, S. & Saberi, A.
    (2017). An O(log n / log log n)-approximation algorithm for the
    asymmetric traveling salesman problem. Operations Research, 65(4),
    1043-1061. https://doi.org/10.1287/opre.2017.1603
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class ATSPInstance:
    """Asymmetric TSP instance.

    Attributes:
        n: Number of cities.
        dist_matrix: Directed distance matrix, shape (n, n).
        name: Optional instance name.
    """

    n: int
    dist_matrix: np.ndarray
    name: str = ""

    def __post_init__(self):
        self.dist_matrix = np.asarray(self.dist_matrix, dtype=float)

    @classmethod
    def random(
        cls,
        n: int = 8,
        dist_range: tuple[int, int] = (1, 50),
        seed: int | None = None,
    ) -> ATSPInstance:
        rng = np.random.default_rng(seed)
        dm = rng.integers(dist_range[0], dist_range[1] + 1, size=(n, n)).astype(float)
        np.fill_diagonal(dm, 0.0)
        return cls(n=n, dist_matrix=dm, name=f"random_{n}")

    def tour_cost(self, tour: list[int]) -> float:
        """Cost of a directed tour."""
        cost = 0.0
        for i in range(len(tour)):
            cost += self.dist_matrix[tour[i]][tour[(i + 1) % len(tour)]]
        return cost


@dataclass
class ATSPSolution:
    """ATSP solution.

    Attributes:
        tour: Ordered city permutation.
        cost: Total directed tour cost.
    """

    tour: list[int]
    cost: float

    def __repr__(self) -> str:
        return f"ATSPSolution(cost={self.cost:.1f})"


def validate_solution(
    instance: ATSPInstance, solution: ATSPSolution
) -> tuple[bool, list[str]]:
    errors = []
    if sorted(solution.tour) != list(range(instance.n)):
        errors.append("Tour is not a valid permutation")
    actual = instance.tour_cost(solution.tour)
    if abs(actual - solution.cost) > 1e-4:
        errors.append(f"Cost mismatch: {solution.cost:.2f} != {actual:.2f}")
    return len(errors) == 0, errors


def small_atsp_5() -> ATSPInstance:
    return ATSPInstance(
        n=5,
        dist_matrix=np.array([
            [0, 10, 25, 30, 15],
            [20, 0, 35, 10, 25],
            [15, 30, 0, 20, 10],
            [25, 15, 10, 0, 35],
            [10, 20, 15, 25, 0],
        ], dtype=float),
        name="small_5",
    )


if __name__ == "__main__":
    inst = small_atsp_5()
    print(f"{inst.name}: n={inst.n}")
    print(f"  Identity tour cost: {inst.tour_cost(list(range(inst.n))):.1f}")
