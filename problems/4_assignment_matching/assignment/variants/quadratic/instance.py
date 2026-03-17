"""
Quadratic Assignment Problem (QAP) — Instance and Solution.

Problem notation: QAP

Given n facilities and n locations, a flow matrix F (flow between
facilities) and a distance matrix D (distance between locations),
assign each facility to a unique location to minimize total cost:
Σ_i Σ_j f_ij * d_π(i),π(j).

Applications: hospital layout, campus planning, keyboard design,
circuit board component placement, office layout.

Complexity: NP-hard (one of the hardest combinatorial problems).

References:
    Koopmans, T.C. & Beckmann, M. (1957). Assignment problems and the
    location of economic activities. Econometrica, 25(1), 53-76.
    https://doi.org/10.2307/1907742

    Burkard, R.E., Dell'Amico, M. & Martello, S. (2009). Assignment
    Problems. SIAM. https://doi.org/10.1137/1.9780898717754
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class QAPInstance:
    """Quadratic Assignment Problem instance.

    Attributes:
        n: Number of facilities/locations.
        flow_matrix: Flow between facilities, shape (n, n).
        distance_matrix: Distance between locations, shape (n, n).
        name: Optional instance name.
    """

    n: int
    flow_matrix: np.ndarray
    distance_matrix: np.ndarray
    name: str = ""

    def __post_init__(self):
        self.flow_matrix = np.asarray(self.flow_matrix, dtype=float)
        self.distance_matrix = np.asarray(self.distance_matrix, dtype=float)

    @classmethod
    def random(
        cls, n: int = 8,
        flow_range: tuple[int, int] = (0, 10),
        dist_range: tuple[int, int] = (1, 20),
        seed: int | None = None,
    ) -> QAPInstance:
        rng = np.random.default_rng(seed)
        flow = rng.integers(flow_range[0], flow_range[1] + 1, size=(n, n)).astype(float)
        np.fill_diagonal(flow, 0)
        flow = (flow + flow.T) / 2
        dist = rng.integers(dist_range[0], dist_range[1] + 1, size=(n, n)).astype(float)
        np.fill_diagonal(dist, 0)
        dist = (dist + dist.T) / 2
        return cls(n=n, flow_matrix=flow, distance_matrix=dist, name=f"random_{n}")

    def objective(self, perm: list[int]) -> float:
        """Compute QAP objective: Σ_i Σ_j f_ij * d_π(i),π(j)."""
        cost = 0.0
        for i in range(self.n):
            for j in range(self.n):
                cost += self.flow_matrix[i][j] * self.distance_matrix[perm[i]][perm[j]]
        return cost


@dataclass
class QAPSolution:
    assignment: list[int]
    cost: float

    def __repr__(self) -> str:
        return f"QAPSolution(cost={self.cost:.1f})"


def validate_solution(
    instance: QAPInstance, solution: QAPSolution
) -> tuple[bool, list[str]]:
    errors = []
    if sorted(solution.assignment) != list(range(instance.n)):
        errors.append("Assignment is not a valid permutation")
    actual = instance.objective(solution.assignment)
    if abs(actual - solution.cost) > 1e-4:
        errors.append(f"Reported cost {solution.cost:.2f} != actual {actual:.2f}")
    return len(errors) == 0, errors


def small_qap_4() -> QAPInstance:
    return QAPInstance(
        n=4,
        flow_matrix=np.array([
            [0, 5, 2, 4],
            [5, 0, 3, 0],
            [2, 3, 0, 1],
            [4, 0, 1, 0],
        ], dtype=float),
        distance_matrix=np.array([
            [0, 1, 2, 3],
            [1, 0, 1, 2],
            [2, 1, 0, 1],
            [3, 2, 1, 0],
        ], dtype=float),
        name="small_4",
    )


if __name__ == "__main__":
    inst = small_qap_4()
    print(f"{inst.name}: n={inst.n}")
    print(f"  Identity perm cost: {inst.objective(list(range(inst.n))):.1f}")
