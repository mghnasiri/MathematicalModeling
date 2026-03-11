"""
Capacitated p-Median Problem (CPMP) — Instance and Solution.

Extends p-Median with facility capacities. Each open facility can serve
at most a given total demand. Minimize total weighted distance subject to
capacity constraints.

Complexity: NP-hard (generalizes p-Median).

References:
    Mulvey, J.M. & Beck, M.P. (1984). Solving capacitated clustering
    problems. European Journal of Operational Research, 18(3), 339-348.
    https://doi.org/10.1016/0377-2217(84)90155-3

    Lorena, L.A.N. & Senne, E.L.F. (2004). A column generation approach
    to capacitated p-median problems. Computers & Operations Research,
    31(6), 863-876. https://doi.org/10.1016/S0305-0548(03)00039-X
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class CPMedianInstance:
    """Capacitated p-Median instance.

    Attributes:
        n: Number of customers.
        m: Number of candidate facilities.
        p: Number of facilities to open.
        demands: Customer demands, shape (n,).
        capacities: Facility capacities, shape (m,).
        distance_matrix: Distance from facility i to customer j, shape (m, n).
        name: Optional instance name.
    """

    n: int
    m: int
    p: int
    demands: np.ndarray
    capacities: np.ndarray
    distance_matrix: np.ndarray
    name: str = ""

    def __post_init__(self):
        self.demands = np.asarray(self.demands, dtype=float)
        self.capacities = np.asarray(self.capacities, dtype=float)
        self.distance_matrix = np.asarray(self.distance_matrix, dtype=float)

    @classmethod
    def random(
        cls,
        n: int = 10,
        m: int = 5,
        p: int = 2,
        seed: int | None = None,
    ) -> CPMedianInstance:
        rng = np.random.default_rng(seed)
        cust_coords = rng.uniform(0, 100, size=(n, 2))
        fac_coords = rng.uniform(0, 100, size=(m, 2))
        dist_matrix = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                dist_matrix[i][j] = np.linalg.norm(fac_coords[i] - cust_coords[j])
        demands = rng.integers(5, 20, size=n).astype(float)
        total_demand = demands.sum()
        # Capacity: total demand spread across p facilities + slack
        cap_per = total_demand / p * 1.3
        capacities = np.full(m, cap_per)
        return cls(n=n, m=m, p=p, demands=demands, capacities=capacities,
                   distance_matrix=dist_matrix, name=f"random_cpmp_{n}_{m}_{p}")


@dataclass
class CPMedianSolution:
    """Capacitated p-Median solution.

    Attributes:
        open_facilities: List of opened facility indices.
        assignments: assignments[j] = facility serving customer j.
        total_cost: Total weighted distance.
    """

    open_facilities: list[int]
    assignments: list[int]
    total_cost: float

    def __repr__(self) -> str:
        return f"CPMedianSolution(p={len(self.open_facilities)}, cost={self.total_cost:.1f})"


def validate_solution(
    instance: CPMedianInstance, solution: CPMedianSolution
) -> tuple[bool, list[str]]:
    errors = []

    if len(solution.open_facilities) != instance.p:
        errors.append(f"Opened {len(solution.open_facilities)} != p={instance.p}")

    if len(solution.assignments) != instance.n:
        errors.append(f"Expected {instance.n} assignments, got {len(solution.assignments)}")
        return False, errors

    # Each customer assigned to open facility
    open_set = set(solution.open_facilities)
    for j, fac in enumerate(solution.assignments):
        if fac not in open_set:
            errors.append(f"Customer {j}: assigned to closed facility {fac}")

    # Capacity constraints
    loads = np.zeros(instance.m)
    for j, fac in enumerate(solution.assignments):
        loads[fac] += instance.demands[j]
    for i in solution.open_facilities:
        if loads[i] > instance.capacities[i] + 1e-6:
            errors.append(f"Facility {i}: load {loads[i]:.1f} > cap {instance.capacities[i]:.1f}")

    # Cost
    actual = sum(instance.demands[j] * instance.distance_matrix[solution.assignments[j]][j]
                 for j in range(instance.n))
    if abs(actual - solution.total_cost) > 1e-2:
        errors.append(f"Cost: {solution.total_cost:.2f} != {actual:.2f}")

    return len(errors) == 0, errors


def small_cpmp_6() -> CPMedianInstance:
    cust = np.array([[10, 10], [20, 80], [50, 50], [80, 20], [70, 70], [40, 30]], dtype=float)
    fac = np.array([[15, 45], [60, 60], [75, 25]], dtype=float)
    dist = np.zeros((3, 6))
    for i in range(3):
        for j in range(6):
            dist[i][j] = np.linalg.norm(fac[i] - cust[j])
    return CPMedianInstance(
        n=6, m=3, p=2,
        demands=np.array([10, 8, 12, 15, 7, 11], dtype=float),
        capacities=np.array([40, 35, 45], dtype=float),
        distance_matrix=dist,
        name="small_cpmp_6",
    )


if __name__ == "__main__":
    inst = small_cpmp_6()
    print(f"{inst.name}: n={inst.n}, m={inst.m}, p={inst.p}")
