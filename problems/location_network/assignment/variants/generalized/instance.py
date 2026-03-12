"""
Generalized Assignment Problem (GAP) — Instance and Solution.

Assign n jobs to m agents. Each agent i has capacity b_i. Assigning job j
to agent i consumes a_ij resources and yields cost c_ij. Minimize total
cost subject to capacity constraints.

Complexity: NP-hard (strongly). Even feasibility is NP-complete.

References:
    Ross, G.T. & Soland, R.M. (1975). A branch and bound algorithm for
    the generalized assignment problem. Mathematical Programming, 8(1),
    91-103. https://doi.org/10.1007/BF01580430

    Martello, S. & Toth, P. (1990). Knapsack Problems: Algorithms and
    Computer Implementations. Wiley. ISBN 978-0471924203.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class GAPInstance:
    """Generalized Assignment Problem instance.

    Attributes:
        n: Number of jobs.
        m: Number of agents.
        cost: Cost matrix, shape (m, n). cost[i][j] = cost of assigning j to i.
        resource: Resource matrix, shape (m, n). resource[i][j] = resources consumed.
        capacity: Agent capacities, shape (m,).
        name: Optional instance name.
    """

    n: int
    m: int
    cost: np.ndarray
    resource: np.ndarray
    capacity: np.ndarray
    name: str = ""

    def __post_init__(self):
        self.cost = np.asarray(self.cost, dtype=float)
        self.resource = np.asarray(self.resource, dtype=float)
        self.capacity = np.asarray(self.capacity, dtype=float)

    @classmethod
    def random(
        cls,
        n: int = 8,
        m: int = 3,
        cost_range: tuple[int, int] = (5, 25),
        resource_range: tuple[int, int] = (3, 15),
        seed: int | None = None,
    ) -> GAPInstance:
        rng = np.random.default_rng(seed)
        cost = rng.integers(cost_range[0], cost_range[1] + 1, size=(m, n)).astype(float)
        resource = rng.integers(resource_range[0], resource_range[1] + 1, size=(m, n)).astype(float)
        # Set capacities so problem is feasible
        capacity = np.ceil(resource.sum(axis=1) * 0.6 / m * n / m + resource.max(axis=1))
        return cls(n=n, m=m, cost=cost, resource=resource, capacity=capacity,
                   name=f"random_{n}x{m}")

    def total_cost(self, assignment: list[int]) -> float:
        """Total cost of an assignment. assignment[j] = agent for job j."""
        return float(sum(self.cost[assignment[j]][j] for j in range(self.n)))


@dataclass
class GAPSolution:
    """GAP solution.

    Attributes:
        assignment: assignment[j] = agent index for job j.
        total_cost: Total assignment cost.
    """

    assignment: list[int]
    total_cost: float

    def __repr__(self) -> str:
        return f"GAPSolution(cost={self.total_cost:.1f})"


def validate_solution(
    instance: GAPInstance, solution: GAPSolution
) -> tuple[bool, list[str]]:
    errors = []
    if len(solution.assignment) != instance.n:
        errors.append(f"Expected {instance.n} assignments, got {len(solution.assignment)}")
        return False, errors

    for j, a in enumerate(solution.assignment):
        if a < 0 or a >= instance.m:
            errors.append(f"Job {j}: invalid agent {a}")

    # Check capacity
    loads = np.zeros(instance.m)
    for j, a in enumerate(solution.assignment):
        loads[a] += instance.resource[a][j]
    for i in range(instance.m):
        if loads[i] > instance.capacity[i] + 1e-6:
            errors.append(f"Agent {i}: load {loads[i]:.2f} > capacity {instance.capacity[i]:.2f}")

    actual = instance.total_cost(solution.assignment)
    if abs(actual - solution.total_cost) > 1e-4:
        errors.append(f"Cost: {solution.total_cost:.2f} != {actual:.2f}")

    return len(errors) == 0, errors


def small_gap_6x3() -> GAPInstance:
    return GAPInstance(
        n=6, m=3,
        cost=np.array([
            [8, 12, 5, 15, 7, 10],
            [10, 6, 14, 8, 11, 9],
            [7, 11, 9, 12, 6, 13],
        ], dtype=float),
        resource=np.array([
            [5, 7, 3, 8, 4, 6],
            [6, 4, 8, 5, 7, 3],
            [4, 6, 5, 7, 3, 8],
        ], dtype=float),
        capacity=np.array([18, 17, 16], dtype=float),
        name="small_6x3",
    )


if __name__ == "__main__":
    inst = small_gap_6x3()
    print(f"{inst.name}: n={inst.n}, m={inst.m}")
