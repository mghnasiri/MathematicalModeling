"""
Assignment Problem — Instance and Solution definitions.

Problem notation: LAP (Linear Assignment Problem)

Given an n×n cost matrix C, find a one-to-one assignment of rows
(agents) to columns (tasks) minimizing total cost.

$$\\min \\sum_{i=1}^{n} c_{i, \\sigma(i)}$$

where σ is a permutation of {1, ..., n}.

Complexity:
- Hungarian (Kuhn-Munkres): O(n^3)
- Auction algorithm: O(n^3) average, O(n * max(c_ij)/ε) worst-case

References:
    Kuhn, H.W. (1955). The Hungarian method for the assignment
    problem. Naval Research Logistics Quarterly, 2(1-2), 83-97.
    https://doi.org/10.1002/nav.3800020109

    Munkres, J. (1957). Algorithms for the assignment and
    transportation problems. Journal of the Society for Industrial
    and Applied Mathematics, 5(1), 32-38.
    https://doi.org/10.1137/0105003

    Burkard, R.E. & Çela, E. (1999). Linear assignment problems
    and extensions. In: Du, D.Z. & Pardalos, P.M. (eds) Handbook
    of Combinatorial Optimization. Springer, 75-149.
    https://doi.org/10.1007/978-1-4757-3023-4_2
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class AssignmentInstance:
    """Linear Assignment Problem instance.

    Attributes:
        n: Number of agents (and tasks). Square matrix.
        cost_matrix: n×n cost matrix C[i][j] = cost of assigning agent i to task j.
        name: Optional instance name.
    """

    n: int
    cost_matrix: np.ndarray
    name: str = ""

    def __post_init__(self):
        self.cost_matrix = np.asarray(self.cost_matrix, dtype=float)
        if self.cost_matrix.shape != (self.n, self.n):
            raise ValueError(
                f"cost_matrix shape {self.cost_matrix.shape} != ({self.n}, {self.n})"
            )

    @classmethod
    def random(
        cls, n: int,
        cost_range: tuple[float, float] = (1.0, 100.0),
        seed: int | None = None,
    ) -> AssignmentInstance:
        """Generate a random assignment instance.

        Args:
            n: Number of agents/tasks.
            cost_range: Range for costs.
            seed: Random seed.

        Returns:
            A random AssignmentInstance.
        """
        rng = np.random.default_rng(seed)
        costs = np.round(
            rng.uniform(cost_range[0], cost_range[1], size=(n, n))
        ).astype(float)
        return cls(n=n, cost_matrix=costs, name=f"random_{n}")

    def total_cost(self, assignment: list[int]) -> float:
        """Compute total cost for a given assignment.

        Args:
            assignment: assignment[i] = task assigned to agent i.

        Returns:
            Total cost.
        """
        return sum(self.cost_matrix[i][assignment[i]] for i in range(self.n))


@dataclass
class AssignmentSolution:
    """Solution to a Linear Assignment Problem.

    Attributes:
        assignment: assignment[i] = task assigned to agent i.
        cost: Total cost.
    """

    assignment: list[int]
    cost: float

    def __repr__(self) -> str:
        return f"AssignmentSolution(cost={self.cost:.1f}, assignment={self.assignment})"


def validate_solution(
    instance: AssignmentInstance, solution: AssignmentSolution
) -> tuple[bool, list[str]]:
    """Validate an assignment solution."""
    errors = []

    if len(solution.assignment) != instance.n:
        errors.append(
            f"Assignment length {len(solution.assignment)} != {instance.n}"
        )
        return False, errors

    # Check that assignment is a permutation
    if sorted(solution.assignment) != list(range(instance.n)):
        errors.append("Assignment is not a valid permutation of 0..n-1")

    # Check cost
    actual = instance.total_cost(solution.assignment)
    if abs(actual - solution.cost) > 1e-4:
        errors.append(
            f"Reported cost {solution.cost:.2f} != actual {actual:.2f}"
        )

    return len(errors) == 0, errors


# ── Benchmark instances ──────────────────────────────────────────────────────


def small_assignment_3() -> AssignmentInstance:
    """3×3 assignment. Optimal: agent 0→task 1, 1→0, 2→2; cost=13."""
    costs = np.array([
        [9.0, 2.0, 7.0],
        [6.0, 4.0, 3.0],
        [5.0, 8.0, 1.0],
    ])
    return AssignmentInstance(n=3, cost_matrix=costs, name="small3")


def medium_assignment_5() -> AssignmentInstance:
    """5×5 assignment. Optimal cost = 15."""
    costs = np.array([
        [7.0, 2.0, 1.0, 9.0, 4.0],
        [9.0, 6.0, 9.0, 5.0, 5.0],
        [3.0, 8.0, 3.0, 1.0, 8.0],
        [7.0, 9.0, 4.0, 2.0, 2.0],
        [8.0, 4.0, 7.0, 4.0, 8.0],
    ])
    return AssignmentInstance(n=5, cost_matrix=costs, name="medium5")


if __name__ == "__main__":
    inst = small_assignment_3()
    print(f"{inst.name}: n={inst.n}")
    print(f"  cost matrix:\n{inst.cost_matrix}")
