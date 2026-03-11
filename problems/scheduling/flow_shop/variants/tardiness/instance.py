"""
Flow Shop with Total Weighted Tardiness (Fm | prmu | ΣwjTj).

Standard permutation flow shop but with due dates and weights.
Objective: minimize sum of weighted tardiness instead of makespan.

Complexity: NP-hard (even for single machine).

References:
    Kim, Y.D. (1993). Heuristics for flowshop scheduling problems
    minimizing mean tardiness. Journal of the Operational Research
    Society, 44(1), 19-28. https://doi.org/10.1057/jors.1993.3

    Vallada, E. & Ruiz, R. (2010). Genetic algorithms with path
    relinking for the minimum tardiness permutation flowshop problem.
    Omega, 38(1-2), 57-67.
    https://doi.org/10.1016/j.omega.2009.04.002
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class TardinessFlowShopInstance:
    """Tardiness Flow Shop instance.

    Attributes:
        n: Number of jobs.
        m: Number of machines.
        processing_times: Shape (n, m).
        due_dates: Due date per job, shape (n,).
        weights: Weight per job, shape (n,).
        name: Optional instance name.
    """

    n: int
    m: int
    processing_times: np.ndarray
    due_dates: np.ndarray
    weights: np.ndarray
    name: str = ""

    def __post_init__(self):
        self.processing_times = np.asarray(self.processing_times, dtype=float)
        self.due_dates = np.asarray(self.due_dates, dtype=float)
        self.weights = np.asarray(self.weights, dtype=float)

    @classmethod
    def random(
        cls,
        n: int = 8,
        m: int = 3,
        proc_range: tuple[int, int] = (1, 20),
        seed: int | None = None,
    ) -> TardinessFlowShopInstance:
        rng = np.random.default_rng(seed)
        proc = rng.integers(proc_range[0], proc_range[1] + 1, size=(n, m)).astype(float)
        total = proc.sum(axis=1)
        # Due dates: somewhere around total processing time
        due = total * rng.uniform(0.8, 2.0, size=n)
        weights = rng.integers(1, 10, size=n).astype(float)
        return cls(n=n, m=m, processing_times=proc, due_dates=due,
                   weights=weights, name=f"random_{n}x{m}")

    def completion_times(self, perm: list[int]) -> np.ndarray:
        """Compute completion times on last machine for each job."""
        nj = len(perm)
        C = np.zeros((nj, self.m))
        for i, job in enumerate(perm):
            for k in range(self.m):
                prev_job = C[i - 1][k] if i > 0 else 0.0
                prev_machine = C[i][k - 1] if k > 0 else 0.0
                C[i][k] = max(prev_job, prev_machine) + self.processing_times[job][k]
        return np.array([C[i][-1] for i, _ in enumerate(perm)])

    def total_weighted_tardiness(self, perm: list[int]) -> float:
        """Compute ΣwjTj for a permutation."""
        ct = self.completion_times(perm)
        twt = 0.0
        for i, job in enumerate(perm):
            tardiness = max(0.0, ct[i] - self.due_dates[job])
            twt += self.weights[job] * tardiness
        return twt


@dataclass
class TardinessFlowShopSolution:
    """Tardiness Flow Shop solution.

    Attributes:
        permutation: Job processing order.
        total_weighted_tardiness: ΣwjTj objective.
    """

    permutation: list[int]
    total_weighted_tardiness: float

    def __repr__(self) -> str:
        return f"TardinessFSSolution(TWT={self.total_weighted_tardiness:.1f})"


def validate_solution(
    instance: TardinessFlowShopInstance, solution: TardinessFlowShopSolution
) -> tuple[bool, list[str]]:
    errors = []
    if sorted(solution.permutation) != list(range(instance.n)):
        errors.append("Invalid permutation")
    actual = instance.total_weighted_tardiness(solution.permutation)
    if abs(actual - solution.total_weighted_tardiness) > 1e-4:
        errors.append(f"TWT: {solution.total_weighted_tardiness:.2f} != {actual:.2f}")
    return len(errors) == 0, errors


def small_tfs_4x3() -> TardinessFlowShopInstance:
    return TardinessFlowShopInstance(
        n=4, m=3,
        processing_times=np.array([
            [5, 8, 3],
            [3, 6, 7],
            [8, 4, 5],
            [6, 7, 2],
        ], dtype=float),
        due_dates=np.array([25, 30, 20, 35], dtype=float),
        weights=np.array([3, 1, 4, 2], dtype=float),
        name="small_4x3",
    )


if __name__ == "__main__":
    inst = small_tfs_4x3()
    perm = list(range(inst.n))
    print(f"Identity TWT: {inst.total_weighted_tardiness(perm):.1f}")
