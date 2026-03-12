"""
Job Sequencing with Deadlines — Instance and Solution definitions.

Problem notation: 1 | d_j | Sigma w_j U_j (weighted number of late jobs)

Given n jobs, each with a processing time, deadline, and profit/weight,
select and sequence a subset of jobs on a single machine to maximize total
profit such that each selected job completes before its deadline.

Complexity: NP-hard (weighted case). The unit-weight case (all profits = 1)
is solvable in O(n log n) using a greedy algorithm.

References:
    Moore, J.M. (1968). An n job, one machine sequencing algorithm for
    minimizing the number of late jobs. Management Science, 15(1), 102-109.
    https://doi.org/10.1287/mnsc.15.1.102

    Lawler, E.L. & Moore, J.M. (1969). A functional equation and its
    application to resource allocation and sequencing problems.
    Management Science, 16(1), 77-84.
    https://doi.org/10.1287/mnsc.16.1.77
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class JobSequencingInstance:
    """Job Sequencing with Deadlines instance.

    Attributes:
        n: Number of jobs.
        processing_times: Processing time per job, shape (n,).
        deadlines: Deadline per job, shape (n,).
        profits: Profit per job, shape (n,).
        name: Optional instance name.
    """

    n: int
    processing_times: np.ndarray
    deadlines: np.ndarray
    profits: np.ndarray
    name: str = ""

    def __post_init__(self):
        self.processing_times = np.asarray(self.processing_times, dtype=float)
        self.deadlines = np.asarray(self.deadlines, dtype=float)
        self.profits = np.asarray(self.profits, dtype=float)

    def is_feasible(self, sequence: list[int]) -> bool:
        """Check if a job sequence completes all jobs by their deadlines."""
        time = 0.0
        for j in sequence:
            time += self.processing_times[j]
            if time > self.deadlines[j] + 1e-10:
                return False
        return True

    def total_profit(self, sequence: list[int]) -> float:
        """Compute total profit of selected jobs."""
        return sum(self.profits[j] for j in sequence)

    @classmethod
    def random(
        cls,
        n: int = 10,
        p_range: tuple[float, float] = (1.0, 10.0),
        profit_range: tuple[float, float] = (1.0, 20.0),
        seed: int | None = None,
    ) -> JobSequencingInstance:
        rng = np.random.default_rng(seed)
        p = np.round(rng.uniform(*p_range, size=n)).astype(float)
        profits = np.round(rng.uniform(*profit_range, size=n)).astype(float)
        # Deadlines: some jobs have tight deadlines, others loose
        deadlines = np.round(
            rng.uniform(p, p * n * 0.5, size=n)
        ).astype(float)
        return cls(n=n, processing_times=p, deadlines=deadlines,
                   profits=profits, name=f"random_js_{n}")

    @classmethod
    def unit_processing(cls, n: int = 5) -> JobSequencingInstance:
        """Unit processing times (p_j = 1). Greedy by deadline is optimal."""
        p = np.ones(n)
        deadlines = np.array([1, 3, 2, 4, 2], dtype=float)[:n]
        profits = np.array([5, 10, 3, 8, 6], dtype=float)[:n]
        return cls(n=n, processing_times=p, deadlines=deadlines,
                   profits=profits, name="unit_5")


@dataclass
class JobSequencingSolution:
    """Solution to Job Sequencing with Deadlines.

    Attributes:
        sequence: Ordered list of selected job indices.
        total_profit: Total profit of selected jobs.
        n_selected: Number of jobs selected.
    """

    sequence: list[int]
    total_profit: float
    n_selected: int

    def __repr__(self) -> str:
        return (
            f"JobSequencingSolution(profit={self.total_profit:.1f}, "
            f"selected={self.n_selected}/{len(self.sequence)})"
        )


if __name__ == "__main__":
    inst = JobSequencingInstance.unit_processing(5)
    print(f"unit_5: n={inst.n}, deadlines={inst.deadlines}, profits={inst.profits}")
