"""
Unrelated Parallel Machine with Total Tardiness (Rm || ΣTj) — Instance and Solution.

n jobs assigned to m unrelated machines. Processing time of job j on
machine i is p_ij (varies by both job and machine). Each job has a due
date. Minimize total tardiness ΣTj where Tj = max(0, Cj - dj).

Complexity: Strongly NP-hard.

References:
    Pinedo, M. (2016). Scheduling: Theory, Algorithms, and Systems.
    5th ed. Springer. https://doi.org/10.1007/978-3-319-26580-3

    Weng, M.X., Lu, J. & Ren, H. (2001). Unrelated parallel machine
    scheduling with setup consideration and a total weighted completion
    time objective. International Journal of Production Economics, 70(3),
    215-226. https://doi.org/10.1016/S0925-5273(00)00066-9
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class RmTardinessInstance:
    """Unrelated Parallel Machine with Tardiness instance.

    Attributes:
        n: Number of jobs.
        m: Number of machines.
        processing_times: Processing time matrix, shape (m, n).
            processing_times[i][j] = time of job j on machine i.
        due_dates: Due date for each job, shape (n,).
        name: Optional instance name.
    """

    n: int
    m: int
    processing_times: np.ndarray
    due_dates: np.ndarray
    name: str = ""

    def __post_init__(self):
        self.processing_times = np.asarray(self.processing_times, dtype=float)
        self.due_dates = np.asarray(self.due_dates, dtype=float)

    def total_tardiness(self, assignment: list[int], sequence: list[list[int]]) -> float:
        """Compute total tardiness.

        Args:
            assignment: assignment[j] = machine for job j.
            sequence: sequence[i] = list of jobs in processing order on machine i.

        Returns:
            Total tardiness.
        """
        total = 0.0
        for i in range(self.m):
            time_now = 0.0
            for j in sequence[i]:
                time_now += self.processing_times[i][j]
                tardiness = max(0.0, time_now - self.due_dates[j])
                total += tardiness
        return total

    @classmethod
    def random(
        cls,
        n: int = 8,
        m: int = 3,
        seed: int | None = None,
    ) -> RmTardinessInstance:
        rng = np.random.default_rng(seed)
        pt = rng.integers(5, 30, size=(m, n)).astype(float)
        # Due dates: based on average processing
        avg_pt = pt.mean(axis=0)
        due_dates = np.cumsum(np.sort(avg_pt)) * rng.uniform(0.8, 1.5, size=n)
        return cls(n=n, m=m, processing_times=pt, due_dates=due_dates,
                   name=f"random_rm_tard_{n}x{m}")


@dataclass
class RmTardinessSolution:
    """Solution for Rm||ΣTj.

    Attributes:
        assignment: assignment[j] = machine for job j.
        sequence: sequence[i] = job order on machine i.
        total_tardiness: Total tardiness value.
    """

    assignment: list[int]
    sequence: list[list[int]]
    total_tardiness: float

    def __repr__(self) -> str:
        return f"RmTardSolution(ΣTj={self.total_tardiness:.1f})"


def validate_solution(
    instance: RmTardinessInstance, solution: RmTardinessSolution
) -> tuple[bool, list[str]]:
    errors = []

    # All jobs assigned
    all_jobs = []
    for seq in solution.sequence:
        all_jobs.extend(seq)
    if sorted(all_jobs) != list(range(instance.n)):
        errors.append("Not all jobs scheduled exactly once")

    # Assignment consistency
    for i in range(instance.m):
        for j in solution.sequence[i]:
            if solution.assignment[j] != i:
                errors.append(f"Job {j}: assignment={solution.assignment[j]} but in seq[{i}]")

    # Tardiness check
    actual = instance.total_tardiness(solution.assignment, solution.sequence)
    if abs(actual - solution.total_tardiness) > 1e-2:
        errors.append(f"Tardiness: {solution.total_tardiness:.2f} != {actual:.2f}")

    return len(errors) == 0, errors


def small_rm_tard_6x2() -> RmTardinessInstance:
    return RmTardinessInstance(
        n=6, m=2,
        processing_times=np.array([
            [10, 8, 15, 6, 12, 9],
            [12, 10, 8, 14, 7, 11],
        ], dtype=float),
        due_dates=np.array([15, 20, 30, 12, 25, 18], dtype=float),
        name="small_rm_tard_6x2",
    )


if __name__ == "__main__":
    inst = small_rm_tard_6x2()
    print(f"{inst.name}: n={inst.n}, m={inst.m}")
