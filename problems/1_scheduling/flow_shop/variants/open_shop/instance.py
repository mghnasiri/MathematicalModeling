"""
Open Shop Scheduling (Om || Cmax) — Instance and Solution.

Each job must be processed on all machines, but the order of operations
within each job is free (no precedence constraints). The objective is
to minimize the makespan.

Complexity: NP-hard for m >= 3 (Gonzalez & Sahni, 1976).
O2||Cmax is polynomial.

References:
    Gonzalez, T. & Sahni, S. (1976). Open shop scheduling to minimize
    finish time. Journal of the ACM, 23(4), 665-679.
    https://doi.org/10.1145/321978.321985

    Gueret, C. & Prins, C. (1999). A new lower bound for the open-shop
    problem. Annals of Operations Research, 92, 165-183.
    https://doi.org/10.1023/A:1018930613891
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class OpenShopInstance:
    """Open Shop Scheduling instance.

    Attributes:
        n: Number of jobs.
        m: Number of machines.
        processing_times: Processing times, shape (n, m).
        name: Optional instance name.
    """

    n: int
    m: int
    processing_times: np.ndarray
    name: str = ""

    def __post_init__(self):
        self.processing_times = np.asarray(self.processing_times, dtype=float)

    def makespan(self, schedule: list[list[tuple[int, float]]]) -> float:
        """Compute makespan from a schedule.

        Args:
            schedule: schedule[j] = [(machine, start_time), ...] for each job.

        Returns:
            Makespan.
        """
        cmax = 0.0
        for j in range(self.n):
            for mach, st in schedule[j]:
                cmax = max(cmax, st + self.processing_times[j][mach])
        return cmax

    @classmethod
    def random(
        cls,
        n: int = 4,
        m: int = 3,
        time_range: tuple[int, int] = (5, 30),
        seed: int | None = None,
    ) -> OpenShopInstance:
        rng = np.random.default_rng(seed)
        pt = rng.integers(time_range[0], time_range[1] + 1, size=(n, m)).astype(float)
        return cls(n=n, m=m, processing_times=pt, name=f"open_shop_{n}x{m}")


@dataclass
class OpenShopSolution:
    """Open Shop solution.

    Attributes:
        schedule: schedule[j] = [(machine, start_time), ...] for each job.
        makespan: Makespan value.
    """

    schedule: list[list[tuple[int, float]]]
    makespan: float

    def __repr__(self) -> str:
        return f"OpenShopSolution(Cmax={self.makespan:.1f})"


def validate_solution(
    instance: OpenShopInstance, solution: OpenShopSolution
) -> tuple[bool, list[str]]:
    errors = []

    # Check each job processes on all machines
    for j in range(instance.n):
        machines = [m for m, _ in solution.schedule[j]]
        if sorted(machines) != list(range(instance.m)):
            errors.append(f"Job {j}: not all machines covered")

    # Check no overlaps on same machine
    machine_intervals = {}
    for j in range(instance.n):
        for mach, st in solution.schedule[j]:
            dur = instance.processing_times[j][mach]
            machine_intervals.setdefault(mach, []).append((st, st + dur, j))

    for mach, intervals in machine_intervals.items():
        intervals.sort()
        for i in range(len(intervals) - 1):
            if intervals[i][1] > intervals[i + 1][0] + 1e-6:
                errors.append(
                    f"Machine {mach}: overlap between job {intervals[i][2]} "
                    f"and job {intervals[i+1][2]}")

    # Check no overlaps for same job (job can only be on one machine at a time)
    for j in range(instance.n):
        ops = [(st, st + instance.processing_times[j][m], m)
               for m, st in solution.schedule[j]]
        ops.sort()
        for i in range(len(ops) - 1):
            if ops[i][1] > ops[i + 1][0] + 1e-6:
                errors.append(
                    f"Job {j}: overlap on machines {ops[i][2]} and {ops[i+1][2]}")

    # Check makespan
    actual = instance.makespan(solution.schedule)
    if abs(actual - solution.makespan) > 1e-2:
        errors.append(f"Makespan: {solution.makespan:.2f} != {actual:.2f}")

    return len(errors) == 0, errors


def small_os_3x3() -> OpenShopInstance:
    return OpenShopInstance(
        n=3, m=3,
        processing_times=np.array([
            [10, 5, 8],
            [6, 12, 7],
            [8, 9, 11],
        ], dtype=float),
        name="small_os_3x3",
    )


if __name__ == "__main__":
    inst = small_os_3x3()
    print(f"{inst.name}: n={inst.n}, m={inst.m}")
