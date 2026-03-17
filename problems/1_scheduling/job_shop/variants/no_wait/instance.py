"""
No-Wait Job Shop Scheduling Problem (NW-JSP) — Instance and Solution.

Problem notation: Jm | no-wait | Cmax

Extends JSP with no-wait constraint: once a job starts, it must be
processed on all its machines without any idle time between consecutive
operations. The start time of a job fully determines all its operation
start times.

Applications: steel manufacturing (continuous casting), chemical
processing, food processing (perishable goods), pharmaceutical
manufacturing.

Complexity: NP-hard (generalizes JSP).

References:
    Mascis, A. & Pacciarelli, D. (2002). Job-shop scheduling with
    blocking and no-wait constraints. European Journal of Operational
    Research, 143(3), 498-517.
    https://doi.org/10.1016/S0377-2217(01)00338-1

    Sahni, S. & Cho, Y. (1979). Complexity of scheduling shops with
    no wait in process. Mathematics of Operations Research, 4(4), 448-457.
    https://doi.org/10.1287/moor.4.4.448
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class NWJSPInstance:
    """No-Wait Job Shop instance.

    Attributes:
        n: Number of jobs.
        m: Number of machines.
        operations: operations[j] = [(machine, duration), ...] for job j.
        name: Optional instance name.
    """

    n: int
    m: int
    operations: list[list[tuple[int, int]]]
    name: str = ""

    @classmethod
    def random(
        cls,
        n: int = 6,
        m: int = 3,
        duration_range: tuple[int, int] = (1, 10),
        seed: int | None = None,
    ) -> NWJSPInstance:
        """Generate a random NW-JSP instance."""
        rng = np.random.default_rng(seed)
        operations = []
        for j in range(n):
            machines = list(rng.permutation(m))
            ops = [
                (int(machines[k]), int(rng.integers(duration_range[0], duration_range[1] + 1)))
                for k in range(m)
            ]
            operations.append(ops)
        return cls(n=n, m=m, operations=operations, name=f"random_{n}_{m}")

    def job_duration(self, j: int) -> int:
        """Total processing time of job j (sum of all operations)."""
        return sum(d for _, d in self.operations[j])

    def compute_schedule(self, job_start_times: list[float]) -> dict:
        """From job start times, compute all operation start/end times.

        Returns dict with:
            op_starts[j][k] = start time of operation k of job j
            op_ends[j][k] = end time of operation k of job j
            makespan = max completion time
        """
        op_starts = []
        op_ends = []
        for j in range(self.n):
            starts = []
            ends = []
            t = job_start_times[j]
            for _, dur in self.operations[j]:
                starts.append(t)
                ends.append(t + dur)
                t += dur
            op_starts.append(starts)
            op_ends.append(ends)
        makespan = max(op_ends[j][-1] for j in range(self.n))
        return {"op_starts": op_starts, "op_ends": op_ends, "makespan": makespan}

    def makespan(self, job_start_times: list[float]) -> float:
        """Compute makespan from job start times."""
        return self.compute_schedule(job_start_times)["makespan"]

    def is_feasible(self, job_start_times: list[float]) -> bool:
        """Check if no two operations on the same machine overlap."""
        sched = self.compute_schedule(job_start_times)
        for machine in range(self.m):
            intervals = []
            for j in range(self.n):
                for k, (mach, dur) in enumerate(self.operations[j]):
                    if mach == machine:
                        intervals.append(
                            (sched["op_starts"][j][k], sched["op_ends"][j][k])
                        )
            intervals.sort()
            for i in range(len(intervals) - 1):
                if intervals[i][1] > intervals[i + 1][0] + 1e-10:
                    return False
        return True


@dataclass
class NWJSPSolution:
    """Solution to a NW-JSP instance.

    Attributes:
        job_start_times: Start time for each job.
        makespan: Maximum completion time.
    """

    job_start_times: list[float]
    makespan: float

    def __repr__(self) -> str:
        return f"NWJSPSolution(makespan={self.makespan:.1f})"


def validate_solution(
    instance: NWJSPInstance, solution: NWJSPSolution
) -> tuple[bool, list[str]]:
    """Validate a NW-JSP solution."""
    errors = []

    if len(solution.job_start_times) != instance.n:
        errors.append(f"Expected {instance.n} start times, got {len(solution.job_start_times)}")
        return False, errors

    for j in range(instance.n):
        if solution.job_start_times[j] < -1e-10:
            errors.append(f"Job {j}: negative start time {solution.job_start_times[j]:.2f}")

    if not instance.is_feasible(solution.job_start_times):
        errors.append("Machine conflicts detected (overlapping operations)")

    actual_ms = instance.makespan(solution.job_start_times)
    if abs(actual_ms - solution.makespan) > 1e-4:
        errors.append(
            f"Reported makespan {solution.makespan:.2f} != actual {actual_ms:.2f}"
        )

    return len(errors) == 0, errors


# ── Benchmark instances ──────────────────────────────────────────────────────


def small_nwjsp_3_3() -> NWJSPInstance:
    """3 jobs, 3 machines."""
    return NWJSPInstance(
        n=3, m=3,
        operations=[
            [(0, 3), (1, 2), (2, 4)],  # job 0: M0->M1->M2
            [(1, 4), (2, 3), (0, 2)],  # job 1: M1->M2->M0
            [(2, 2), (0, 3), (1, 5)],  # job 2: M2->M0->M1
        ],
        name="small_3_3",
    )


if __name__ == "__main__":
    inst = small_nwjsp_3_3()
    print(f"{inst.name}: {inst.n} jobs, {inst.m} machines")
    starts = [0.0, 5.0, 10.0]
    print(f"  Start times {starts}: makespan={inst.makespan(starts):.1f}")
    print(f"  Feasible: {inst.is_feasible(starts)}")
