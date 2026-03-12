"""
Flexible Job Shop with Tardiness (FJm || ΣwjTj) — Instance and Solution.

Combines FJSP flexibility (machine assignment) with tardiness minimization.
Each operation can be processed on any machine from an eligible set, with
machine-dependent processing times. Objective: minimize total weighted
tardiness.

Complexity: NP-hard (strongly; generalizes both FJSP and 1||ΣwjTj).

References:
    Brandimarte, P. (1993). Routing and scheduling in a flexible job shop
    by tabu search. Annals of Operations Research, 41(3), 157-183.
    https://doi.org/10.1007/BF02023073

    Mastrolilli, M. & Gambardella, L.M. (2000). Effective neighbourhood
    functions for the flexible job shop problem. Journal of Scheduling,
    3(1), 3-20. https://doi.org/10.1002/(SICI)1099-1425(200001/02)3:1<3::AID-JOS32>3.0.CO;2-Y
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class FlexTardJSPInstance:
    """Flexible Job Shop with Tardiness instance.

    Attributes:
        n: Number of jobs.
        m: Number of machines.
        operations: operations[j] = [[(machine, duration), ...], ...]
            Each operation is a list of (machine, duration) alternatives.
        due_dates: Due date for each job, shape (n,).
        weights: Weight for each job, shape (n,).
        name: Optional instance name.
    """

    n: int
    m: int
    operations: list[list[list[tuple[int, int]]]]
    due_dates: np.ndarray
    weights: np.ndarray
    name: str = ""

    def __post_init__(self):
        self.due_dates = np.asarray(self.due_dates, dtype=float)
        self.weights = np.asarray(self.weights, dtype=float)

    def num_operations(self, j: int) -> int:
        return len(self.operations[j])

    def total_operations(self) -> int:
        return sum(self.num_operations(j) for j in range(self.n))

    @classmethod
    def random(
        cls,
        n: int = 4,
        m: int = 3,
        ops_per_job: int = 3,
        seed: int | None = None,
    ) -> FlexTardJSPInstance:
        rng = np.random.default_rng(seed)
        operations = []
        total_proc = np.zeros(n)
        for j in range(n):
            job_ops = []
            for _ in range(ops_per_job):
                # Each operation has 1 to m eligible machines
                num_eligible = rng.integers(1, m + 1)
                eligible = rng.choice(m, size=num_eligible, replace=False)
                alts = [(int(mach), int(rng.integers(5, 25)))
                        for mach in eligible]
                job_ops.append(alts)
                total_proc[j] += min(d for _, d in alts)
            operations.append(job_ops)

        # Due dates: tight factor
        due_dates = total_proc * rng.uniform(1.2, 2.0, size=n)
        weights = rng.uniform(1.0, 5.0, size=n).round(1)

        return cls(n=n, m=m, operations=operations, due_dates=due_dates,
                   weights=weights, name=f"random_ftjsp_{n}x{m}")


@dataclass
class FlexTardJSPSolution:
    """Flexible JSP with Tardiness solution.

    Attributes:
        machine_assignments: machine_assignments[j][o] = machine for op o of job j.
        start_times: start_times[j][o] = start time for op o of job j.
        total_weighted_tardiness: ΣwjTj.
    """

    machine_assignments: list[list[int]]
    start_times: list[list[float]]
    total_weighted_tardiness: float

    def __repr__(self) -> str:
        return f"FlexTardJSPSolution(ΣwjTj={self.total_weighted_tardiness:.1f})"


def validate_solution(
    instance: FlexTardJSPInstance, solution: FlexTardJSPSolution
) -> tuple[bool, list[str]]:
    errors = []

    # Check dimensions
    if len(solution.machine_assignments) != instance.n:
        errors.append(f"Expected {instance.n} jobs, got {len(solution.machine_assignments)}")
        return False, errors

    # Check each operation
    for j in range(instance.n):
        if len(solution.machine_assignments[j]) != instance.num_operations(j):
            errors.append(f"Job {j}: wrong number of operations")
            continue

        for o in range(instance.num_operations(j)):
            mach = solution.machine_assignments[j][o]
            alts = instance.operations[j][o]
            valid_machines = [m for m, _ in alts]
            if mach not in valid_machines:
                errors.append(f"Job {j} op {o}: machine {mach} not eligible")
                continue

            # Get duration for assigned machine
            dur = next(d for m, d in alts if m == mach)
            st = solution.start_times[j][o]

            # Precedence within job
            if o > 0:
                prev_mach = solution.machine_assignments[j][o - 1]
                prev_dur = next(d for m, d in instance.operations[j][o - 1]
                                if m == prev_mach)
                if st < solution.start_times[j][o - 1] + prev_dur - 1e-6:
                    errors.append(f"Job {j} op {o}: precedence violated")

    # Check machine conflicts (no overlap on same machine)
    machine_ops = {}
    for j in range(instance.n):
        for o in range(instance.num_operations(j)):
            mach = solution.machine_assignments[j][o]
            dur = next(d for m, d in instance.operations[j][o] if m == mach)
            st = solution.start_times[j][o]
            machine_ops.setdefault(mach, []).append((st, st + dur, j, o))

    for mach, ops in machine_ops.items():
        ops.sort()
        for i in range(len(ops) - 1):
            if ops[i][1] > ops[i + 1][0] + 1e-6:
                errors.append(
                    f"Machine {mach}: overlap between "
                    f"job {ops[i][2]} op {ops[i][3]} and "
                    f"job {ops[i+1][2]} op {ops[i+1][3]}")

    # Check weighted tardiness
    actual_wt = 0.0
    for j in range(instance.n):
        last_o = instance.num_operations(j) - 1
        mach = solution.machine_assignments[j][last_o]
        dur = next(d for m, d in instance.operations[j][last_o] if m == mach)
        cj = solution.start_times[j][last_o] + dur
        tj = max(0.0, cj - instance.due_dates[j])
        actual_wt += instance.weights[j] * tj

    if abs(actual_wt - solution.total_weighted_tardiness) > 1e-2:
        errors.append(
            f"ΣwjTj: {solution.total_weighted_tardiness:.2f} != {actual_wt:.2f}")

    return len(errors) == 0, errors


def small_ftjsp_3x3() -> FlexTardJSPInstance:
    """Small flexible tardiness JSP: 3 jobs, 3 machines, 2 ops each."""
    operations = [
        # Job 0: 2 operations
        [[(0, 10), (1, 8)], [(1, 6), (2, 9)]],
        # Job 1: 2 operations
        [[(0, 7), (2, 5)], [(1, 10), (2, 8)]],
        # Job 2: 2 operations
        [[(1, 12), (2, 6)], [(0, 8), (1, 11)]],
    ]
    return FlexTardJSPInstance(
        n=3, m=3, operations=operations,
        due_dates=np.array([20, 18, 22], dtype=float),
        weights=np.array([2.0, 3.0, 1.0]),
        name="small_ftjsp_3x3",
    )


if __name__ == "__main__":
    inst = small_ftjsp_3x3()
    print(f"{inst.name}: n={inst.n}, m={inst.m}, ops={inst.total_operations()}")
