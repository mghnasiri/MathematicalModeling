"""
Job Shop with Weighted Tardiness (Jm || ΣwjTj) — Instance and Solution.

Problem notation: Jm || ΣwjTj

Extends JSP with due dates and weights. Objective: minimize weighted
tardiness ΣwjTj where Tj = max(0, Cj - dj).

Applications: make-to-order manufacturing, semiconductor fab scheduling,
batch chemical processing with customer deadlines.

Complexity: NP-hard (strongly; even 1||ΣwjTj is strongly NP-hard).

References:
    Pinedo, M. (2016). Scheduling: Theory, Algorithms, and Systems.
    5th ed. Springer. https://doi.org/10.1007/978-3-319-26580-3

    Singer, M. & Pinedo, M. (1998). A computational study of branch
    and bound techniques for minimizing the total weighted tardiness
    in job shops. IIE Transactions, 30(2), 109-118.
    https://doi.org/10.1080/07408179808966441
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class WTJSPInstance:
    """Job Shop with Weighted Tardiness instance.

    Attributes:
        n: Number of jobs.
        m: Number of machines.
        operations: operations[j] = [(machine, duration), ...] for job j.
        due_dates: Due date for each job, shape (n,).
        weights: Weight for each job, shape (n,).
        name: Optional instance name.
    """

    n: int
    m: int
    operations: list[list[tuple[int, int]]]
    due_dates: np.ndarray
    weights: np.ndarray
    name: str = ""

    def __post_init__(self):
        self.due_dates = np.asarray(self.due_dates, dtype=float)
        self.weights = np.asarray(self.weights, dtype=float)

    @classmethod
    def random(
        cls,
        n: int = 6,
        m: int = 3,
        duration_range: tuple[int, int] = (1, 10),
        due_date_factor: float = 1.5,
        seed: int | None = None,
    ) -> WTJSPInstance:
        """Generate a random instance."""
        rng = np.random.default_rng(seed)
        operations = []
        total_durations = []
        for j in range(n):
            machines = list(rng.permutation(m))
            ops = [
                (int(machines[k]),
                 int(rng.integers(duration_range[0], duration_range[1] + 1)))
                for k in range(m)
            ]
            operations.append(ops)
            total_durations.append(sum(d for _, d in ops))

        avg_dur = np.mean(total_durations)
        due_dates = np.round(
            rng.uniform(avg_dur * 0.8, avg_dur * due_date_factor * 1.2, size=n)
        )
        weights = np.round(rng.uniform(1, 10, size=n))

        return cls(n=n, m=m, operations=operations, due_dates=due_dates,
                   weights=weights, name=f"random_{n}_{m}")

    def weighted_tardiness(self, completion_times: list[float]) -> float:
        """Compute ΣwjTj."""
        wt = 0.0
        for j in range(self.n):
            tardiness = max(0.0, completion_times[j] - self.due_dates[j])
            wt += self.weights[j] * tardiness
        return wt


@dataclass
class WTJSPSolution:
    """Solution to a WTJSP instance.

    Attributes:
        machine_sequences: machine_sequences[k] = ordered list of (job, op_idx).
        completion_times: Completion time for each job.
        weighted_tardiness: ΣwjTj value.
    """

    machine_sequences: list[list[tuple[int, int]]]
    completion_times: list[float]
    weighted_tardiness: float

    def __repr__(self) -> str:
        return f"WTJSPSolution(wt={self.weighted_tardiness:.1f})"


def schedule_from_sequences(
    instance: WTJSPInstance,
    machine_sequences: list[list[tuple[int, int]]],
) -> tuple[list[float], float]:
    """Compute completion times from machine sequences.

    Args:
        instance: A WTJSPInstance.
        machine_sequences: For each machine, ordered list of (job, op_idx).

    Returns:
        (completion_times, makespan).
    """
    n, m = instance.n, instance.m
    op_start = [[0.0] * len(instance.operations[j]) for j in range(n)]
    op_end = [[0.0] * len(instance.operations[j]) for j in range(n)]

    machine_time = [0.0] * m
    job_time = [0.0] * n

    # Process in topological order respecting both job and machine precedence
    # Simple approach: iterate machine sequences round-robin
    machine_pos = [0] * m
    scheduled = [[False] * len(instance.operations[j]) for j in range(n)]
    total_ops = sum(len(instance.operations[j]) for j in range(n))
    count = 0

    while count < total_ops:
        progress = False
        for k in range(m):
            if machine_pos[k] >= len(machine_sequences[k]):
                continue
            job, op_idx = machine_sequences[k][machine_pos[k]]
            # Check if previous operation of this job is done
            if op_idx > 0 and not scheduled[job][op_idx - 1]:
                continue

            _, dur = instance.operations[job][op_idx]
            start = max(machine_time[k], job_time[job])
            op_start[job][op_idx] = start
            op_end[job][op_idx] = start + dur
            machine_time[k] = start + dur
            job_time[job] = start + dur
            scheduled[job][op_idx] = True
            machine_pos[k] += 1
            count += 1
            progress = True

        if not progress:
            # Deadlock — force schedule remaining
            for k in range(m):
                if machine_pos[k] >= len(machine_sequences[k]):
                    continue
                job, op_idx = machine_sequences[k][machine_pos[k]]
                _, dur = instance.operations[job][op_idx]
                start = max(machine_time[k], job_time[job])
                op_start[job][op_idx] = start
                op_end[job][op_idx] = start + dur
                machine_time[k] = start + dur
                job_time[job] = start + dur
                scheduled[job][op_idx] = True
                machine_pos[k] += 1
                count += 1
                break

    completion_times = [job_time[j] for j in range(n)]
    makespan = max(completion_times)
    return completion_times, makespan


def validate_solution(
    instance: WTJSPInstance, solution: WTJSPSolution
) -> tuple[bool, list[str]]:
    """Validate a WTJSP solution."""
    errors = []

    if len(solution.machine_sequences) != instance.m:
        errors.append(f"Expected {instance.m} machine sequences")
        return False, errors

    # Check all operations present
    all_ops = []
    for k, seq in enumerate(solution.machine_sequences):
        for job, op_idx in seq:
            mach, _ = instance.operations[job][op_idx]
            if mach != k:
                errors.append(f"Op ({job},{op_idx}) on machine {k} but belongs to {mach}")
            all_ops.append((job, op_idx))

    expected = []
    for j in range(instance.n):
        for k in range(len(instance.operations[j])):
            expected.append((j, k))

    if sorted(all_ops) != sorted(expected):
        errors.append("Not all operations scheduled exactly once")

    if errors:
        return False, errors

    ct, _ = schedule_from_sequences(instance, solution.machine_sequences)
    actual_wt = instance.weighted_tardiness(ct)
    if abs(actual_wt - solution.weighted_tardiness) > 1e-4:
        errors.append(f"Reported WT {solution.weighted_tardiness:.2f} != actual {actual_wt:.2f}")

    return len(errors) == 0, errors


def small_wtjsp_3_3() -> WTJSPInstance:
    """3 jobs, 3 machines with due dates and weights."""
    return WTJSPInstance(
        n=3, m=3,
        operations=[
            [(0, 3), (1, 2), (2, 4)],
            [(1, 4), (2, 3), (0, 2)],
            [(2, 2), (0, 3), (1, 5)],
        ],
        due_dates=np.array([12, 15, 18], dtype=float),
        weights=np.array([3, 1, 2], dtype=float),
        name="small_3_3",
    )


if __name__ == "__main__":
    inst = small_wtjsp_3_3()
    print(f"{inst.name}: {inst.n} jobs, {inst.m} machines")
