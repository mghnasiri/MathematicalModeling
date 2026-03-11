"""
Preemptive Single Machine Scheduling — Instance and Solution.

Problem notation: 1 | pmtn, rj | ΣCj

Jobs have release dates rj and processing times pj. Preemption is allowed:
a job can be interrupted and resumed later. Objective: minimize total
completion time ΣCj (or ΣwjCj with weights).

Applications: CPU scheduling, manufacturing with priority interrupts,
hospital patient scheduling.

Complexity: Polynomial for 1|pmtn|ΣCj via SRPT; NP-hard for 1|pmtn|ΣwjCj.

References:
    Schrage, L. (1968). A proof of the optimality of the shortest remaining
    processing time discipline. Operations Research, 16(3), 687-690.
    https://doi.org/10.1287/opre.16.3.687

    Baker, K.R. & Trietsch, D. (2009). Principles of Sequencing and
    Scheduling. Wiley. https://doi.org/10.1002/9780470451793
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class PreemptiveSMInstance:
    """Preemptive Single Machine instance.

    Attributes:
        n: Number of jobs.
        processing_times: Processing time for each job, shape (n,).
        release_dates: Release date for each job, shape (n,).
        weights: Job weights, shape (n,).
        due_dates: Optional due dates, shape (n,).
        name: Optional instance name.
    """

    n: int
    processing_times: np.ndarray
    release_dates: np.ndarray
    weights: np.ndarray
    due_dates: np.ndarray | None = None
    name: str = ""

    def __post_init__(self):
        self.processing_times = np.asarray(self.processing_times, dtype=float)
        self.release_dates = np.asarray(self.release_dates, dtype=float)
        self.weights = np.asarray(self.weights, dtype=float)
        if self.due_dates is not None:
            self.due_dates = np.asarray(self.due_dates, dtype=float)

    @classmethod
    def random(
        cls, n: int = 6,
        proc_range: tuple[int, int] = (3, 15),
        release_spread: float = 0.5,
        seed: int | None = None,
    ) -> PreemptiveSMInstance:
        rng = np.random.default_rng(seed)
        processing_times = rng.integers(proc_range[0], proc_range[1] + 1, size=n).astype(float)
        max_release = processing_times.sum() * release_spread
        release_dates = np.sort(np.round(rng.uniform(0, max_release, size=n)))
        weights = np.round(rng.uniform(1, 10, size=n))
        return cls(n=n, processing_times=processing_times,
                   release_dates=release_dates, weights=weights,
                   name=f"random_{n}")

    def total_weighted_completion(self, completion_times: list[float]) -> float:
        return sum(self.weights[j] * completion_times[j] for j in range(self.n))

    def total_completion(self, completion_times: list[float]) -> float:
        return sum(completion_times)


@dataclass
class PreemptiveSMSolution:
    """Solution: list of (job, start, end) segments and completion times."""

    segments: list[tuple[int, float, float]]
    completion_times: list[float]
    objective: float

    def __repr__(self) -> str:
        return f"PreemptiveSMSolution(obj={self.objective:.1f})"


def validate_solution(
    instance: PreemptiveSMInstance, solution: PreemptiveSMSolution
) -> tuple[bool, list[str]]:
    errors = []
    # Check total processing for each job
    job_processed = [0.0] * instance.n
    for job, start, end in solution.segments:
        if job < 0 or job >= instance.n:
            errors.append(f"Invalid job {job}")
            continue
        if start < instance.release_dates[job] - 1e-10:
            errors.append(f"Job {job}: starts at {start:.1f} before release {instance.release_dates[job]:.1f}")
        job_processed[job] += end - start

    for j in range(instance.n):
        if abs(job_processed[j] - instance.processing_times[j]) > 1e-4:
            errors.append(f"Job {j}: processed {job_processed[j]:.2f} != required {instance.processing_times[j]:.2f}")

    # Check no overlap
    sorted_segs = sorted(solution.segments, key=lambda s: s[1])
    for i in range(len(sorted_segs) - 1):
        if sorted_segs[i][2] > sorted_segs[i + 1][1] + 1e-10:
            errors.append(f"Overlap: segment ending {sorted_segs[i][2]:.1f} > next starting {sorted_segs[i+1][1]:.1f}")

    return len(errors) == 0, errors


def small_preemptive_4() -> PreemptiveSMInstance:
    return PreemptiveSMInstance(
        n=4,
        processing_times=np.array([6, 3, 8, 4], dtype=float),
        release_dates=np.array([0, 2, 4, 6], dtype=float),
        weights=np.array([1, 1, 1, 1], dtype=float),
        name="small_4",
    )


if __name__ == "__main__":
    inst = small_preemptive_4()
    print(f"{inst.name}: n={inst.n}")
