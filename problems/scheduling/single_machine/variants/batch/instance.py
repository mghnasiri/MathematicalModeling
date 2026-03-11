"""
Single Machine Batch Scheduling (1 | batch, sj | ΣwjCj).

Jobs are grouped into batches. All jobs in a batch start together;
the batch processing time equals the maximum processing time of its jobs.
A setup time s is incurred between consecutive batches.

Objective: Minimize total weighted completion time ΣwjCj.

Complexity: NP-hard even with equal setup times.

References:
    Potts, C.N. & Kovalyov, M.Y. (2000). Scheduling with batching:
    a review. European Journal of Operational Research, 120(2), 228-249.
    https://doi.org/10.1016/S0377-2217(99)00153-8

    Webster, S. & Baker, K.R. (1995). Scheduling groups of jobs on a
    single machine. Operations Research, 43(4), 692-703.
    https://doi.org/10.1287/opre.43.4.692
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class BatchSMInstance:
    """Batch Single Machine instance.

    Attributes:
        n: Number of jobs.
        processing_times: Processing time per job, shape (n,).
        weights: Job weights, shape (n,).
        setup_time: Setup time between consecutive batches.
        name: Optional instance name.
    """

    n: int
    processing_times: np.ndarray
    weights: np.ndarray
    setup_time: float
    name: str = ""

    def __post_init__(self):
        self.processing_times = np.asarray(self.processing_times, dtype=float)
        self.weights = np.asarray(self.weights, dtype=float)

    @classmethod
    def random(
        cls,
        n: int = 8,
        proc_range: tuple[int, int] = (1, 20),
        weight_range: tuple[int, int] = (1, 10),
        setup_time: float = 5.0,
        seed: int | None = None,
    ) -> BatchSMInstance:
        rng = np.random.default_rng(seed)
        procs = rng.integers(proc_range[0], proc_range[1] + 1, size=n).astype(float)
        weights = rng.integers(weight_range[0], weight_range[1] + 1, size=n).astype(float)
        return cls(n=n, processing_times=procs, weights=weights,
                   setup_time=setup_time, name=f"random_{n}")

    def evaluate(self, batches: list[list[int]]) -> float:
        """Compute ΣwjCj for a batch schedule.

        Args:
            batches: List of batches, each batch is a list of job indices.

        Returns:
            Total weighted completion time.
        """
        time = 0.0
        total_wc = 0.0
        for b_idx, batch in enumerate(batches):
            if not batch:
                continue
            if b_idx > 0:
                time += self.setup_time
            batch_proc = max(self.processing_times[j] for j in batch)
            time += batch_proc
            for j in batch:
                total_wc += self.weights[j] * time
        return total_wc


@dataclass
class BatchSMSolution:
    """Batch Single Machine solution.

    Attributes:
        batches: List of batches (lists of job indices).
        objective: Total weighted completion time.
    """

    batches: list[list[int]]
    objective: float

    def __repr__(self) -> str:
        return f"BatchSMSolution(batches={len(self.batches)}, obj={self.objective:.1f})"


def validate_solution(
    instance: BatchSMInstance, solution: BatchSMSolution
) -> tuple[bool, list[str]]:
    errors = []
    all_jobs = []
    for batch in solution.batches:
        all_jobs.extend(batch)
    if sorted(all_jobs) != list(range(instance.n)):
        errors.append("Batches do not form a valid partition of all jobs")
    actual = instance.evaluate(solution.batches)
    if abs(actual - solution.objective) > 1e-4:
        errors.append(f"Reported obj {solution.objective:.2f} != actual {actual:.2f}")
    return len(errors) == 0, errors


def small_batch_6() -> BatchSMInstance:
    return BatchSMInstance(
        n=6,
        processing_times=np.array([5, 3, 8, 2, 6, 4], dtype=float),
        weights=np.array([2, 5, 1, 4, 3, 6], dtype=float),
        setup_time=3.0,
        name="small_6",
    )


if __name__ == "__main__":
    inst = small_batch_6()
    print(f"{inst.name}: n={inst.n}, setup={inst.setup_time}")
    # Each job in its own batch
    single = [[i] for i in range(inst.n)]
    print(f"  Single-job batches obj: {inst.evaluate(single):.1f}")
    # All in one batch
    one = [list(range(inst.n))]
    print(f"  One batch obj: {inst.evaluate(one):.1f}")
