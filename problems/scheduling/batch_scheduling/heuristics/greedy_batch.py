"""Greedy batch scheduling heuristic.

Algorithm: Group jobs by family. Within each family, sort by WSPT
(weighted shortest processing time: ascending p_j/w_j). Schedule
families in order of minimum WSPT ratio (most urgent family first).
A setup time is incurred at the start of each family batch.

Complexity: O(n log n) for sorting.

References:
    Potts, C. N., & Kovalyov, M. Y. (2000). Scheduling with batching:
    A review. European Journal of Operational Research, 120(2), 228-249.
"""
import sys
import os
import importlib.util

import numpy as np

def _load_parent(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_inst = _load_parent(
    "batch_instance",
    os.path.join(os.path.dirname(__file__), "..", "instance.py")
)

BatchSchedulingInstance = _inst.BatchSchedulingInstance
BatchSchedulingSolution = _inst.BatchSchedulingSolution


def greedy_batch(instance: BatchSchedulingInstance) -> BatchSchedulingSolution:
    """Greedy batch scheduling with WSPT within families.

    Groups jobs by family, sorts each group by WSPT ratio (p_j/w_j ascending),
    then schedules families in order of their average WSPT ratio.

    Args:
        instance: A BatchSchedulingInstance.

    Returns:
        BatchSchedulingSolution with schedule and completion times.
    """
    n = instance.n_jobs

    # Group jobs by family
    family_jobs: dict[int, list[int]] = {}
    for j in range(n):
        fam = int(instance.families[j])
        if fam not in family_jobs:
            family_jobs[fam] = []
        family_jobs[fam].append(j)

    # Sort jobs within each family by WSPT ratio (p/w ascending = high priority)
    for fam in family_jobs:
        family_jobs[fam].sort(
            key=lambda j: instance.processing_times[j] / max(instance.weights[j], 1e-12)
        )

    # Order families by minimum WSPT ratio in the family (most urgent first)
    def family_priority(fam: int) -> float:
        jobs = family_jobs[fam]
        ratios = [instance.processing_times[j] / max(instance.weights[j], 1e-12)
                  for j in jobs]
        return min(ratios)

    family_order = sorted(family_jobs.keys(), key=family_priority)

    # Build schedule
    schedule: list[int] = []
    batches: list[list[int]] = []
    completion_times = np.zeros(n)
    current_time = 0.0

    for i, fam in enumerate(family_order):
        # Setup time before each family (including the first)
        current_time += instance.setup_time

        batch = family_jobs[fam]
        batches.append(list(batch))

        for job in batch:
            current_time += instance.processing_times[job]
            completion_times[job] = current_time
            schedule.append(job)

    twc = float(np.sum(instance.weights * completion_times))

    return BatchSchedulingSolution(
        schedule=schedule, batches=batches,
        completion_times=completion_times,
        total_weighted_completion=twc
    )


if __name__ == "__main__":
    inst = BatchSchedulingInstance.random()
    sol = greedy_batch(inst)
    print(f"Instance: {inst.n_jobs} jobs, {inst.n_families} families")
    print(sol)
    print(f"Schedule: {sol.schedule}")
    print(f"Batches: {sol.batches}")
