"""
Longest Processing Time (LPT) — Greedy Heuristic for Pm || Cmax

The LPT rule sorts jobs in decreasing order of processing time and assigns
each job to the machine with the currently smallest load. This greedy
approach produces high-quality solutions for makespan minimization.

Approximation guarantee: LPT achieves a ratio of at most 4/3 - 1/(3m)
for identical parallel machines.

For the total completion time objective (Pm || sum Cj), the SPT rule is
used instead — it is optimal when combined with round-robin assignment.

Algorithm (LPT for Cmax):
    1. Sort jobs by decreasing processing time.
    2. For each job (in sorted order):
       - Assign to the machine with the current minimum load.
    3. Return the assignment and makespan.

Notation: Pm || Cmax
Complexity: O(n log n + n log m) — sorting + heap-based machine selection
Reference: Graham, R.L. (1969). "Bounds on Multiprocessing Timing Anomalies"
           SIAM Journal on Applied Mathematics, 17(2):416-429.
           DOI: 10.1137/0117039
"""

from __future__ import annotations
import sys
import os
import heapq
import importlib.util

_this_dir = os.path.dirname(os.path.abspath(__file__))
_instance_path = os.path.join(_this_dir, "..", "instance.py")
_spec = importlib.util.spec_from_file_location("pm_instance", _instance_path)
_pm_instance = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("pm_instance", _pm_instance)
_spec.loader.exec_module(_pm_instance)

ParallelMachineInstance = _pm_instance.ParallelMachineInstance
ParallelMachineSolution = _pm_instance.ParallelMachineSolution
compute_makespan = _pm_instance.compute_makespan
compute_machine_loads = _pm_instance.compute_machine_loads


def lpt(instance: ParallelMachineInstance) -> ParallelMachineSolution:
    """
    Longest Processing Time first heuristic for makespan minimization.

    Args:
        instance: A ParallelMachineInstance (identical, uniform, or unrelated).

    Returns:
        ParallelMachineSolution with the assignment and makespan.
    """
    n = instance.n
    m = instance.m

    if instance.machine_type == "unrelated":
        return _lpt_unrelated(instance)

    # Sort jobs by decreasing processing time
    sorted_jobs = sorted(range(n), key=lambda j: instance.processing_times[j],
                         reverse=True)

    # Min-heap: (current_load, machine_index)
    assignment: list[list[int]] = [[] for _ in range(m)]
    heap = [(0.0, i) for i in range(m)]
    heapq.heapify(heap)

    for job in sorted_jobs:
        load, machine = heapq.heappop(heap)
        assignment[machine].append(job)
        new_load = load + instance.get_processing_time(job, machine)
        heapq.heappush(heap, (new_load, machine))

    ms = compute_makespan(instance, assignment)
    loads = compute_machine_loads(instance, assignment)
    return ParallelMachineSolution(
        assignment=assignment, makespan=ms, machine_loads=loads
    )


def _lpt_unrelated(instance: ParallelMachineInstance) -> ParallelMachineSolution:
    """
    LPT adapted for unrelated machines.

    For each job (sorted by max processing time descending), assign it to
    the machine that results in the smallest increase in makespan.
    """
    n = instance.n
    m = instance.m

    # Sort by maximum processing time across machines (descending)
    max_times = [max(instance.processing_times[i, j] for i in range(m))
                 for j in range(n)]
    sorted_jobs = sorted(range(n), key=lambda j: max_times[j], reverse=True)

    assignment: list[list[int]] = [[] for _ in range(m)]
    loads = [0.0] * m

    for job in sorted_jobs:
        # Assign to machine minimizing the resulting load
        best_machine = 0
        best_load = float('inf')
        for i in range(m):
            new_load = loads[i] + instance.get_processing_time(job, i)
            if new_load < best_load:
                best_load = new_load
                best_machine = i
        assignment[best_machine].append(job)
        loads[best_machine] += instance.get_processing_time(job, best_machine)

    ms = max(loads)
    return ParallelMachineSolution(
        assignment=assignment, makespan=ms, machine_loads=loads
    )


def spt_parallel(instance: ParallelMachineInstance) -> ParallelMachineSolution:
    """
    Shortest Processing Time heuristic for total completion time.

    For Pm || sum Cj, SPT with round-robin assignment is optimal.
    Jobs are sorted by increasing processing time and assigned to
    machines in round-robin order.

    Args:
        instance: A ParallelMachineInstance (identical machines).

    Returns:
        ParallelMachineSolution optimized for total completion time.
    """
    n = instance.n
    m = instance.m

    sorted_jobs = sorted(range(n), key=lambda j: instance.processing_times[j])

    assignment: list[list[int]] = [[] for _ in range(m)]
    for idx, job in enumerate(sorted_jobs):
        assignment[idx % m].append(job)

    ms = compute_makespan(instance, assignment)
    loads = compute_machine_loads(instance, assignment)
    return ParallelMachineSolution(
        assignment=assignment, makespan=ms, machine_loads=loads
    )


if __name__ == "__main__":
    print("=== LPT and SPT for Parallel Machines ===\n")

    instance = ParallelMachineInstance.random_identical(n=12, m=3, seed=42)
    print(f"Instance: {instance.n} jobs, {instance.m} machines")
    print(f"Processing times: {instance.processing_times}")

    sol_lpt = lpt(instance)
    print(f"\nLPT  Makespan: {sol_lpt.makespan:.0f}")
    for i, jobs in enumerate(sol_lpt.assignment):
        print(f"  Machine {i}: {jobs} (load={sol_lpt.machine_loads[i]:.0f})")

    sol_spt = spt_parallel(instance)
    print(f"\nSPT  Makespan: {sol_spt.makespan:.0f}")
    for i, jobs in enumerate(sol_spt.assignment):
        print(f"  Machine {i}: {jobs}")
