"""
List Scheduling — Simple Greedy Heuristic for Pm || Cmax

The simplest scheduling heuristic: process jobs in a given order (or
arbitrary order) and assign each to the least loaded machine.

Approximation guarantee: List scheduling achieves a ratio of at most
2 - 1/m for identical parallel machines, regardless of job ordering.

When combined with LPT ordering, this becomes the LPT algorithm with a
tighter bound of 4/3 - 1/(3m).

Notation: Pm || Cmax
Complexity: O(n log m) with a heap for machine selection
Reference: Graham, R.L. (1966). "Bounds for Certain Multiprocessing Anomalies"
           Bell System Technical Journal, 45(9):1563-1581.
           DOI: 10.1002/j.1538-7305.1966.tb01709.x
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


def list_scheduling(
    instance: ParallelMachineInstance,
    job_order: list[int] | None = None,
) -> ParallelMachineSolution:
    """
    List scheduling heuristic for makespan minimization.

    Assigns each job (in the given order) to the least loaded machine.

    Args:
        instance: A ParallelMachineInstance.
        job_order: Order in which to process jobs. If None, uses natural
                  order [0, 1, ..., n-1].

    Returns:
        ParallelMachineSolution with the assignment and makespan.
    """
    n = instance.n
    m = instance.m

    if job_order is None:
        job_order = list(range(n))

    assignment: list[list[int]] = [[] for _ in range(m)]
    # Min-heap: (current_load, machine_index)
    heap = [(0.0, i) for i in range(m)]
    heapq.heapify(heap)

    for job in job_order:
        load, machine = heapq.heappop(heap)
        assignment[machine].append(job)
        new_load = load + instance.get_processing_time(job, machine)
        heapq.heappush(heap, (new_load, machine))

    ms = compute_makespan(instance, assignment)
    loads = compute_machine_loads(instance, assignment)
    return ParallelMachineSolution(
        assignment=assignment, makespan=ms, machine_loads=loads
    )


if __name__ == "__main__":
    print("=== List Scheduling for Parallel Machines ===\n")

    instance = ParallelMachineInstance.random_identical(n=12, m=3, seed=42)
    print(f"Instance: {instance.n} jobs, {instance.m} machines")
    print(f"Processing times: {instance.processing_times}")

    sol = list_scheduling(instance)
    print(f"\nList scheduling Makespan: {sol.makespan:.0f}")
    for i, jobs in enumerate(sol.assignment):
        print(f"  Machine {i}: {jobs} (load={sol.machine_loads[i]:.0f})")

    # With LPT ordering
    sorted_jobs = sorted(range(instance.n),
                         key=lambda j: instance.processing_times[j],
                         reverse=True)
    sol_lpt = list_scheduling(instance, job_order=sorted_jobs)
    print(f"\nLPT-ordered Makespan: {sol_lpt.makespan:.0f}")
