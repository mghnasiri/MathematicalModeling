"""
Moore's Algorithm for Single Machine Scheduling — 1 || ΣUj

Minimizes the number of tardy jobs on a single machine. The algorithm
processes jobs in EDD order and removes the longest job whenever a
tardy job is encountered, placing removed jobs at the end.

Optimality: This greedy algorithm is provably optimal for 1 || ΣUj.

Complexity: O(n log n) using a max-heap for efficient longest-job removal.

Reference: Moore, J.M. (1968). "An n Job, One Machine Sequencing Algorithm
           for Minimizing the Number of Late Jobs"
           Management Science, 15(1):102-109.
           DOI: 10.1287/mnsc.15.1.102
"""

from __future__ import annotations
import sys
import os
import heapq
import importlib.util

_this_dir = os.path.dirname(os.path.abspath(__file__))
_instance_path = os.path.join(os.path.dirname(_this_dir), "instance.py")
_spec = importlib.util.spec_from_file_location("sm_instance", _instance_path)
_sm_instance = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("sm_instance", _sm_instance)
_spec.loader.exec_module(_sm_instance)

SingleMachineInstance = _sm_instance.SingleMachineInstance
SingleMachineSolution = _sm_instance.SingleMachineSolution
compute_number_tardy = _sm_instance.compute_number_tardy


def moores_algorithm(instance: SingleMachineInstance) -> SingleMachineSolution:
    """
    Moore's Algorithm — optimal for 1 || ΣUj.

    Processes jobs in EDD order. Maintains a set of on-time jobs. When
    adding a job causes its completion to exceed its due date, the job
    with the longest processing time in the current set is removed and
    placed in the late set. Late jobs are appended at the end in
    arbitrary order.

    Args:
        instance: A SingleMachineInstance (must have due dates).

    Returns:
        SingleMachineSolution with optimal ΣUj.

    Complexity: O(n log n)
    """
    assert instance.due_dates is not None, "Moore's algorithm requires due dates"

    n = instance.n
    p = instance.processing_times
    d = instance.due_dates

    # Sort by EDD
    edd_order = sorted(range(n), key=lambda j: (d[j], j))

    # Max-heap (negate for Python's min-heap)
    on_time = []  # heap of (-processing_time, job_index)
    on_time_set = set()
    current_time = 0

    for job in edd_order:
        current_time += p[job]
        heapq.heappush(on_time, (-int(p[job]), job))
        on_time_set.add(job)

        # If current job is tardy, remove the longest job
        if current_time > d[job]:
            neg_p, longest_job = heapq.heappop(on_time)
            on_time_set.remove(longest_job)
            current_time += neg_p  # subtract processing time (neg_p is negative)

    # Build sequence: on-time jobs in EDD order, then late jobs
    on_time_sequence = [j for j in edd_order if j in on_time_set]
    late_sequence = [j for j in edd_order if j not in on_time_set]
    sequence = on_time_sequence + late_sequence

    obj = compute_number_tardy(instance, sequence)
    return SingleMachineSolution(
        sequence=sequence, objective_value=obj, objective_name="ΣUj"
    )


if __name__ == "__main__":
    import numpy as np

    print("=== Moore's Algorithm Demo ===\n")

    # Classic example: 5 jobs
    inst = SingleMachineInstance.from_arrays(
        processing_times=[3, 5, 2, 7, 4],
        due_dates=[6, 8, 10, 15, 11],
    )
    print(f"p = {inst.processing_times}")
    print(f"d = {inst.due_dates}")

    sol = moores_algorithm(inst)
    print(f"\nMoore's: {sol}")
    print(f"Tardy jobs: {sol.objective_value}")
