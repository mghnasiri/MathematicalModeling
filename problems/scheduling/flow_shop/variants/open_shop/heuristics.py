"""
Open Shop Scheduling — Heuristics.

Algorithms:
    - LPT dispatching (Longest Processing Time first).
    - Greedy list scheduling with machine-earliest availability.

References:
    Gonzalez, T. & Sahni, S. (1976). Open shop scheduling to minimize
    finish time. Journal of the ACM, 23(4), 665-679.
    https://doi.org/10.1145/321978.321985
"""

from __future__ import annotations

import sys
import os
import importlib.util

import numpy as np

_this_dir = os.path.dirname(os.path.abspath(__file__))


def _load_mod(name, filepath):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_inst = _load_mod("os_instance_h", os.path.join(_this_dir, "instance.py"))
OpenShopInstance = _inst.OpenShopInstance
OpenShopSolution = _inst.OpenShopSolution


def _build_schedule(instance: OpenShopInstance,
                    job_machine_order: list[list[int]]) -> OpenShopSolution:
    """Build a schedule from a job-machine ordering.

    Args:
        instance: Open shop instance.
        job_machine_order: job_machine_order[j] = machine processing order for job j.

    Returns:
        OpenShopSolution.
    """
    n, m = instance.n, instance.m
    machine_avail = [0.0] * m
    job_avail = [0.0] * n
    schedule = [[] for _ in range(n)]

    # Schedule operations in a greedy manner
    # Priority: process operations by earliest possible start
    ops = []
    next_op = [0] * n  # next machine index in job_machine_order[j]

    # Initialize with first operation of each job
    for j in range(n):
        mach = job_machine_order[j][0]
        ops.append((0.0, j, mach))

    import heapq
    heapq.heapify(ops)

    scheduled_count = 0
    while ops and scheduled_count < n * m:
        _, j, mach = heapq.heappop(ops)
        dur = instance.processing_times[j][mach]
        st = max(job_avail[j], machine_avail[mach])
        schedule[j].append((mach, st))
        machine_avail[mach] = st + dur
        job_avail[j] = st + dur
        next_op[j] += 1
        scheduled_count += 1

        if next_op[j] < m:
            next_mach = job_machine_order[j][next_op[j]]
            est = max(job_avail[j], machine_avail[next_mach])
            heapq.heappush(ops, (est, j, next_mach))

    makespan = instance.makespan(schedule)
    return OpenShopSolution(schedule=schedule, makespan=makespan)


def lpt_open_shop(instance: OpenShopInstance) -> OpenShopSolution:
    """LPT dispatching: process longest operations first per job.

    Each job's machine order is sorted by decreasing processing time.

    Args:
        instance: Open shop instance.

    Returns:
        OpenShopSolution.
    """
    job_machine_order = []
    for j in range(instance.n):
        order = sorted(range(instance.m),
                       key=lambda k: instance.processing_times[j][k],
                       reverse=True)
        job_machine_order.append(order)

    return _build_schedule(instance, job_machine_order)


def greedy_open_shop(instance: OpenShopInstance) -> OpenShopSolution:
    """Greedy scheduling: assign operations by earliest availability.

    Iteratively pick the unscheduled operation with the earliest
    possible start time.

    Args:
        instance: Open shop instance.

    Returns:
        OpenShopSolution.
    """
    n, m = instance.n, instance.m
    machine_avail = [0.0] * m
    job_avail = [0.0] * n
    schedule = [[] for _ in range(n)]
    done = [[False] * m for _ in range(n)]

    for _ in range(n * m):
        best_st = float("inf")
        best_j = -1
        best_m = -1

        for j in range(n):
            for k in range(m):
                if done[j][k]:
                    continue
                st = max(job_avail[j], machine_avail[k])
                if st < best_st:
                    best_st = st
                    best_j = j
                    best_m = k

        if best_j == -1:
            break

        dur = instance.processing_times[best_j][best_m]
        schedule[best_j].append((best_m, best_st))
        machine_avail[best_m] = best_st + dur
        job_avail[best_j] = best_st + dur
        done[best_j][best_m] = True

    makespan = instance.makespan(schedule)
    return OpenShopSolution(schedule=schedule, makespan=makespan)


if __name__ == "__main__":
    from instance import small_os_3x3

    inst = small_os_3x3()
    sol1 = lpt_open_shop(inst)
    print(f"LPT: {sol1}")
    sol2 = greedy_open_shop(inst)
    print(f"Greedy: {sol2}")
