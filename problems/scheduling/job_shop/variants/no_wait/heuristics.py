"""
Constructive Heuristics for No-Wait Job Shop.

Problem: Jm | no-wait | Cmax
Complexity: O(n^2 * m)

1. Greedy Insertion: insert jobs one at a time at the position
   (start time) that minimizes makespan, checking feasibility.
2. SPT order: schedule jobs in shortest processing time order
   with greedy start time assignment.

References:
    Mascis, A. & Pacciarelli, D. (2002). Job-shop scheduling with
    blocking and no-wait constraints. European Journal of Operational
    Research, 143(3), 498-517.
    https://doi.org/10.1016/S0377-2217(01)00338-1
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


_inst = _load_mod("nwjsp_instance_h", os.path.join(_this_dir, "instance.py"))
NWJSPInstance = _inst.NWJSPInstance
NWJSPSolution = _inst.NWJSPSolution


def _find_earliest_feasible_start(
    instance: NWJSPInstance,
    scheduled_starts: list[float],
    scheduled_jobs: list[int],
    new_job: int,
) -> float:
    """Find earliest feasible start time for new_job given already scheduled jobs."""
    if not scheduled_jobs:
        return 0.0

    # Build machine intervals from already scheduled jobs
    machine_intervals: dict[int, list[tuple[float, float]]] = {m: [] for m in range(instance.m)}
    for j in scheduled_jobs:
        t = scheduled_starts[j]
        for mach, dur in instance.operations[j]:
            machine_intervals[mach].append((t, t + dur))
            t += dur

    # Try start times: 0, and right after each existing interval on relevant machines
    candidates = [0.0]
    for k, (mach, dur) in enumerate(instance.operations[new_job]):
        for start, end in machine_intervals[mach]:
            # new_job operation k starts at offset from job start
            offset = sum(d for _, d in instance.operations[new_job][:k])
            candidates.append(end - offset)

    candidates.sort()

    for start_time in candidates:
        if start_time < -1e-10:
            continue
        start_time = max(0.0, start_time)

        feasible = True
        t = start_time
        for mach, dur in instance.operations[new_job]:
            for s, e in machine_intervals[mach]:
                if t < e - 1e-10 and s < t + dur - 1e-10:
                    feasible = False
                    break
            if not feasible:
                break
            t += dur

        if feasible:
            return start_time

    # Fallback: schedule after all existing operations
    max_end = max(
        (scheduled_starts[j] + instance.job_duration(j) for j in scheduled_jobs),
        default=0.0,
    )
    return max_end


def greedy_insertion(instance: NWJSPInstance) -> NWJSPSolution:
    """Schedule jobs one at a time using greedy insertion.

    Sort by total processing time (longest first), then insert each
    at the earliest feasible start time.

    Args:
        instance: A NWJSPInstance.

    Returns:
        NWJSPSolution.
    """
    n = instance.n
    order = sorted(range(n), key=lambda j: instance.job_duration(j), reverse=True)

    starts = [0.0] * n
    scheduled: list[int] = []

    for j in order:
        starts[j] = _find_earliest_feasible_start(instance, starts, scheduled, j)
        scheduled.append(j)

    ms = instance.makespan(starts)
    return NWJSPSolution(job_start_times=starts, makespan=ms)


def spt_schedule(instance: NWJSPInstance) -> NWJSPSolution:
    """Schedule jobs in SPT order with greedy start times.

    Args:
        instance: A NWJSPInstance.

    Returns:
        NWJSPSolution.
    """
    n = instance.n
    order = sorted(range(n), key=lambda j: instance.job_duration(j))

    starts = [0.0] * n
    scheduled: list[int] = []

    for j in order:
        starts[j] = _find_earliest_feasible_start(instance, starts, scheduled, j)
        scheduled.append(j)

    ms = instance.makespan(starts)
    return NWJSPSolution(job_start_times=starts, makespan=ms)


if __name__ == "__main__":
    inst = _inst.small_nwjsp_3_3()
    print(f"NW-JSP: {inst.n} jobs, {inst.m} machines")

    sol1 = greedy_insertion(inst)
    print(f"Greedy: makespan={sol1.makespan:.1f}, feasible={inst.is_feasible(sol1.job_start_times)}")

    sol2 = spt_schedule(inst)
    print(f"SPT: makespan={sol2.makespan:.1f}, feasible={inst.is_feasible(sol2.job_start_times)}")
