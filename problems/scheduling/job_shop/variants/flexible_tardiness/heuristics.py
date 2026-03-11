"""
Flexible Job Shop with Tardiness — Heuristics.

Algorithms:
    - EDD dispatching with ECT machine assignment.
    - WATC (Weighted Apparent Tardiness Cost) dispatching.

References:
    Brandimarte, P. (1993). Routing and scheduling in a flexible job shop
    by tabu search. Annals of Operations Research, 41(3), 157-183.
    https://doi.org/10.1007/BF02023073

    Vepsalainen, A.P.J. & Morton, T.E. (1987). Priority rules for job
    shops with weighted tardiness costs. Management Science, 33(8),
    1035-1047. https://doi.org/10.1287/mnsc.33.8.1035
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


_inst = _load_mod("ftjsp_instance_h", os.path.join(_this_dir, "instance.py"))
FlexTardJSPInstance = _inst.FlexTardJSPInstance
FlexTardJSPSolution = _inst.FlexTardJSPSolution


def _build_solution(
    instance: FlexTardJSPInstance,
    machine_assign: list[list[int]],
    start_times: list[list[float]],
) -> FlexTardJSPSolution:
    """Build solution and compute weighted tardiness."""
    wt = 0.0
    for j in range(instance.n):
        last_o = instance.num_operations(j) - 1
        mach = machine_assign[j][last_o]
        dur = next(d for m, d in instance.operations[j][last_o] if m == mach)
        cj = start_times[j][last_o] + dur
        tj = max(0.0, cj - instance.due_dates[j])
        wt += instance.weights[j] * tj
    return FlexTardJSPSolution(
        machine_assignments=machine_assign,
        start_times=start_times,
        total_weighted_tardiness=wt,
    )


def edd_ect(instance: FlexTardJSPInstance) -> FlexTardJSPSolution:
    """EDD dispatching with Earliest Completion Time machine assignment.

    Jobs prioritized by due date. Each operation assigned to the machine
    giving earliest completion.

    Args:
        instance: FlexTardJSPInstance.

    Returns:
        FlexTardJSPSolution.
    """
    n, m = instance.n, instance.m

    machine_assign = [[-1] * instance.num_operations(j) for j in range(n)]
    start_times = [[0.0] * instance.num_operations(j) for j in range(n)]

    # Track next available time per machine
    machine_avail = [0.0] * m
    # Track next operation index per job
    next_op = [0] * n
    # Track completion time of last op per job
    job_avail = [0.0] * n

    scheduled = 0
    total_ops = instance.total_operations()

    while scheduled < total_ops:
        # Find ready operations (jobs that have unscheduled ops)
        ready = []
        for j in range(n):
            if next_op[j] < instance.num_operations(j):
                ready.append(j)

        if not ready:
            break

        # EDD priority: sort by due date
        ready.sort(key=lambda j: instance.due_dates[j])
        j = ready[0]
        o = next_op[j]

        # ECT machine assignment
        best_mach = -1
        best_end = float("inf")
        for mach, dur in instance.operations[j][o]:
            start = max(job_avail[j], machine_avail[mach])
            end = start + dur
            if end < best_end:
                best_end = end
                best_mach = mach

        dur = next(d for mc, d in instance.operations[j][o] if mc == best_mach)
        st = max(job_avail[j], machine_avail[best_mach])

        machine_assign[j][o] = best_mach
        start_times[j][o] = st
        machine_avail[best_mach] = st + dur
        job_avail[j] = st + dur
        next_op[j] = o + 1
        scheduled += 1

    return _build_solution(instance, machine_assign, start_times)


def watc_dispatch(instance: FlexTardJSPInstance) -> FlexTardJSPSolution:
    """Weighted Apparent Tardiness Cost dispatching.

    Priority combines WSPT ratio with urgency (slack relative to due date).

    Args:
        instance: FlexTardJSPInstance.

    Returns:
        FlexTardJSPSolution.
    """
    n, m = instance.n, instance.m

    machine_assign = [[-1] * instance.num_operations(j) for j in range(n)]
    start_times = [[0.0] * instance.num_operations(j) for j in range(n)]

    machine_avail = [0.0] * m
    next_op = [0] * n
    job_avail = [0.0] * n

    # Average processing time for scaling
    avg_p = np.mean([d for j in range(n) for ops in instance.operations[j]
                     for _, d in ops])

    scheduled = 0
    total_ops = instance.total_operations()

    while scheduled < total_ops:
        ready = [j for j in range(n) if next_op[j] < instance.num_operations(j)]
        if not ready:
            break

        t = min(max(job_avail[j], min(machine_avail[mc]
                for mc, _ in instance.operations[j][next_op[j]]))
                for j in ready)

        best_j = None
        best_priority = -float("inf")

        for j in ready:
            o = next_op[j]
            # Min processing time across eligible machines
            min_p = min(d for _, d in instance.operations[j][o])
            # Remaining processing time
            rem_p = sum(min(d for _, d in instance.operations[j][op])
                        for op in range(o, instance.num_operations(j)))
            slack = instance.due_dates[j] - t - rem_p
            # ATC priority
            priority = (instance.weights[j] / max(min_p, 1)) * \
                       np.exp(-max(0, slack) / max(avg_p, 1))
            if priority > best_priority:
                best_priority = priority
                best_j = j

        j = best_j
        o = next_op[j]

        # ECT machine assignment
        best_mach = -1
        best_end = float("inf")
        for mach, dur in instance.operations[j][o]:
            start = max(job_avail[j], machine_avail[mach])
            end = start + dur
            if end < best_end:
                best_end = end
                best_mach = mach

        dur = next(d for mc, d in instance.operations[j][o] if mc == best_mach)
        st = max(job_avail[j], machine_avail[best_mach])

        machine_assign[j][o] = best_mach
        start_times[j][o] = st
        machine_avail[best_mach] = st + dur
        job_avail[j] = st + dur
        next_op[j] = o + 1
        scheduled += 1

    return _build_solution(instance, machine_assign, start_times)


if __name__ == "__main__":
    from instance import small_ftjsp_3x3

    inst = small_ftjsp_3x3()
    sol1 = edd_ect(inst)
    print(f"EDD-ECT: {sol1}")
    sol2 = watc_dispatch(inst)
    print(f"WATC: {sol2}")
