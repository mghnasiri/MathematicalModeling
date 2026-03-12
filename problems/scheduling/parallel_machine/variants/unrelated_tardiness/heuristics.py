"""
Unrelated Parallel Machine with Tardiness — Heuristics.

Algorithms:
    - EDD-ECT: Earliest Due Date with Earliest Completion Time machine.
    - ATC-based dispatching.

References:
    Pinedo, M. (2016). Scheduling: Theory, Algorithms, and Systems.
    5th ed. Springer. https://doi.org/10.1007/978-3-319-26580-3
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


_inst = _load_mod("rm_tard_instance_h", os.path.join(_this_dir, "instance.py"))
RmTardinessInstance = _inst.RmTardinessInstance
RmTardinessSolution = _inst.RmTardinessSolution


def edd_ect(instance: RmTardinessInstance) -> RmTardinessSolution:
    """EDD dispatching with ECT machine assignment.

    Sort jobs by EDD, assign each to machine with earliest completion.

    Args:
        instance: RmTardiness instance.

    Returns:
        RmTardinessSolution.
    """
    n, m = instance.n, instance.m
    job_order = sorted(range(n), key=lambda j: instance.due_dates[j])

    assignment = [-1] * n
    sequence = [[] for _ in range(m)]
    machine_avail = [0.0] * m

    for j in job_order:
        # Assign to machine giving earliest completion
        best_m = -1
        best_end = float("inf")
        for i in range(m):
            end = machine_avail[i] + instance.processing_times[i][j]
            if end < best_end:
                best_end = end
                best_m = i
        assignment[j] = best_m
        sequence[best_m].append(j)
        machine_avail[best_m] = best_end

    total = instance.total_tardiness(assignment, sequence)
    return RmTardinessSolution(assignment=assignment, sequence=sequence,
                                total_tardiness=total)


def atc_dispatch(instance: RmTardinessInstance) -> RmTardinessSolution:
    """ATC-based dispatching for Rm||ΣTj.

    At each step, pick the unscheduled job-machine pair with highest
    ATC priority.

    Args:
        instance: RmTardiness instance.

    Returns:
        RmTardinessSolution.
    """
    n, m = instance.n, instance.m
    assignment = [-1] * n
    sequence = [[] for _ in range(m)]
    machine_avail = [0.0] * m
    scheduled = [False] * n

    avg_p = float(np.mean(instance.processing_times))

    for _ in range(n):
        best_j = -1
        best_m = -1
        best_priority = -float("inf")

        for j in range(n):
            if scheduled[j]:
                continue
            for i in range(m):
                p = instance.processing_times[i][j]
                t = machine_avail[i]
                slack = instance.due_dates[j] - t - p
                priority = (1.0 / max(p, 1)) * np.exp(-max(0, slack) / max(avg_p, 1))
                if priority > best_priority:
                    best_priority = priority
                    best_j = j
                    best_m = i

        assignment[best_j] = best_m
        sequence[best_m].append(best_j)
        machine_avail[best_m] += instance.processing_times[best_m][best_j]
        scheduled[best_j] = True

    total = instance.total_tardiness(assignment, sequence)
    return RmTardinessSolution(assignment=assignment, sequence=sequence,
                                total_tardiness=total)


if __name__ == "__main__":
    from instance import small_rm_tard_6x2

    inst = small_rm_tard_6x2()
    sol1 = edd_ect(inst)
    print(f"EDD-ECT: {sol1}")
    sol2 = atc_dispatch(inst)
    print(f"ATC: {sol2}")
