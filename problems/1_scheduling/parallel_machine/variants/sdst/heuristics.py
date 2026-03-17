"""
Constructive Heuristics for Parallel Machine with SDST.

Problem: Rm | Ssd | Cmax
Complexity: O(n^2 * m)

1. Greedy ECT-SDST: assign each job to the machine where it finishes
   earliest (considering setup from last job on that machine).
2. LPT-SDST: sort by average processing time descending, assign with
   setup-aware ECT.

References:
    Rabadi, G., Moraga, R.J. & Al-Salem, A. (2006). Heuristics for the
    unrelated parallel machine scheduling problem with setup times.
    Journal of Intelligent Manufacturing, 17(2), 199-207.
    https://doi.org/10.1007/s10845-005-6636-x
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


_inst = _load_mod("pmsdst_instance_h", os.path.join(_this_dir, "instance.py"))
PMSDSTInstance = _inst.PMSDSTInstance
PMSDSTSolution = _inst.PMSDSTSolution


def greedy_ect_sdst(instance: PMSDSTInstance) -> PMSDSTSolution:
    """Assign jobs greedily to earliest-completing machine with setup.

    For each unassigned job, compute completion on each machine
    (including setup from last job) and assign to minimum.

    Args:
        instance: A PMSDSTInstance.

    Returns:
        PMSDSTSolution.
    """
    n, m = instance.n, instance.m
    schedule: list[list[int]] = [[] for _ in range(m)]
    completion = np.zeros(m)
    assigned = [False] * n

    for _ in range(n):
        best_job = -1
        best_machine = -1
        best_finish = float("inf")

        for j in range(n):
            if assigned[j]:
                continue
            for k in range(m):
                if schedule[k]:
                    prev = schedule[k][-1]
                    setup = instance.setup_times[prev][j][k]
                else:
                    setup = instance.setup_times[j][j][k]
                finish = completion[k] + setup + instance.processing_times[j][k]
                if finish < best_finish:
                    best_finish = finish
                    best_job = j
                    best_machine = k

        schedule[best_machine].append(best_job)
        if len(schedule[best_machine]) == 1:
            setup = instance.setup_times[best_job][best_job][best_machine]
        else:
            prev = schedule[best_machine][-2]
            setup = instance.setup_times[prev][best_job][best_machine]
        completion[best_machine] += setup + instance.processing_times[best_job][best_machine]
        assigned[best_job] = True

    return PMSDSTSolution(
        schedule=schedule,
        makespan=instance.makespan(schedule),
    )


def lpt_sdst(instance: PMSDSTInstance) -> PMSDSTSolution:
    """LPT order with setup-aware ECT assignment.

    Sort jobs by average processing time (descending), then assign
    each to the machine with earliest completion including setup.

    Args:
        instance: A PMSDSTInstance.

    Returns:
        PMSDSTSolution.
    """
    n, m = instance.n, instance.m
    avg_proc = instance.processing_times.mean(axis=1)
    order = sorted(range(n), key=lambda j: avg_proc[j], reverse=True)

    schedule: list[list[int]] = [[] for _ in range(m)]
    completion = np.zeros(m)

    for j in order:
        best_machine = -1
        best_finish = float("inf")

        for k in range(m):
            if schedule[k]:
                prev = schedule[k][-1]
                setup = instance.setup_times[prev][j][k]
            else:
                setup = instance.setup_times[j][j][k]
            finish = completion[k] + setup + instance.processing_times[j][k]
            if finish < best_finish:
                best_finish = finish
                best_machine = k

        schedule[best_machine].append(j)
        if len(schedule[best_machine]) == 1:
            setup = instance.setup_times[j][j][best_machine]
        else:
            prev = schedule[best_machine][-2]
            setup = instance.setup_times[prev][j][best_machine]
        completion[best_machine] += setup + instance.processing_times[j][best_machine]

    return PMSDSTSolution(
        schedule=schedule,
        makespan=instance.makespan(schedule),
    )


if __name__ == "__main__":
    inst = _inst.small_pmsdst_4_2()
    print(f"PM-SDST: {inst.n} jobs, {inst.m} machines")

    sol1 = greedy_ect_sdst(inst)
    print(f"ECT-SDST: makespan={sol1.makespan:.1f}")

    sol2 = lpt_sdst(inst)
    print(f"LPT-SDST: makespan={sol2.makespan:.1f}")
