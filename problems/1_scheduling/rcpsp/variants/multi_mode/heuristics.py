"""
Multi-Mode RCPSP — Heuristics.

Algorithms:
    - Serial SGS with mode selection (shortest duration mode first).
    - Serial SGS with resource-aware mode selection.

References:
    Sprecher, A. & Drexl, A. (1998). Multi-mode resource-constrained
    project scheduling by a simple, general and powerful sequencing
    algorithm. European Journal of Operational Research, 107(2), 431-450.
    https://doi.org/10.1016/S0377-2217(97)00348-2
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


_inst = _load_mod("mrcpsp_instance_h", os.path.join(_this_dir, "instance.py"))
MRCPSPInstance = _inst.MRCPSPInstance
MRCPSPSolution = _inst.MRCPSPSolution


def serial_sgs_shortest(instance: MRCPSPInstance) -> MRCPSPSolution:
    """Serial SGS selecting shortest-duration mode for each activity.

    Activities scheduled by topological order (LFT-like priority).

    Args:
        instance: MRCPSP instance.

    Returns:
        MRCPSPSolution.
    """
    n = instance.n
    total = n + 2
    preds = instance.predecessors()

    # Choose shortest-duration mode for each activity
    mode_assign = []
    for j in range(total):
        best_mode = min(range(len(instance.modes[j])),
                        key=lambda m: instance.modes[j][m][0])
        mode_assign.append(best_mode)

    # Topological sort by number of predecessors (simple BFS)
    in_deg = [len(preds[j]) for j in range(total)]
    queue = [j for j in range(total) if in_deg[j] == 0]
    topo = []
    while queue:
        queue.sort()
        j = queue.pop(0)
        topo.append(j)
        for s in instance.successors[j]:
            in_deg[s] -= 1
            if in_deg[s] == 0:
                queue.append(s)

    # Schedule using serial SGS
    start_times = [0] * total
    # Track resource usage over time
    max_horizon = sum(instance.modes[j][mode_assign[j]][0] for j in range(total)) + 1
    usage = np.zeros((max_horizon, instance.num_resources), dtype=int)

    for j in topo:
        dur, reqs = instance.modes[j][mode_assign[j]]
        # Earliest start: after all predecessors finish
        es = 0
        for p in preds[j]:
            p_dur = instance.modes[p][mode_assign[p]][0]
            es = max(es, start_times[p] + p_dur)

        # Find earliest feasible start (resource-feasible)
        t = es
        while t + dur <= max_horizon:
            feasible = True
            for tt in range(t, t + dur):
                for r in range(instance.num_resources):
                    if usage[tt][r] + reqs[r] > instance.resource_capacities[r]:
                        feasible = False
                        break
                if not feasible:
                    break
            if feasible:
                break
            t += 1

        start_times[j] = t
        for tt in range(t, t + dur):
            usage[tt] += reqs

    makespan = max(start_times[j] + instance.modes[j][mode_assign[j]][0]
                   for j in range(total))

    return MRCPSPSolution(
        mode_assignments=mode_assign,
        start_times=start_times,
        makespan=makespan,
    )


if __name__ == "__main__":
    from instance import small_mrcpsp_4

    inst = small_mrcpsp_4()
    sol = serial_sgs_shortest(inst)
    print(f"Serial SGS (shortest): {sol}")
    print(f"  Start times: {sol.start_times}")
    print(f"  Mode assignments: {sol.mode_assignments}")
