"""
Multi-Mode RCPSP — Metaheuristics.

Algorithms:
    - SA with mode switching and activity swapping.

References:
    Hartmann, S. & Briskorn, D. (2010). A survey of variants and
    extensions of the resource-constrained project scheduling problem.
    https://doi.org/10.1016/j.ejor.2009.11.005
"""

from __future__ import annotations

import math
import sys
import os
import importlib.util
import time

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


_inst = _load_mod("mrcpsp_instance_m", os.path.join(_this_dir, "instance.py"))
MRCPSPInstance = _inst.MRCPSPInstance
MRCPSPSolution = _inst.MRCPSPSolution

_heur = _load_mod("mrcpsp_heuristics_m", os.path.join(_this_dir, "heuristics.py"))
serial_sgs_shortest = _heur.serial_sgs_shortest


def _decode(instance: MRCPSPInstance, activity_list: list[int],
            mode_assign: list[int]) -> MRCPSPSolution:
    """Decode activity list + mode assignment into a schedule via serial SGS."""
    n = instance.n
    total = n + 2
    preds = instance.predecessors()

    max_horizon = sum(instance.modes[j][mode_assign[j]][0] for j in range(total)) + 1
    usage = np.zeros((max_horizon, instance.num_resources), dtype=int)
    start_times = [0] * total

    for j in activity_list:
        dur, reqs = instance.modes[j][mode_assign[j]]
        es = 0
        for p in preds[j]:
            p_dur = instance.modes[p][mode_assign[p]][0]
            es = max(es, start_times[p] + p_dur)

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
    return MRCPSPSolution(mode_assignments=list(mode_assign),
                          start_times=start_times, makespan=makespan)


def _is_precedence_feasible(instance: MRCPSPInstance, activity_list: list[int]) -> bool:
    """Check if activity list respects precedence."""
    pos = {j: i for i, j in enumerate(activity_list)}
    for j in range(instance.n + 2):
        for s in instance.successors[j]:
            if pos.get(j, 0) >= pos.get(s, 0):
                return False
    return True


def simulated_annealing(
    instance: MRCPSPInstance,
    max_iterations: int = 50000,
    cooling_rate: float = 0.9995,
    seed: int | None = None,
    time_limit: float | None = None,
) -> MRCPSPSolution:
    """SA for MRCPSP with mode switching and activity reordering."""
    rng = np.random.default_rng(seed)
    n = instance.n
    total = n + 2

    init = serial_sgs_shortest(instance)

    # Build initial activity list (topological)
    preds = instance.predecessors()
    in_deg = [len(preds[j]) for j in range(total)]
    queue = [j for j in range(total) if in_deg[j] == 0]
    act_list = []
    while queue:
        queue.sort()
        j = queue.pop(0)
        act_list.append(j)
        for s in instance.successors[j]:
            in_deg[s] -= 1
            if in_deg[s] == 0:
                queue.append(s)

    mode_assign = list(init.mode_assignments)
    cost = init.makespan

    best_act_list = list(act_list)
    best_mode = list(mode_assign)
    best_cost = cost

    temp = best_cost * 0.2
    start = time.time()

    for it in range(max_iterations):
        if time_limit and time.time() - start > time_limit:
            break

        new_act = list(act_list)
        new_mode = list(mode_assign)
        move = rng.integers(0, 2)

        if move == 0:
            # Swap two adjacent activities in the list (if precedence allows)
            idx = rng.integers(1, total - 1)
            new_act[idx], new_act[idx - 1] = new_act[idx - 1], new_act[idx]
            if not _is_precedence_feasible(instance, new_act):
                continue

        elif move == 1:
            # Switch mode of a random real activity
            j = rng.integers(1, n + 1)
            if len(instance.modes[j]) > 1:
                new_mode[j] = rng.integers(0, len(instance.modes[j]))

        sol = _decode(instance, new_act, new_mode)
        new_cost = sol.makespan
        delta = new_cost - cost

        if delta < 0 or rng.random() < math.exp(-delta / max(temp, 1e-15)):
            act_list = new_act
            mode_assign = new_mode
            cost = new_cost
            if cost < best_cost:
                best_cost = cost
                best_act_list = list(act_list)
                best_mode = list(mode_assign)

        temp *= cooling_rate

    return _decode(instance, best_act_list, best_mode)


if __name__ == "__main__":
    from instance import small_mrcpsp_4

    inst = small_mrcpsp_4()
    sol = simulated_annealing(inst, max_iterations=10000, seed=42)
    print(f"SA: {sol}")
