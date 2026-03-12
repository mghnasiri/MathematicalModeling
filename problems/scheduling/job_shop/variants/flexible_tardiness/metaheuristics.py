"""
Flexible Job Shop with Tardiness — Metaheuristics.

Algorithms:
    - Simulated Annealing with machine reassignment and operation swap.

References:
    Mastrolilli, M. & Gambardella, L.M. (2000). Effective neighbourhood
    functions for the flexible job shop problem. Journal of Scheduling,
    3(1), 3-20.
    https://doi.org/10.1002/(SICI)1099-1425(200001/02)3:1<3::AID-JOS32>3.0.CO;2-Y
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


_inst = _load_mod("ftjsp_instance_m", os.path.join(_this_dir, "instance.py"))
FlexTardJSPInstance = _inst.FlexTardJSPInstance
FlexTardJSPSolution = _inst.FlexTardJSPSolution

_heur = _load_mod("ftjsp_heuristics_m", os.path.join(_this_dir, "heuristics.py"))
edd_ect = _heur.edd_ect


def _schedule_from_encoding(
    instance: FlexTardJSPInstance,
    priority: list[tuple[int, int]],
    machine_choices: list[list[int]],
) -> FlexTardJSPSolution:
    """Build schedule from priority list and machine assignments."""
    n, m = instance.n, instance.m

    machine_assign = [list(mc) for mc in machine_choices]
    start_times = [[0.0] * instance.num_operations(j) for j in range(n)]
    machine_avail = [0.0] * m
    job_avail = [0.0] * n

    for j, o in priority:
        mach = machine_assign[j][o]
        dur = next(d for mc, d in instance.operations[j][o] if mc == mach)
        st = max(job_avail[j], machine_avail[mach])
        start_times[j][o] = st
        machine_avail[mach] = st + dur
        job_avail[j] = st + dur

    # Compute weighted tardiness
    wt = 0.0
    for j in range(n):
        last_o = instance.num_operations(j) - 1
        mach = machine_assign[j][last_o]
        dur = next(d for mc, d in instance.operations[j][last_o] if mc == mach)
        cj = start_times[j][last_o] + dur
        tj = max(0.0, cj - instance.due_dates[j])
        wt += instance.weights[j] * tj

    return FlexTardJSPSolution(
        machine_assignments=machine_assign,
        start_times=start_times,
        total_weighted_tardiness=wt,
    )


def simulated_annealing(
    instance: FlexTardJSPInstance,
    max_iterations: int = 50000,
    cooling_rate: float = 0.9995,
    seed: int | None = None,
    time_limit: float | None = None,
) -> FlexTardJSPSolution:
    """SA for flexible JSP with tardiness objective.

    Moves: swap adjacent operations in priority list, reassign machine.

    Args:
        instance: FlexTardJSPInstance.
        max_iterations: Maximum iterations.
        cooling_rate: Temperature decay factor.
        seed: Random seed.
        time_limit: Time limit in seconds.

    Returns:
        FlexTardJSPSolution.
    """
    rng = np.random.default_rng(seed)
    n, m = instance.n, instance.m

    # Initialize from EDD-ECT
    init = edd_ect(instance)

    # Build priority list (topological order respecting job precedence)
    # from start times
    ops = []
    for j in range(n):
        for o in range(instance.num_operations(j)):
            ops.append((init.start_times[j][o], j, o))
    ops.sort()
    priority = [(j, o) for _, j, o in ops]

    machine_choices = [list(ma) for ma in init.machine_assignments]
    cost = init.total_weighted_tardiness

    best_priority = list(priority)
    best_machines = [list(mc) for mc in machine_choices]
    best_cost = cost

    temp = max(1.0, best_cost * 0.2) if best_cost > 0 else 5.0
    start = time.time()

    for it in range(max_iterations):
        if time_limit and time.time() - start > time_limit:
            break

        new_priority = list(priority)
        new_machines = [list(mc) for mc in machine_choices]
        move = rng.integers(0, 2)

        if move == 0:
            # Swap two adjacent operations (respecting job precedence)
            idx = int(rng.integers(0, len(new_priority) - 1))
            j1, o1 = new_priority[idx]
            j2, o2 = new_priority[idx + 1]

            # Only swap if not from the same job (preserving job precedence)
            if j1 != j2:
                new_priority[idx], new_priority[idx + 1] = \
                    new_priority[idx + 1], new_priority[idx]
            else:
                temp *= cooling_rate
                continue

            # Verify job precedence
            pos = {(j, o): i for i, (j, o) in enumerate(new_priority)}
            valid = True
            for j in range(n):
                for o in range(1, instance.num_operations(j)):
                    if pos[(j, o)] < pos[(j, o - 1)]:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                temp *= cooling_rate
                continue

        elif move == 1:
            # Reassign machine for a random operation
            j = int(rng.integers(0, n))
            o = int(rng.integers(0, instance.num_operations(j)))
            alts = instance.operations[j][o]
            if len(alts) > 1:
                new_mach = rng.choice([mc for mc, _ in alts])
                new_machines[j][o] = int(new_mach)

        sol = _schedule_from_encoding(instance, new_priority, new_machines)
        new_cost = sol.total_weighted_tardiness
        delta = new_cost - cost

        if delta < 0 or rng.random() < math.exp(-delta / max(temp, 1e-15)):
            priority = new_priority
            machine_choices = new_machines
            cost = new_cost
            if cost < best_cost:
                best_cost = cost
                best_priority = list(priority)
                best_machines = [list(mc) for mc in machine_choices]

        temp *= cooling_rate

    return _schedule_from_encoding(instance, best_priority, best_machines)


if __name__ == "__main__":
    from instance import small_ftjsp_3x3

    inst = small_ftjsp_3x3()
    sol = simulated_annealing(inst, max_iterations=10000, seed=42)
    print(f"SA: {sol}")
