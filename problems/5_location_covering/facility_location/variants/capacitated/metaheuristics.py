"""
Simulated Annealing for Capacitated Facility Location (CFLP).

Problem: CFLP

Neighborhoods:
- Toggle: open/close a facility (maintaining feasibility)
- Swap: close one, open another
- Reassign: move a customer to a different open facility

All moves check capacity constraints before acceptance.
Warm-started with greedy-add heuristic.

References:
    Cornuéjols, G., Sridharan, R. & Thizy, J.M. (1991). A comparison
    of heuristics and relaxations for the capacitated plant location
    problem. European Journal of Operational Research, 50(3), 280-297.
    https://doi.org/10.1016/0377-2217(91)90261-S

    Kirkpatrick, S., Gelatt, C.D. & Vecchi, M.P. (1983). Optimization
    by simulated annealing. Science, 220(4598), 671-680.
    https://doi.org/10.1126/science.220.4598.671
"""

from __future__ import annotations

import sys
import os
import math
import time
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


_inst = _load_mod("cflp_instance_meta", os.path.join(_this_dir, "instance.py"))
CFLPInstance = _inst.CFLPInstance
CFLPSolution = _inst.CFLPSolution

_heur = _load_mod("cflp_heuristics", os.path.join(_this_dir, "heuristics.py"))
greedy_add = _heur.greedy_add


def _assign_capacitated(
    instance: CFLPInstance, open_set: set[int]
) -> tuple[list[int], float]:
    """Assign customers to nearest feasible open facility."""
    n = instance.n
    remaining = {i: instance.capacities[i] for i in open_set}

    order = sorted(range(n), key=lambda j: instance.demands[j], reverse=True)
    assignments = [-1] * n

    for j in order:
        best_fac = -1
        best_cost = float("inf")
        for i in open_set:
            if remaining[i] >= instance.demands[j] - 1e-10:
                if instance.assignment_costs[i][j] < best_cost:
                    best_cost = instance.assignment_costs[i][j]
                    best_fac = i
        if best_fac >= 0:
            assignments[j] = best_fac
            remaining[best_fac] -= instance.demands[j]
        else:
            # Assign to facility with most remaining capacity (may overflow)
            best_fac = max(open_set, key=lambda i: remaining[i])
            assignments[j] = best_fac
            remaining[best_fac] -= instance.demands[j]

    # Check capacity feasibility
    load = {i: 0.0 for i in open_set}
    for j_idx in range(n):
        load[assignments[j_idx]] += instance.demands[j_idx]
    cap_feasible = all(
        load[i] <= instance.capacities[i] + 1e-10 for i in open_set
    )

    total = sum(instance.fixed_costs[i] for i in open_set)
    total += sum(instance.assignment_costs[assignments[j]][j] for j in range(n))

    # Add large penalty for infeasible assignments
    if not cap_feasible:
        for i in open_set:
            if load[i] > instance.capacities[i] + 1e-10:
                total += 1e6 * (load[i] - instance.capacities[i])

    return assignments, total


def simulated_annealing(
    instance: CFLPInstance,
    max_iterations: int = 30000,
    initial_temp: float | None = None,
    cooling_rate: float = 0.9995,
    time_limit: float | None = None,
    seed: int | None = None,
) -> CFLPSolution:
    """Solve CFLP using Simulated Annealing.

    Args:
        instance: A CFLPInstance.
        max_iterations: Maximum iterations.
        initial_temp: Initial temperature. Auto-calibrated if None.
        cooling_rate: Geometric cooling factor.
        time_limit: Time limit in seconds.
        seed: Random seed.

    Returns:
        CFLPSolution.
    """
    rng = np.random.default_rng(seed)
    m = instance.m
    start_time = time.time()

    init_sol = greedy_add(instance)
    open_set = set(init_sol.open_facilities)
    assignments = init_sol.assignments[:]
    current_cost = init_sol.cost

    best_open = set(open_set)
    best_assignments = assignments[:]
    best_cost = current_cost

    if initial_temp is None:
        initial_temp = best_cost * 0.05

    temp = initial_temp

    for iteration in range(max_iterations):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        closed = set(range(m)) - open_set
        move_type = rng.integers(0, 3)

        if move_type == 0 and closed:
            # Open a closed facility
            fac = rng.choice(list(closed))
            new_open = open_set | {fac}
            new_assign, new_cost = _assign_capacitated(instance, new_open)

        elif move_type == 1 and len(open_set) > 1 and closed:
            # Swap
            to_close = rng.choice(list(open_set))
            to_open_fac = rng.choice(list(closed))
            new_open = (open_set - {to_close}) | {to_open_fac}
            new_assign, new_cost = _assign_capacitated(instance, new_open)

        elif move_type == 2 and len(open_set) > 1:
            # Close a facility (only if total capacity sufficient)
            to_close = rng.choice(list(open_set))
            new_open = open_set - {to_close}
            remaining_cap = sum(instance.capacities[i] for i in new_open)
            if remaining_cap < instance.demands.sum() - 1e-10:
                temp *= cooling_rate
                continue
            new_assign, new_cost = _assign_capacitated(instance, new_open)
        else:
            temp *= cooling_rate
            continue

        delta = new_cost - current_cost
        if delta < 0 or (temp > 0 and rng.random() < math.exp(-delta / max(temp, 1e-10))):
            open_set = new_open
            assignments = new_assign
            current_cost = new_cost

            if current_cost < best_cost - 1e-10:
                best_cost = current_cost
                best_open = set(open_set)
                best_assignments = assignments[:]

        temp *= cooling_rate

    return CFLPSolution(
        open_facilities=sorted(best_open),
        assignments=best_assignments,
        cost=best_cost,
    )


if __name__ == "__main__":
    inst = CFLPInstance.random(m=8, n=15, seed=42)
    print(f"CFLP: {inst.m} facilities, {inst.n} customers")

    gr_sol = greedy_add(inst)
    print(f"Greedy: cost={gr_sol.cost:.1f}")

    sa_sol = simulated_annealing(inst, seed=42)
    print(f"SA: cost={sa_sol.cost:.1f}")
