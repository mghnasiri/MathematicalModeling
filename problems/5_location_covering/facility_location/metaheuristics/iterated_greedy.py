"""
Iterated Greedy for Uncapacitated Facility Location (UFLP).

Problem: UFLP (Uncapacitated Facility Location)

Iterated Greedy repeatedly destroys and reconstructs solutions:
    1. Destroy: close d random open facilities
    2. Repair: greedily reopen facilities that reduce cost
    3. Accept: Boltzmann-based acceptance criterion

Warm-started with greedy-add heuristic.

Complexity: O(iterations * d * m * n) per run.

References:
    Ruiz, R. & Stuetzle, T. (2007). A simple and effective iterated
    greedy algorithm for the permutation flowshop scheduling problem.
    European Journal of Operational Research, 177(3), 2033-2049.
    https://doi.org/10.1016/j.ejor.2005.12.009

    Cornuéjols, G., Nemhauser, G.L. & Wolsey, L.A. (1990). The
    uncapacitated facility location problem. In: Mirchandani, P.B.
    & Francis, R.L. (eds) Discrete Location Theory, Wiley, 119-171.
"""

from __future__ import annotations

import sys
import os
import math
import time
import importlib.util

import numpy as np

_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_mod(name, filepath):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_inst = _load_mod("fl_instance_ig", os.path.join(_parent_dir, "instance.py"))
FacilityLocationInstance = _inst.FacilityLocationInstance
FacilityLocationSolution = _inst.FacilityLocationSolution


def _assign_and_cost(
    instance: FacilityLocationInstance, open_set: set[int]
) -> tuple[list[int], float]:
    """Assign customers and compute total cost."""
    assignments = []
    total = sum(instance.fixed_costs[i] for i in open_set)
    for j in range(instance.n):
        best = min(open_set, key=lambda i: instance.assignment_costs[i][j])
        assignments.append(best)
        total += instance.assignment_costs[best][j]
    return assignments, total


def iterated_greedy(
    instance: FacilityLocationInstance,
    max_iterations: int = 3000,
    d: int | None = None,
    temperature_factor: float = 0.05,
    time_limit: float | None = None,
    seed: int | None = None,
) -> FacilityLocationSolution:
    """Solve UFLP using Iterated Greedy.

    Args:
        instance: A FacilityLocationInstance.
        max_iterations: Maximum number of iterations.
        d: Number of facilities to close per iteration.
        temperature_factor: Temperature as fraction of initial cost.
        time_limit: Maximum wall-clock time in seconds.
        seed: Random seed for reproducibility.

    Returns:
        FacilityLocationSolution with the best solution found.
    """
    rng = np.random.default_rng(seed)
    m = instance.m
    start_time = time.time()

    if d is None:
        d = max(1, m // 4)

    # Warm-start with greedy-add
    _gr = _load_mod(
        "fl_gr_ig",
        os.path.join(_parent_dir, "heuristics", "greedy_facility.py"),
    )
    init_sol = _gr.greedy_add(instance)
    open_set = set(init_sol.open_facilities)
    _, current_cost = _assign_and_cost(instance, open_set)

    best_open = set(open_set)
    best_cost = current_cost

    temperature = temperature_factor * current_cost if current_cost > 0 else 1.0

    for iteration in range(max_iterations):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        # Destroy: close d random open facilities (keep at least 1)
        open_list = list(open_set)
        d_actual = min(d, len(open_list) - 1)
        if d_actual <= 0:
            # Only 1 facility open, try adding instead
            closed = set(range(m)) - open_set
            if closed:
                to_add = rng.choice(list(closed))
                open_set.add(to_add)
            continue

        to_close = rng.choice(open_list, size=d_actual, replace=False)
        new_open = open_set - set(to_close)

        # Repair: greedily add facilities that reduce cost
        closed = set(range(m)) - new_open
        improved = True
        while improved:
            improved = False
            _, current_trial_cost = _assign_and_cost(instance, new_open)
            best_add = None
            best_add_cost = current_trial_cost

            for candidate in closed:
                trial = new_open | {candidate}
                _, cost = _assign_and_cost(instance, trial)
                if cost < best_add_cost - 1e-10:
                    best_add_cost = cost
                    best_add = candidate

            if best_add is not None:
                new_open.add(best_add)
                closed.remove(best_add)
                improved = True

        _, new_cost = _assign_and_cost(instance, new_open)

        # Acceptance
        delta = new_cost - current_cost
        if delta < 0 or (temperature > 0 and rng.random() < math.exp(-delta / temperature)):
            open_set = new_open
            current_cost = new_cost

            if current_cost < best_cost - 1e-10:
                best_cost = current_cost
                best_open = set(open_set)

    assignments, final_cost = _assign_and_cost(instance, best_open)
    return FacilityLocationSolution(
        open_facilities=sorted(best_open),
        assignments=assignments,
        cost=final_cost,
    )


if __name__ == "__main__":
    inst = FacilityLocationInstance.random(m=8, n=15, seed=42)
    print(f"UFLP: {inst.m} facilities, {inst.n} customers")

    _gr = _load_mod(
        "fl_gr_ig_main",
        os.path.join(_parent_dir, "heuristics", "greedy_facility.py"),
    )
    gr_sol = _gr.greedy_add(inst)
    print(f"Greedy Add: cost={gr_sol.cost:.1f}")

    ig_sol = iterated_greedy(inst, seed=42)
    print(f"IG: cost={ig_sol.cost:.1f}")
