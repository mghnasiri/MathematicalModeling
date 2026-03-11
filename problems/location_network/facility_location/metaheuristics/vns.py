"""
Variable Neighborhood Search for Facility Location (UFLP).

Problem: Uncapacitated Facility Location Problem (UFLP)

VNS uses multiple neighborhood structures to escape local optima:
    N1: Toggle — open or close a single facility
    N2: Swap — close one open facility, open one closed facility
    N3: Multi-toggle — toggle two facilities simultaneously

Local search uses best-improvement toggle moves.
Warm-started with greedy add heuristic.

Complexity: O(iterations * m * n) per run.

References:
    Mladenović, N. & Hansen, P. (1997). Variable neighborhood search.
    Computers & Operations Research, 24(11), 1097-1100.
    https://doi.org/10.1016/S0305-0548(97)00031-2

    Hansen, P., Mladenović, N. & Pérez, J.A.M. (2010). Variable
    neighbourhood search: methods and applications. Annals of Operations
    Research, 175(1), 367-407.
    https://doi.org/10.1007/s10479-009-0657-6
"""

from __future__ import annotations

import os
import time
import numpy as np

_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_module(name, filepath):
    import importlib.util
    import sys as _sys
    if name in _sys.modules:
        return _sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    _sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_inst = _load_module("fl_instance_vns", os.path.join(_parent_dir, "instance.py"))
FacilityLocationInstance = _inst.FacilityLocationInstance
FacilityLocationSolution = _inst.FacilityLocationSolution


def vns(
    instance: FacilityLocationInstance,
    max_iterations: int = 500,
    k_max: int = 3,
    time_limit: float | None = None,
    seed: int | None = None,
) -> FacilityLocationSolution:
    """Solve UFLP using Variable Neighborhood Search.

    Args:
        instance: A FacilityLocationInstance.
        max_iterations: Maximum number of VNS iterations.
        k_max: Number of neighborhood structures (1-3).
        time_limit: Maximum wall-clock time in seconds.
        seed: Random seed for reproducibility.

    Returns:
        FacilityLocationSolution with the best configuration found.
    """
    rng = np.random.default_rng(seed)
    m = instance.m
    n = instance.n
    start_time = time.time()

    # Warm-start with greedy add
    _greedy_mod = _load_module(
        "fl_greedy_vns",
        os.path.join(_parent_dir, "heuristics", "greedy_facility.py"),
    )
    init_sol = _greedy_mod.greedy_add(instance)
    open_set = set(init_sol.open_facilities)
    current_cost = init_sol.cost

    best_open = set(open_set)
    best_cost = current_cost

    for iteration in range(max_iterations):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        k = 1
        while k <= k_max:
            if time_limit is not None and time.time() - start_time >= time_limit:
                break

            # Shaking
            shaken = _shake(m, open_set, k, rng)

            # Local search
            ls_open, ls_cost = _local_search(instance, shaken)

            if ls_cost < current_cost - 1e-10:
                open_set = ls_open
                current_cost = ls_cost
                k = 1

                if current_cost < best_cost - 1e-10:
                    best_cost = current_cost
                    best_open = set(open_set)
            else:
                k += 1

    open_list = sorted(best_open)
    assignments = _compute_assignments(instance, open_list)
    actual_cost = instance.total_cost(open_list, assignments)
    return FacilityLocationSolution(
        open_facilities=open_list,
        assignments=assignments,
        cost=actual_cost,
    )


def _evaluate(instance: FacilityLocationInstance, open_set: set[int]) -> float:
    """Compute total cost for a set of open facilities."""
    if not open_set:
        return float("inf")
    open_list = sorted(open_set)
    assignments = _compute_assignments(instance, open_list)
    return instance.total_cost(open_list, assignments)


def _compute_assignments(
    instance: FacilityLocationInstance, open_facilities: list[int]
) -> list[int]:
    """Assign each customer to nearest open facility."""
    assignments = []
    for j in range(instance.n):
        best_fac = open_facilities[0]
        best_cost = instance.assignment_costs[best_fac, j]
        for f in open_facilities[1:]:
            c = instance.assignment_costs[f, j]
            if c < best_cost:
                best_cost = c
                best_fac = f
        assignments.append(best_fac)
    return assignments


def _shake(m: int, open_set: set[int], k: int, rng: np.random.Generator) -> set[int]:
    """Random perturbation in neighborhood k."""
    new_open = set(open_set)

    if k == 1:
        # Toggle a random facility
        fac = rng.integers(0, m)
        if fac in new_open:
            if len(new_open) > 1:
                new_open.remove(fac)
        else:
            new_open.add(fac)

    elif k == 2:
        # Swap: close one, open another
        open_list = list(new_open)
        closed_list = [i for i in range(m) if i not in new_open]
        if open_list and closed_list:
            to_close = rng.choice(open_list)
            to_open = rng.choice(closed_list)
            new_open.remove(to_close)
            new_open.add(to_open)
        if not new_open:
            new_open.add(rng.integers(0, m))

    elif k == 3:
        # Multi-toggle: toggle two random facilities
        facs = rng.choice(m, size=min(2, m), replace=False)
        for fac in facs:
            if fac in new_open:
                if len(new_open) > 1:
                    new_open.remove(fac)
            else:
                new_open.add(fac)

    return new_open


def _local_search(
    instance: FacilityLocationInstance, open_set: set[int]
) -> tuple[set[int], float]:
    """Best-improvement local search with toggle neighborhood."""
    m = instance.m
    current = set(open_set)
    current_cost = _evaluate(instance, current)

    improved = True
    while improved:
        improved = False
        best_delta = 0.0
        best_fac = -1

        for fac in range(m):
            trial = set(current)
            if fac in trial:
                if len(trial) <= 1:
                    continue
                trial.remove(fac)
            else:
                trial.add(fac)

            trial_cost = _evaluate(instance, trial)
            delta = trial_cost - current_cost
            if delta < best_delta - 1e-10:
                best_delta = delta
                best_fac = fac

        if best_fac >= 0:
            if best_fac in current:
                current.remove(best_fac)
            else:
                current.add(best_fac)
            current_cost += best_delta
            improved = True

    return current, _evaluate(instance, current)


if __name__ == "__main__":
    from instance import small_uflp_3_5, medium_uflp_5_10

    print("=== VNS for Facility Location ===\n")

    for name, inst_fn in [
        ("small_3_5", small_uflp_3_5),
        ("medium_5_10", medium_uflp_5_10),
    ]:
        inst = inst_fn()
        sol = vns(inst, seed=42)
        print(f"{name}: cost={sol.cost:.2f}, open={sol.open_facilities}")
