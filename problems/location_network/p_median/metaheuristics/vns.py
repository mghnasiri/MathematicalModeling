"""
Variable Neighborhood Search for p-Median Problem.

Problem: p-Median Problem (PMP)

VNS uses multiple neighborhood structures to escape local optima while
maintaining exactly p open facilities:
    N1: Swap — close one open, open one closed
    N2: Double swap — perform two swaps simultaneously
    N3: Triple perturbation — swap three facilities

Local search uses best-improvement single swap.
Warm-started with greedy + interchange heuristic.

Complexity: O(iterations * m^2 * n) per run.

References:
    Mladenović, N. & Hansen, P. (1997). Variable neighborhood search.
    Computers & Operations Research, 24(11), 1097-1100.
    https://doi.org/10.1016/S0305-0548(97)00031-2

    Hansen, P. & Mladenović, N. (1997). Variable neighborhood search
    for the p-median. Location Science, 5(4), 207-226.
    https://doi.org/10.1016/S0966-8349(98)00030-8
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


_inst = _load_module("pmedian_instance_vns", os.path.join(_parent_dir, "instance.py"))
PMedianInstance = _inst.PMedianInstance
PMedianSolution = _inst.PMedianSolution


def _assign_and_cost(instance: PMedianInstance, open_set: set[int]) -> tuple[list[int], float]:
    """Assign customers to nearest open facility."""
    assignments = []
    total = 0.0
    for j in range(instance.n):
        best = min(open_set, key=lambda i: instance.distance_matrix[i][j])
        assignments.append(best)
        total += instance.weights[j] * instance.distance_matrix[best][j]
    return assignments, total


def vns(
    instance: PMedianInstance,
    max_iterations: int = 500,
    k_max: int = 3,
    time_limit: float | None = None,
    seed: int | None = None,
) -> PMedianSolution:
    """Solve p-Median using Variable Neighborhood Search.

    Args:
        instance: A PMedianInstance.
        max_iterations: Maximum number of VNS iterations.
        k_max: Number of neighborhood structures (1-3).
        time_limit: Maximum wall-clock time in seconds.
        seed: Random seed for reproducibility.

    Returns:
        PMedianSolution with the best configuration found.
    """
    rng = np.random.default_rng(seed)
    m, p = instance.m, instance.p
    start_time = time.time()

    # Warm-start with greedy + interchange
    _greedy_mod = _load_module(
        "pm_greedy_vns",
        os.path.join(_parent_dir, "heuristics", "greedy_pmedian.py"),
    )
    init_sol = _greedy_mod.interchange(instance)
    open_set = set(init_sol.open_facilities)
    _, current_cost = _assign_and_cost(instance, open_set)

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
            shaken = _shake(m, p, open_set, k, rng)

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
    assignments, cost = _assign_and_cost(instance, best_open)
    return PMedianSolution(
        open_facilities=open_list,
        assignments=assignments,
        cost=cost,
    )


def _shake(m: int, p: int, open_set: set[int], k: int, rng: np.random.Generator) -> set[int]:
    """Random perturbation maintaining exactly p open facilities."""
    new_open = set(open_set)
    closed = [i for i in range(m) if i not in new_open]
    open_list = list(new_open)

    swaps = min(k, len(open_list), len(closed))
    if swaps == 0:
        return new_open

    to_close = rng.choice(open_list, size=swaps, replace=False)
    to_open = rng.choice(closed, size=swaps, replace=False)

    for c in to_close:
        new_open.remove(c)
    for o in to_open:
        new_open.add(o)

    return new_open


def _local_search(
    instance: PMedianInstance, open_set: set[int]
) -> tuple[set[int], float]:
    """Best-improvement swap local search."""
    m = instance.m
    current = set(open_set)
    _, current_cost = _assign_and_cost(instance, current)

    improved = True
    while improved:
        improved = False
        best_delta = 0.0
        best_out = -1
        best_in = -1

        closed = [i for i in range(m) if i not in current]
        for out_fac in list(current):
            for in_fac in closed:
                trial = (current - {out_fac}) | {in_fac}
                _, trial_cost = _assign_and_cost(instance, trial)
                delta = trial_cost - current_cost
                if delta < best_delta - 1e-10:
                    best_delta = delta
                    best_out = out_fac
                    best_in = in_fac

        if best_out >= 0:
            current.remove(best_out)
            current.add(best_in)
            current_cost += best_delta
            improved = True

    _, actual_cost = _assign_and_cost(instance, current)
    return current, actual_cost


if __name__ == "__main__":
    from instance import small_pmedian_6_2

    inst = small_pmedian_6_2()
    sol = vns(inst, seed=42)
    print(f"VNS: {sol}")
