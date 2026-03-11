"""
Iterated Greedy for p-Median Problem (PMP).

Problem: p-Median

Iterated Greedy repeatedly destroys and reconstructs solutions:
    1. Destroy: replace d random open facilities with random closed ones
    2. Repair: iteratively swap facilities to reduce cost (Teitz-Bart style)
    3. Accept: Boltzmann-based acceptance criterion

Maintains exactly p facilities open at all times.
Warm-started with greedy heuristic.

Complexity: O(iterations * d * m * n) per run.

References:
    Ruiz, R. & Stuetzle, T. (2007). A simple and effective iterated
    greedy algorithm for the permutation flowshop scheduling problem.
    European Journal of Operational Research, 177(3), 2033-2049.
    https://doi.org/10.1016/j.ejor.2005.12.009

    Teitz, M.B. & Bart, P. (1968). Heuristic methods for estimating
    the generalized vertex median of a weighted graph. Operations
    Research, 16(5), 955-961.
    https://doi.org/10.1287/opre.16.5.955
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


_inst = _load_mod("pmed_instance_ig", os.path.join(_parent_dir, "instance.py"))
PMedianInstance = _inst.PMedianInstance
PMedianSolution = _inst.PMedianSolution


def _assign_and_cost(
    instance: PMedianInstance, open_set: set[int]
) -> tuple[list[int], float]:
    """Assign customers to nearest open facility and compute cost."""
    assignments = []
    total = 0.0
    for j in range(instance.n):
        best = min(open_set, key=lambda i: instance.distance_matrix[i][j])
        assignments.append(best)
        total += instance.weights[j] * instance.distance_matrix[best][j]
    return assignments, total


def iterated_greedy(
    instance: PMedianInstance,
    max_iterations: int = 3000,
    d: int | None = None,
    temperature_factor: float = 0.05,
    time_limit: float | None = None,
    seed: int | None = None,
) -> PMedianSolution:
    """Solve p-Median using Iterated Greedy.

    Args:
        instance: A PMedianInstance.
        max_iterations: Maximum number of iterations.
        d: Number of facilities to replace per iteration.
        temperature_factor: Temperature as fraction of initial cost.
        time_limit: Maximum wall-clock time in seconds.
        seed: Random seed for reproducibility.

    Returns:
        PMedianSolution with the best solution found.
    """
    rng = np.random.default_rng(seed)
    m, p = instance.m, instance.p
    start_time = time.time()

    if d is None:
        d = max(1, p // 3)

    # Warm-start with greedy
    _gr = _load_mod(
        "pmed_gr_ig",
        os.path.join(_parent_dir, "heuristics", "greedy_pmedian.py"),
    )
    init_sol = _gr.greedy_pmedian(instance)
    open_set = set(init_sol.open_facilities)
    _, current_cost = _assign_and_cost(instance, open_set)

    best_open = set(open_set)
    best_cost = current_cost

    temperature = temperature_factor * current_cost if current_cost > 0 else 1.0

    for iteration in range(max_iterations):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        # Destroy: replace d random open facilities with random closed ones
        open_list = list(open_set)
        closed_list = list(set(range(m)) - open_set)

        d_actual = min(d, len(open_list), len(closed_list))
        if d_actual == 0:
            continue

        to_close = rng.choice(open_list, size=d_actual, replace=False)
        to_open = rng.choice(closed_list, size=d_actual, replace=False)

        new_open = (open_set - set(to_close)) | set(to_open)

        # Repair: best-improvement swap until no improvement
        improved = True
        while improved:
            improved = False
            _, trial_cost = _assign_and_cost(instance, new_open)
            best_swap = None
            best_swap_cost = trial_cost

            new_closed = set(range(m)) - new_open
            for i_open in list(new_open):
                for i_closed in new_closed:
                    trial = (new_open - {i_open}) | {i_closed}
                    _, cost = _assign_and_cost(instance, trial)
                    if cost < best_swap_cost - 1e-10:
                        best_swap_cost = cost
                        best_swap = (i_open, i_closed)

            if best_swap is not None:
                new_open.remove(best_swap[0])
                new_open.add(best_swap[1])
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
    return PMedianSolution(
        open_facilities=sorted(best_open),
        assignments=assignments,
        cost=final_cost,
    )


if __name__ == "__main__":
    inst = PMedianInstance.random(n=15, m=15, p=3, seed=42)
    print(f"p-Median: {inst.m} sites, {inst.n} customers, p={inst.p}")

    _gr = _load_mod(
        "pmed_gr_ig_main",
        os.path.join(_parent_dir, "heuristics", "greedy_pmedian.py"),
    )
    gr_sol = _gr.greedy_pmedian(inst)
    print(f"Greedy: cost={gr_sol.cost:.1f}")

    ig_sol = iterated_greedy(inst, seed=42)
    print(f"IG: cost={ig_sol.cost:.1f}")
