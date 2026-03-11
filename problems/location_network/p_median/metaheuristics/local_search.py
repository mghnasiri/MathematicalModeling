"""
Local Search for p-Median Problem (PMP).

Neighborhood: swap — close one open facility and open a closed one,
maintaining exactly p open facilities.

Uses best-improvement search with random restart on stagnation.
Warm-started with greedy heuristic.

Complexity: O(iterations * m * p * n) per run.

References:
    Teitz, M.B. & Bart, P. (1968). Heuristic methods for estimating
    the generalized vertex median of a weighted graph. Operations
    Research, 16(5), 955-961.
    https://doi.org/10.1287/opre.16.5.955

    Resende, M.G.C. & Werneck, R.F. (2004). A hybrid heuristic for
    the p-median problem. Journal of Heuristics, 10(1), 59-88.
    https://doi.org/10.1023/B:HEUR.0000019986.96257.50
"""

from __future__ import annotations

import sys
import os
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


_inst = _load_mod("pmed_instance_ls", os.path.join(_parent_dir, "instance.py"))
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


def local_search(
    instance: PMedianInstance,
    max_iterations: int = 1000,
    time_limit: float | None = None,
    seed: int | None = None,
) -> PMedianSolution:
    """Solve p-Median using local search with swap neighborhood.

    Args:
        instance: A PMedianInstance.
        max_iterations: Maximum number of iterations.
        time_limit: Maximum wall-clock time in seconds.
        seed: Random seed for reproducibility.

    Returns:
        PMedianSolution with the best solution found.
    """
    rng = np.random.default_rng(seed)
    m, p = instance.m, instance.p
    start_time = time.time()

    # Warm-start with greedy
    _gr = _load_mod(
        "pmed_gr_ls",
        os.path.join(_parent_dir, "heuristics", "greedy_pmedian.py"),
    )
    init_sol = _gr.greedy_pmedian(instance)
    open_set = set(init_sol.open_facilities)
    _, current_cost = _assign_and_cost(instance, open_set)

    best_open = set(open_set)
    best_cost = current_cost

    for iteration in range(max_iterations):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        closed = set(range(m)) - open_set
        best_delta = 0.0
        best_swap = None

        # Swap neighborhood: close one, open one
        for i_open in list(open_set):
            for i_closed in closed:
                trial = (open_set - {i_open}) | {i_closed}
                _, cost = _assign_and_cost(instance, trial)
                delta = cost - current_cost
                if delta < best_delta - 1e-10:
                    best_delta = delta
                    best_swap = (i_open, i_closed)

        if best_swap is not None:
            open_set.remove(best_swap[0])
            open_set.add(best_swap[1])
            current_cost += best_delta

            if current_cost < best_cost - 1e-10:
                best_cost = current_cost
                best_open = set(open_set)
        else:
            # Random perturbation
            open_list = list(open_set)
            closed_list = list(closed) if closed else []
            if closed_list:
                to_close = open_list[rng.integers(len(open_list))]
                to_open = closed_list[rng.integers(len(closed_list))]
                open_set.remove(to_close)
                open_set.add(to_open)
                _, current_cost = _assign_and_cost(instance, open_set)

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
        "pmed_gr_ls_main",
        os.path.join(_parent_dir, "heuristics", "greedy_pmedian.py"),
    )
    gr_sol = _gr.greedy_pmedian(inst)
    print(f"Greedy: cost={gr_sol.cost:.1f}")

    ls_sol = local_search(inst, seed=42)
    print(f"LS: cost={ls_sol.cost:.1f}")
