"""
Tabu Search for the p-Median Problem.

Problem: PMP (p-Median Problem)

Neighborhood: Swap — close one open facility and open a closed one,
maintaining exactly p open facilities.

Uses short-term memory preventing recently swapped facilities from
being swapped again. Aspiration criterion overrides tabu when a
move yields a new global best.

Warm-started with greedy p-median heuristic.

Complexity: O(iterations * m * (m-p) * n) per run.

References:
    Mladenović, N., Brimberg, J., Hansen, P. & Moreno-Pérez, J.A.
    (2007). The p-median problem: A survey of metaheuristic
    approaches. European Journal of Operational Research, 179(3),
    927-939.
    https://doi.org/10.1016/j.ejor.2005.05.034

    Resende, M.G.C. & Werneck, R.F. (2004). A hybrid heuristic for
    the p-median problem. Journal of Heuristics, 10(1), 59-88.
    https://doi.org/10.1023/B:HEUR.0000019986.96257.50
"""

from __future__ import annotations

import os
import sys
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


_inst = _load_mod("pmedian_instance_ts", os.path.join(_parent_dir, "instance.py"))
PMedianInstance = _inst.PMedianInstance
PMedianSolution = _inst.PMedianSolution

_greedy = _load_mod(
    "pm_greedy_ts",
    os.path.join(_parent_dir, "heuristics", "greedy_pmedian.py"),
)
greedy_pmedian = _greedy.greedy_pmedian


def _evaluate(
    instance: PMedianInstance,
    open_set: set[int],
) -> tuple[float, list[int]]:
    """Evaluate total weighted distance for an open facility set."""
    assignments = []
    total = 0.0
    for j in range(instance.n):
        best_fac = min(open_set, key=lambda i: instance.distance_matrix[i][j])
        assignments.append(best_fac)
        total += instance.weights[j] * instance.distance_matrix[best_fac][j]
    return total, assignments


def tabu_search(
    instance: PMedianInstance,
    max_iterations: int = 2000,
    tabu_tenure: int | None = None,
    time_limit: float | None = None,
    seed: int | None = None,
) -> PMedianSolution:
    """Solve p-Median using Tabu Search.

    Args:
        instance: p-Median instance.
        max_iterations: Maximum iterations.
        tabu_tenure: Iterations a move stays tabu. Default: sqrt(m).
        time_limit: Time limit in seconds.
        seed: Random seed for reproducibility.

    Returns:
        Best PMedianSolution found.
    """
    rng = np.random.default_rng(seed)
    m, p = instance.m, instance.p
    start_time = time.time()

    if tabu_tenure is None:
        tabu_tenure = max(3, int(m ** 0.5))

    # Initialize with greedy
    init_sol = greedy_pmedian(instance)
    open_set = set(init_sol.open_facilities)
    current_cost, current_assign = _evaluate(instance, open_set)

    best_open = set(open_set)
    best_cost = current_cost
    best_assign = list(current_assign)

    # Tabu list: facility -> iteration when tabu expires
    tabu_dict: dict[int, int] = {}

    for iteration in range(max_iterations):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        best_delta = float("inf")
        best_move = None

        # Swap neighborhood: close one, open another
        closed = [i for i in range(m) if i not in open_set]
        for i_open in list(open_set):
            for i_closed in closed:
                is_tabu = (
                    (i_open in tabu_dict and tabu_dict[i_open] > iteration)
                    or (i_closed in tabu_dict and tabu_dict[i_closed] > iteration)
                )

                test_set = (open_set - {i_open}) | {i_closed}
                new_cost, _ = _evaluate(instance, test_set)
                delta = new_cost - current_cost

                if is_tabu and current_cost + delta >= best_cost:
                    continue

                if delta < best_delta:
                    best_delta = delta
                    best_move = (i_open, i_closed)

        if best_move is None:
            tabu_dict.clear()
            continue

        # Apply move
        i_open, i_closed = best_move
        open_set.remove(i_open)
        open_set.add(i_closed)
        tabu_dict[i_open] = iteration + tabu_tenure
        tabu_dict[i_closed] = iteration + tabu_tenure

        current_cost, current_assign = _evaluate(instance, open_set)

        if current_cost < best_cost:
            best_cost = current_cost
            best_open = set(open_set)
            best_assign = list(current_assign)

    return PMedianSolution(
        open_facilities=sorted(best_open),
        assignments=best_assign,
        cost=best_cost,
    )


if __name__ == "__main__":
    from instance import small_pmedian_6_2

    inst = small_pmedian_6_2()
    sol = tabu_search(inst, seed=42)
    print(f"TS: cost={sol.cost:.1f}, open={sol.open_facilities}")
