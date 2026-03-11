"""
Simulated Annealing for p-Median Problem (PMP).

Problem: p-Median
Neighborhood: Swap — close one open facility, open a closed one.
Constraint: exactly p facilities remain open at all times.

Warm-started with greedy heuristic.

Complexity: O(iterations * m * n) per run.

References:
    Mladenović, N., Brimberg, J., Hansen, P. & Moreno-Pérez, J.A. (2007).
    The p-median problem: A survey of metaheuristic approaches. European
    Journal of Operational Research, 179(3), 927-939.
    https://doi.org/10.1016/j.ejor.2005.05.034

    Kirkpatrick, S., Gelatt, C.D. & Vecchi, M.P. (1983). Optimization
    by simulated annealing. Science, 220(4598), 671-680.
    https://doi.org/10.1126/science.220.4598.671
"""

from __future__ import annotations

import os
import sys
import math
import time
import importlib.util

import numpy as np

_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_mod(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_inst = _load_mod("pmed_instance_sa", os.path.join(_parent_dir, "instance.py"))
PMedianInstance = _inst.PMedianInstance
PMedianSolution = _inst.PMedianSolution


def _assign_and_cost(
    instance: PMedianInstance, open_set: set[int]
) -> tuple[list[int], float]:
    """Assign each customer to nearest open facility and compute cost."""
    open_list = sorted(open_set)
    assignments = []
    total = 0.0
    for j in range(instance.n):
        best_fac = min(open_list, key=lambda i: instance.distance_matrix[i][j])
        assignments.append(best_fac)
        total += instance.weights[j] * instance.distance_matrix[best_fac][j]
    return assignments, total


def simulated_annealing(
    instance: PMedianInstance,
    max_iterations: int = 5000,
    initial_temp: float | None = None,
    cooling_rate: float = 0.995,
    time_limit: float | None = None,
    seed: int | None = None,
) -> PMedianSolution:
    """Solve p-Median using Simulated Annealing.

    Args:
        instance: p-Median instance.
        max_iterations: Maximum iterations.
        initial_temp: Initial temperature. If None, auto-calibrated.
        cooling_rate: Geometric cooling factor.
        time_limit: Time limit in seconds.
        seed: Random seed for reproducibility.

    Returns:
        Best PMedianSolution found.
    """
    rng = np.random.default_rng(seed)
    start_time = time.time()
    p = instance.p
    m = instance.m

    # ── Initialize with greedy ───────────────────────────────────────────
    _greedy_mod = _load_mod(
        "pmed_greedy_sa",
        os.path.join(_parent_dir, "heuristics", "greedy_pmedian.py"),
    )
    init_sol = _greedy_mod.greedy_pmedian(instance)

    open_set = set(init_sol.open_facilities)
    closed_set = set(range(m)) - open_set
    assignments, current_cost = _assign_and_cost(instance, open_set)

    best_open = set(open_set)
    best_assignments = list(assignments)
    best_cost = current_cost

    if initial_temp is None:
        initial_temp = max(1.0, current_cost * 0.05)
    temp = initial_temp

    for iteration in range(max_iterations):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        if not closed_set:
            break

        # Swap: close one open facility, open a closed one
        open_list = sorted(open_set)
        closed_list = sorted(closed_set)

        fac_close = open_list[rng.integers(0, len(open_list))]
        fac_open = closed_list[rng.integers(0, len(closed_list))]

        # Apply swap
        new_open = (open_set - {fac_close}) | {fac_open}
        new_assignments, new_cost = _assign_and_cost(instance, new_open)

        delta = new_cost - current_cost

        if delta <= 0 or rng.random() < math.exp(-delta / max(temp, 1e-10)):
            open_set = new_open
            closed_set = (closed_set - {fac_open}) | {fac_close}
            assignments = new_assignments
            current_cost = new_cost

            if current_cost < best_cost:
                best_open = set(open_set)
                best_assignments = list(assignments)
                best_cost = current_cost

        temp *= cooling_rate

    return PMedianSolution(
        open_facilities=sorted(best_open),
        assignments=best_assignments,
        cost=best_cost,
    )


if __name__ == "__main__":
    inst = PMedianInstance.random(n=20, m=20, p=3, seed=42)
    sol = simulated_annealing(inst, seed=42)
    print(f"SA: cost={sol.cost:.2f}, open={sol.open_facilities}")
