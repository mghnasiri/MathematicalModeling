"""
Asymmetric TSP — Metaheuristics.

Algorithms:
    - Simulated Annealing with or-opt and swap neighborhoods.

References:
    Kanellakis, P.C. & Papadimitriou, C.H. (1980). Local search for the
    asymmetric traveling salesman problem. Operations Research, 28(5),
    1086-1099. https://doi.org/10.1287/opre.28.5.1086
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


_inst = _load_mod("atsp_instance_m", os.path.join(_this_dir, "instance.py"))
ATSPInstance = _inst.ATSPInstance
ATSPSolution = _inst.ATSPSolution

_heur = _load_mod("atsp_heuristics_m", os.path.join(_this_dir, "heuristics.py"))
multi_start_nn = _heur.multi_start_nn


def simulated_annealing(
    instance: ATSPInstance,
    max_iterations: int = 50000,
    cooling_rate: float = 0.9995,
    seed: int | None = None,
    time_limit: float | None = None,
) -> ATSPSolution:
    """SA for ATSP with or-opt (insertion) and swap moves.

    Note: 2-opt reversal is not used since reversing a segment changes
    arc directions, which can be very costly in asymmetric instances.

    Args:
        instance: ATSP instance.
        max_iterations: Maximum iterations.
        cooling_rate: Geometric cooling factor.
        seed: Random seed.
        time_limit: Wall-clock time limit in seconds.

    Returns:
        Best ATSPSolution found.
    """
    rng = np.random.default_rng(seed)
    n = instance.n

    init = multi_start_nn(instance)
    tour = list(init.tour)
    cost = init.cost

    best_tour = list(tour)
    best_cost = cost

    temp = best_cost * 0.05
    start = time.time()

    for it in range(max_iterations):
        if time_limit and time.time() - start > time_limit:
            break

        new_tour = list(tour)
        move = rng.integers(0, 2)

        if move == 0:
            # Or-opt: relocate a city
            i = rng.integers(0, n)
            j = rng.integers(0, n - 1)
            city = new_tour.pop(i)
            new_tour.insert(j, city)
        else:
            # Swap two cities
            i, j = rng.choice(n, 2, replace=False)
            new_tour[i], new_tour[j] = new_tour[j], new_tour[i]

        new_cost = instance.tour_cost(new_tour)
        delta = new_cost - cost

        if delta < 0 or rng.random() < math.exp(-delta / max(temp, 1e-15)):
            tour = new_tour
            cost = new_cost
            if cost < best_cost - 1e-10:
                best_cost = cost
                best_tour = list(tour)

        temp *= cooling_rate

    return ATSPSolution(tour=best_tour, cost=best_cost)


if __name__ == "__main__":
    from instance import small_atsp_5

    inst = small_atsp_5()
    sol = simulated_annealing(inst, max_iterations=10000, seed=42)
    print(f"SA: {sol}")
