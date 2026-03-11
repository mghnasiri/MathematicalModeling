"""
Simulated Annealing for TSPTW.

Problem: TSP with Time Windows

Uses Or-opt moves (relocate a city segment) with time window feasibility
checks. Accepts infeasible solutions with a weighted penalty to allow
the search to cross infeasible regions.

Warm-started with nearest feasible neighbor.

Complexity: O(iterations * n) per run.

References:
    Ohlmann, J.W. & Thomas, B.W. (2007). A compressed-annealing
    heuristic for the traveling salesman problem with time windows.
    INFORMS Journal on Computing, 19(1), 80-90.
    https://doi.org/10.1287/ijoc.1050.0145

    Gendreau, M., Hertz, A., Laporte, G. & Stan, M. (1998). A
    generalized insertion heuristic for the traveling salesman problem
    with time windows. Operations Research, 46(3), 330-335.
    https://doi.org/10.1287/opre.46.3.330
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


_inst = _load_mod("tsptw_instance_meta", os.path.join(_this_dir, "instance.py"))
TSPTWInstance = _inst.TSPTWInstance
TSPTWSolution = _inst.TSPTWSolution

_heur = _load_mod("tsptw_heuristics", os.path.join(_this_dir, "heuristics.py"))
nearest_feasible = _heur.nearest_feasible


def _tw_penalty(instance: TSPTWInstance, tour: list[int]) -> float:
    """Compute total time window violation."""
    penalty = 0.0
    times = instance.tour_schedule(tour)
    for k, city in enumerate(tour):
        if times[k] > instance.time_windows[city][1] + 1e-10:
            penalty += times[k] - instance.time_windows[city][1]
    return penalty


def simulated_annealing(
    instance: TSPTWInstance,
    max_iterations: int = 50000,
    initial_temp: float | None = None,
    cooling_rate: float = 0.9995,
    penalty_weight: float = 100.0,
    time_limit: float | None = None,
    seed: int | None = None,
) -> TSPTWSolution:
    """Solve TSPTW using Simulated Annealing.

    Args:
        instance: A TSPTWInstance.
        max_iterations: Maximum iterations.
        initial_temp: Initial temperature. Auto-calibrated if None.
        cooling_rate: Geometric cooling factor.
        penalty_weight: Weight for time window violations.
        time_limit: Time limit in seconds.
        seed: Random seed.

    Returns:
        TSPTWSolution with the best feasible tour found.
    """
    rng = np.random.default_rng(seed)
    n = instance.n
    start_time = time.time()

    if n <= 2:
        tour = list(range(n))
        return TSPTWSolution(
            tour=tour,
            distance=instance.tour_distance(tour),
            feasible=instance.tour_feasible(tour),
        )

    # Warm-start
    init_sol = nearest_feasible(instance)
    tour = init_sol.tour[:]
    current_dist = init_sol.distance
    current_penalty = _tw_penalty(instance, tour)
    current_cost = current_dist + penalty_weight * current_penalty

    best_tour = tour[:]
    best_dist = current_dist
    best_feasible = current_penalty < 1e-10

    if initial_temp is None:
        initial_temp = current_dist * 0.1

    temp = initial_temp

    for iteration in range(max_iterations):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        move = rng.integers(0, 3)
        new_tour = tour[:]

        if move == 0:
            # Or-opt: relocate a single city
            i = rng.integers(1, n) if n > 2 else 0
            city = new_tour.pop(i)
            j = rng.integers(0, len(new_tour))
            new_tour.insert(j, city)

        elif move == 1 and n > 3:
            # 2-opt: reverse a segment
            i = rng.integers(0, n - 2)
            j_high = min(i + n // 2 + 1, n)
            if i + 2 < j_high:
                j = rng.integers(i + 2, j_high)
                new_tour[i + 1:j + 1] = new_tour[i + 1:j + 1][::-1]
            else:
                i2, j2 = rng.choice(n, 2, replace=False)
                new_tour[i2], new_tour[j2] = new_tour[j2], new_tour[i2]

        else:
            # Swap two cities
            i, j = rng.choice(n, 2, replace=False)
            new_tour[i], new_tour[j] = new_tour[j], new_tour[i]

        new_dist = instance.tour_distance(new_tour)
        new_penalty = _tw_penalty(instance, new_tour)
        new_cost = new_dist + penalty_weight * new_penalty

        delta = new_cost - current_cost
        if delta < 0 or (temp > 0 and rng.random() < math.exp(-delta / max(temp, 1e-10))):
            tour = new_tour
            current_dist = new_dist
            current_penalty = new_penalty
            current_cost = new_cost

            if new_penalty < 1e-10:
                if not best_feasible or new_dist < best_dist - 1e-10:
                    best_dist = new_dist
                    best_tour = tour[:]
                    best_feasible = True
            elif not best_feasible and new_dist < best_dist - 1e-10:
                best_dist = new_dist
                best_tour = tour[:]

        temp *= cooling_rate

    return TSPTWSolution(
        tour=best_tour,
        distance=instance.tour_distance(best_tour),
        feasible=instance.tour_feasible(best_tour),
    )


if __name__ == "__main__":
    inst = TSPTWInstance.random(n=10, seed=42)
    print(f"TSPTW: {inst.n} cities")

    nf_sol = nearest_feasible(inst)
    print(f"Nearest feasible: dist={nf_sol.distance:.1f}, feasible={nf_sol.feasible}")

    sa_sol = simulated_annealing(inst, seed=42)
    print(f"SA: dist={sa_sol.distance:.1f}, feasible={sa_sol.feasible}")
