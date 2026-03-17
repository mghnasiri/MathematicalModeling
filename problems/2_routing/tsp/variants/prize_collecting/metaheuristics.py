"""
Prize-Collecting TSP — Metaheuristics.

Algorithms:
    - Simulated Annealing with add/remove/swap/2-opt moves.

References:
    Balas, E. (1989). The prize collecting traveling salesman problem.
    Networks, 19(6), 621-636.
    https://doi.org/10.1002/net.3230190602
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


_inst = _load_mod("pctsp_instance_m", os.path.join(_this_dir, "instance.py"))
PCTSPInstance = _inst.PCTSPInstance
PCTSPSolution = _inst.PCTSPSolution

_heur = _load_mod("pctsp_heuristics_m", os.path.join(_this_dir, "heuristics.py"))
greedy_prize = _heur.greedy_prize


def simulated_annealing(
    instance: PCTSPInstance,
    max_iterations: int = 50000,
    cooling_rate: float = 0.9995,
    seed: int | None = None,
    time_limit: float | None = None,
) -> PCTSPSolution:
    """Simulated Annealing for PCTSP.

    Args:
        instance: PCTSP instance.
        max_iterations: Maximum iterations.
        cooling_rate: Geometric cooling factor.
        seed: Random seed.
        time_limit: Wall-clock time limit in seconds.

    Returns:
        Best PCTSPSolution found.
    """
    rng = np.random.default_rng(seed)
    n = instance.n
    penalty = 1000.0  # penalty for violating min_prize

    init = greedy_prize(instance)
    tour = list(init.tour)
    obj = init.objective + (penalty * max(0, instance.min_prize - init.total_prize))

    best_tour = list(tour)
    best_obj = obj
    best_travel = init.travel_cost
    best_prize = init.total_prize

    temp = abs(obj) * 0.1 if abs(obj) > 1 else 10.0
    start = time.time()

    for it in range(max_iterations):
        if time_limit and time.time() - start > time_limit:
            break

        new_tour = list(tour)
        move = rng.integers(0, 4)

        visited = set(new_tour)

        if move == 0 and len(visited) < n:
            # Add a city
            candidates = [c for c in range(n) if c not in visited]
            if candidates:
                c = rng.choice(candidates)
                pos = rng.integers(0, len(new_tour) + 1)
                new_tour.insert(pos, c)

        elif move == 1 and len(new_tour) > 2:
            # Remove a city
            idx = rng.integers(0, len(new_tour))
            new_tour.pop(idx)

        elif move == 2 and len(new_tour) >= 2:
            # Swap two cities in tour
            i, j = rng.choice(len(new_tour), 2, replace=False)
            new_tour[i], new_tour[j] = new_tour[j], new_tour[i]

        elif move == 3 and len(new_tour) >= 4:
            # 2-opt
            i = rng.integers(0, len(new_tour) - 2)
            j = rng.integers(i + 2, len(new_tour))
            new_tour[i + 1:j + 1] = new_tour[i + 1:j + 1][::-1]

        if not new_tour:
            continue

        new_travel = instance.tour_cost(new_tour)
        new_prize = instance.tour_prize(new_tour)
        new_obj = new_travel - new_prize
        new_obj += penalty * max(0, instance.min_prize - new_prize)

        delta = new_obj - obj
        if delta < 0 or rng.random() < math.exp(-delta / max(temp, 1e-15)):
            tour = new_tour
            obj = new_obj

            actual_obj = new_travel - new_prize
            if (new_prize >= instance.min_prize - 1e-4 and
                    actual_obj < best_obj - 1e-10):
                best_obj = actual_obj
                best_tour = list(tour)
                best_travel = new_travel
                best_prize = new_prize

        temp *= cooling_rate

    # If no feasible solution found, return best anyway
    if best_prize < instance.min_prize - 1e-4:
        best_travel = instance.tour_cost(tour)
        best_prize = instance.tour_prize(tour)
        best_obj = best_travel - best_prize
        best_tour = list(tour)

    return PCTSPSolution(
        tour=best_tour, travel_cost=best_travel,
        total_prize=best_prize, objective=best_obj
    )


if __name__ == "__main__":
    from instance import small_pctsp_6

    inst = small_pctsp_6()
    sol = simulated_annealing(inst, max_iterations=10000, seed=42)
    print(f"SA: {sol}")
