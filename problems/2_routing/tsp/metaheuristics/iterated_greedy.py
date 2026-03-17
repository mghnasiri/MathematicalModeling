"""
Iterated Greedy for the Traveling Salesman Problem (TSP).

Iteratively removes d random cities from the tour and reinserts them
at the cheapest positions. Uses Boltzmann acceptance criterion.

Warm-started with nearest neighbor multistart heuristic.

Complexity: O(iterations * d * n) per run.

References:
    Ruiz, R. & Stützle, T. (2007). A simple and effective iterated
    greedy algorithm for the permutation flowshop scheduling problem.
    European Journal of Operational Research, 177(3), 2033-2049.
    https://doi.org/10.1016/j.ejor.2005.12.009

    Jacobs, L.W. & Brusco, M.J. (1995). Note: A local-search heuristic
    for large set-covering problems. Naval Research Logistics, 42(7),
    1129-1140.
    https://doi.org/10.1002/1520-6750(199510)42:7<1129::AID-NAV3220420711>3.0.CO;2-M
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


_inst = _load_mod("tsp_instance_ig", os.path.join(_parent_dir, "instance.py"))
TSPInstance = _inst.TSPInstance
TSPSolution = _inst.TSPSolution
validate_tour = _inst.validate_tour

_nn = _load_mod(
    "tsp_nn_ig",
    os.path.join(_parent_dir, "heuristics", "nearest_neighbor.py"),
)
nearest_neighbor_multistart = _nn.nearest_neighbor_multistart


def iterated_greedy(
    instance: TSPInstance,
    max_iterations: int = 5000,
    d: int | None = None,
    temperature_factor: float = 0.1,
    time_limit: float | None = None,
    seed: int | None = None,
) -> TSPSolution:
    """Solve TSP using Iterated Greedy.

    Args:
        instance: A TSPInstance.
        max_iterations: Maximum number of iterations.
        d: Number of cities to remove. Default: max(2, n//5).
        temperature_factor: Controls acceptance probability.
        time_limit: Maximum wall-clock time in seconds.
        seed: Random seed for reproducibility.

    Returns:
        TSPSolution with the best tour found.
    """
    rng = np.random.default_rng(seed)
    n = instance.n
    start_time = time.time()

    if d is None:
        d = max(2, n // 5)
    d = min(d, n - 1)

    dist = instance.distance_matrix

    # Warm-start
    init_sol = nearest_neighbor_multistart(instance)
    tour = list(init_sol.tour)
    current_dist = instance.tour_distance(tour)

    best_tour = tour[:]
    best_dist = current_dist

    # Temperature
    avg_edge = current_dist / n if n > 0 else 1.0
    temperature = temperature_factor * avg_edge

    for iteration in range(max_iterations):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        # Destruction: remove d random cities
        indices = sorted(rng.choice(n, size=d, replace=False), reverse=True)
        removed = []
        partial = tour[:]
        for idx in indices:
            removed.append(partial.pop(idx))

        # Reconstruction: greedily insert each removed city at cheapest position
        for city in removed:
            best_pos = 0
            best_cost = float("inf")
            for pos in range(len(partial) + 1):
                # Cost of inserting city at position pos
                if len(partial) == 0:
                    cost = 0
                elif pos == 0:
                    cost = dist[city][partial[0]] + dist[partial[-1]][city] - dist[partial[-1]][partial[0]]
                elif pos == len(partial):
                    cost = dist[partial[-1]][city] + dist[city][partial[0]] - dist[partial[-1]][partial[0]]
                else:
                    prev = partial[pos - 1]
                    nxt = partial[pos]
                    cost = dist[prev][city] + dist[city][nxt] - dist[prev][nxt]

                if cost < best_cost:
                    best_cost = cost
                    best_pos = pos

            partial.insert(best_pos, city)

        new_dist = instance.tour_distance(partial)

        # Acceptance
        delta = new_dist - current_dist
        if delta < 0 or (temperature > 0 and
                         rng.random() < math.exp(-delta / max(temperature, 1e-10))):
            tour = partial
            current_dist = new_dist

            if current_dist < best_dist:
                best_dist = current_dist
                best_tour = tour[:]

    return TSPSolution(tour=best_tour, distance=best_dist)


if __name__ == "__main__":
    inst = TSPInstance.random(n=20, seed=42)
    print(f"TSP: {inst.n} cities")

    nn_sol = nearest_neighbor_multistart(inst)
    print(f"NN multistart: distance={nn_sol.distance:.2f}")

    ig_sol = iterated_greedy(inst, seed=42)
    print(f"IG: distance={ig_sol.distance:.2f}")
