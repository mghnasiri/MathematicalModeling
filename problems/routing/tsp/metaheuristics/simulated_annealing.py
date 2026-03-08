"""
Simulated Annealing for TSP.

Problem: TSP (Traveling Salesman Problem)
Neighborhood: 2-opt moves (segment reversal)

Uses geometric cooling schedule with Boltzmann acceptance criterion.
Initial temperature calibrated to accept ~80% of uphill moves.
Warm-started with nearest neighbor heuristic.

References:
    Kirkpatrick, S., Gelatt, C.D. & Vecchi, M.P. (1983). Optimization
    by simulated annealing. Science, 220(4598), 671-680.
    https://doi.org/10.1126/science.220.4598.671

    Černý, V. (1985). Thermodynamical approach to the traveling
    salesman problem: An efficient simulation algorithm. Journal of
    Optimization Theory and Applications, 45(1), 41-51.
    https://doi.org/10.1007/BF00940812
"""

from __future__ import annotations

import os
import math
import numpy as np

_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _load_module(name, filepath):
    import importlib.util
    import sys as _sys
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    _sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_inst = _load_module("tsp_instance_sa", os.path.join(_parent_dir, "instance.py"))
TSPInstance = _inst.TSPInstance
TSPSolution = _inst.TSPSolution


def simulated_annealing(
    instance: TSPInstance,
    initial_tour: list[int] | None = None,
    max_iterations: int = 100_000,
    initial_temp: float | None = None,
    cooling_rate: float = 0.9995,
    seed: int | None = None,
) -> TSPSolution:
    """Solve TSP using simulated annealing with 2-opt neighborhood.

    Args:
        instance: A TSPInstance.
        initial_tour: Starting tour. If None, uses nearest neighbor.
        max_iterations: Maximum number of iterations.
        initial_temp: Initial temperature. If None, auto-calibrated.
        cooling_rate: Geometric cooling factor (0 < alpha < 1).
        seed: Random seed for reproducibility.

    Returns:
        TSPSolution with the best tour found.
    """
    rng = np.random.default_rng(seed)
    n = instance.n
    dist = instance.distance_matrix

    if n <= 3:
        tour = list(range(n))
        return TSPSolution(tour=tour, distance=instance.tour_distance(tour))

    # Initialize with nearest neighbor if no tour given
    if initial_tour is None:
        _nn_mod = _load_module(
            "tsp_nn_sa", os.path.join(_parent_dir, "heuristics", "nearest_neighbor.py"))
        nearest_neighbor = _nn_mod.nearest_neighbor
        sol = nearest_neighbor(instance)
        tour = sol.tour[:]
    else:
        tour = initial_tour[:]

    current_cost = instance.tour_distance(tour)
    best_tour = tour[:]
    best_cost = current_cost

    # Auto-calibrate temperature: accept ~80% of random 2-opt moves
    if initial_temp is None:
        deltas = []
        for _ in range(min(1000, n * n)):
            i = rng.integers(0, n - 1)
            j = rng.integers(i + 1, n)
            if i == 0 and j == n - 1:
                continue
            a, b = tour[i], tour[(i + 1) % n]
            c, d = tour[j], tour[(j + 1) % n]
            delta = (dist[a][c] + dist[b][d]) - (dist[a][b] + dist[c][d])
            if delta > 0:
                deltas.append(delta)
        if deltas:
            avg_delta = np.mean(deltas)
            initial_temp = -avg_delta / math.log(0.8)
        else:
            initial_temp = 1.0

    temp = initial_temp

    for iteration in range(max_iterations):
        # Generate random 2-opt move
        i = rng.integers(0, n - 1)
        j = rng.integers(i + 1, n)
        if i == 0 and j == n - 1:
            continue

        a, b = tour[i], tour[(i + 1) % n]
        c, d = tour[j], tour[(j + 1) % n]
        delta = (dist[a][c] + dist[b][d]) - (dist[a][b] + dist[c][d])

        # Acceptance criterion
        if delta < 0 or (temp > 0 and rng.random() < math.exp(-delta / temp)):
            tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
            current_cost += delta

            if current_cost < best_cost:
                best_cost = current_cost
                best_tour = tour[:]

        temp *= cooling_rate

    return TSPSolution(tour=best_tour, distance=instance.tour_distance(best_tour))


if __name__ == "__main__":
    from instance import small4, small5, gr17

    print("=== Simulated Annealing ===\n")

    for name, inst_fn in [("small4", small4), ("small5", small5), ("gr17", gr17)]:
        inst = inst_fn()
        sol = simulated_annealing(inst, seed=42)
        print(f"{name}: distance={sol.distance:.1f}")
