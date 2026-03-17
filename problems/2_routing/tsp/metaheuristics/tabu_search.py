"""
Tabu Search for TSP.

Problem: TSP (Traveling Salesman Problem)
Neighborhood: 2-opt moves (segment reversal)

Uses a tabu list that forbids recently reversed segments from being
reversed again for a number of iterations (tabu tenure). An aspiration
criterion overrides tabu status when a move improves the global best.

Warm-started with nearest neighbor heuristic.

Complexity: O(iterations * n^2) per run (evaluating all 2-opt neighbors).

References:
    Knox, J. (1994). Tabu search performance on the symmetric traveling
    salesman problem. Computers & Operations Research, 21(8), 867-876.
    https://doi.org/10.1016/0305-0548(94)90021-3

    Glover, F. (1989). Tabu search — Part I. ORSA Journal on Computing,
    1(3), 190-206.
    https://doi.org/10.1287/ijoc.1.3.190
"""

from __future__ import annotations

import os
import time
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


_inst = _load_module("tsp_instance_ts", os.path.join(_parent_dir, "instance.py"))
TSPInstance = _inst.TSPInstance
TSPSolution = _inst.TSPSolution


def tabu_search(
    instance: TSPInstance,
    initial_tour: list[int] | None = None,
    max_iterations: int = 5000,
    tabu_tenure: int | None = None,
    time_limit: float | None = None,
    seed: int | None = None,
) -> TSPSolution:
    """Solve TSP using Tabu Search with 2-opt neighborhood.

    Args:
        instance: A TSPInstance.
        initial_tour: Starting tour. If None, uses nearest neighbor.
        max_iterations: Maximum number of iterations.
        tabu_tenure: Number of iterations a move stays tabu. Default: sqrt(n).
        time_limit: Time limit in seconds.
        seed: Random seed for reproducibility (tie-breaking).

    Returns:
        TSPSolution with the best tour found.
    """
    rng = np.random.default_rng(seed)
    n = instance.n
    dist = instance.distance_matrix
    start_time = time.time()

    if n <= 3:
        tour = list(range(n))
        return TSPSolution(tour=tour, distance=instance.tour_distance(tour))

    if tabu_tenure is None:
        tabu_tenure = max(5, int(n ** 0.5))

    # ── Initial solution ─────────────────────────────────────────────────
    if initial_tour is None:
        _nn_mod = _load_module(
            "tsp_nn_ts",
            os.path.join(_parent_dir, "heuristics", "nearest_neighbor.py"),
        )
        nearest_neighbor = _nn_mod.nearest_neighbor
        sol = nearest_neighbor(instance)
        tour = sol.tour[:]
    else:
        tour = initial_tour[:]

    current_dist = instance.tour_distance(tour)
    best_tour = tour[:]
    best_dist = current_dist

    # Tabu list: (i, j) -> iteration when tabu expires
    # We store the edge pair that was broken, using min/max for symmetry
    tabu_dict: dict[tuple[int, int], int] = {}

    for iteration in range(max_iterations):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        best_move_delta = float("inf")
        best_i = -1
        best_j = -1

        # Evaluate all 2-opt moves
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue  # Skip trivial full reversal

                # Delta = new edges - old edges
                # Remove: (tour[i], tour[i+1]) and (tour[j], tour[(j+1)%n])
                # Add: (tour[i], tour[j]) and (tour[i+1], tour[(j+1)%n])
                a, b = tour[i], tour[i + 1]
                c, d = tour[j], tour[(j + 1) % n]

                delta = (
                    dist[a][c] + dist[b][d]
                    - dist[a][b] - dist[c][d]
                )

                # Check tabu status
                edge_key = (min(a, c), max(a, c))
                is_tabu = (
                    edge_key in tabu_dict
                    and tabu_dict[edge_key] > iteration
                )

                # Aspiration: accept if improves global best
                if is_tabu and current_dist + delta >= best_dist:
                    continue

                if delta < best_move_delta:
                    best_move_delta = delta
                    best_i = i
                    best_j = j

        if best_i == -1:
            # All moves are tabu; clear and retry
            tabu_dict.clear()
            continue

        # Apply 2-opt: reverse segment [i+1..j]
        # Record edges being broken as tabu
        a, b = tour[best_i], tour[best_i + 1]
        c, d = tour[best_j], tour[(best_j + 1) % n]
        tabu_dict[(min(a, b), max(a, b))] = iteration + tabu_tenure
        tabu_dict[(min(c, d), max(c, d))] = iteration + tabu_tenure

        tour[best_i + 1: best_j + 1] = tour[best_i + 1: best_j + 1][::-1]
        current_dist += best_move_delta

        if current_dist < best_dist:
            best_dist = current_dist
            best_tour = tour[:]

    return TSPSolution(tour=best_tour, distance=best_dist)


if __name__ == "__main__":
    from instance import small5, gr17

    print("=== Tabu Search on small5 (optimal=19) ===")
    inst = small5()
    sol = tabu_search(inst, max_iterations=500, seed=42)
    print(f"TS distance: {sol.distance:.2f}, tour: {sol.tour}")

    print("\n=== Tabu Search on gr17 (optimal=2016) ===")
    inst = gr17()
    sol = tabu_search(inst, max_iterations=5000, seed=42)
    print(f"TS distance: {sol.distance:.2f}")
