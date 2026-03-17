"""
Nearest Neighbor Heuristic — Greedy TSP construction.

Problem: TSP (Traveling Salesman Problem)
Complexity: O(n^2)

Starting from a given city, repeatedly visit the nearest unvisited city.
Simple and fast, but quality depends heavily on the starting city.
Multi-start variant tries all starting cities and returns the best tour.

Approximation ratio: O(log n) in the worst case.

References:
    Rosenkrantz, D.J., Stearns, R.E. & Lewis, P.M. (1977).
    An analysis of several heuristics for the traveling salesman problem.
    SIAM Journal on Computing, 6(3), 563-581.
    https://doi.org/10.1137/0206041
"""

from __future__ import annotations

import os
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

_inst = _load_module("tsp_instance_nn", os.path.join(_parent_dir, "instance.py"))
TSPInstance = _inst.TSPInstance
TSPSolution = _inst.TSPSolution


def nearest_neighbor(
    instance: TSPInstance, start: int = 0
) -> TSPSolution:
    """Construct a tour using the nearest neighbor heuristic.

    Args:
        instance: A TSPInstance.
        start: Starting city index.

    Returns:
        TSPSolution with the constructed tour.
    """
    n = instance.n
    dist = instance.distance_matrix
    visited = [False] * n
    tour = [start]
    visited[start] = True

    for _ in range(n - 1):
        current = tour[-1]
        best_next = -1
        best_dist = float("inf")
        for j in range(n):
            if not visited[j] and dist[current][j] < best_dist:
                best_dist = dist[current][j]
                best_next = j
        tour.append(best_next)
        visited[best_next] = True

    return TSPSolution(tour=tour, distance=instance.tour_distance(tour))


def nearest_neighbor_multistart(instance: TSPInstance) -> TSPSolution:
    """Run nearest neighbor from every starting city and return the best tour.

    Args:
        instance: A TSPInstance.

    Returns:
        Best TSPSolution across all starting cities.
    """
    best_sol = None
    for start in range(instance.n):
        sol = nearest_neighbor(instance, start=start)
        if best_sol is None or sol.distance < best_sol.distance:
            best_sol = sol
    return best_sol


if __name__ == "__main__":
    from instance import small4, small5, gr17

    print("=== Nearest Neighbor ===\n")

    for name, inst_fn in [("small4", small4), ("small5", small5), ("gr17", gr17)]:
        inst = inst_fn()
        sol_single = nearest_neighbor(inst)
        sol_multi = nearest_neighbor_multistart(inst)
        print(f"{name}: single-start={sol_single.distance:.1f}, "
              f"multi-start={sol_multi.distance:.1f}")
