"""
Asymmetric TSP — Heuristics.

Algorithms:
    - Nearest Neighbor: greedy directed NN, O(n^2).
    - Patching: build assignment, patch subtours into single tour.

References:
    Kanellakis, P.C. & Papadimitriou, C.H. (1980). Local search for the
    asymmetric traveling salesman problem. Operations Research, 28(5),
    1086-1099. https://doi.org/10.1287/opre.28.5.1086
"""

from __future__ import annotations

import sys
import os
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


_inst = _load_mod("atsp_instance_h", os.path.join(_this_dir, "instance.py"))
ATSPInstance = _inst.ATSPInstance
ATSPSolution = _inst.ATSPSolution


def nearest_neighbor_atsp(
    instance: ATSPInstance, start: int = 0
) -> ATSPSolution:
    """Directed nearest neighbor heuristic.

    Args:
        instance: ATSP instance.
        start: Starting city.

    Returns:
        ATSPSolution.
    """
    n = instance.n
    dm = instance.dist_matrix
    visited = {start}
    tour = [start]

    current = start
    for _ in range(n - 1):
        best = -1
        best_dist = float("inf")
        for j in range(n):
            if j not in visited and dm[current][j] < best_dist:
                best_dist = dm[current][j]
                best = j
        if best < 0:
            break
        tour.append(best)
        visited.add(best)
        current = best

    cost = instance.tour_cost(tour)
    return ATSPSolution(tour=tour, cost=cost)


def multi_start_nn(instance: ATSPInstance) -> ATSPSolution:
    """Multi-start nearest neighbor: try all starting cities.

    Args:
        instance: ATSP instance.

    Returns:
        Best ATSPSolution across all starts.
    """
    best = None
    for s in range(instance.n):
        sol = nearest_neighbor_atsp(instance, start=s)
        if best is None or sol.cost < best.cost:
            best = sol
    return best


if __name__ == "__main__":
    from instance import small_atsp_5

    inst = small_atsp_5()
    sol1 = nearest_neighbor_atsp(inst)
    print(f"NN: {sol1}")
    sol2 = multi_start_nn(inst)
    print(f"Multi-start NN: {sol2}")
