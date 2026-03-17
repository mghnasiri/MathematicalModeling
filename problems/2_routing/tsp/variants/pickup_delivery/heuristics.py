"""
Constructive Heuristics for Pickup and Delivery Problem (PDP).

Problem: 1-PDTSP
Complexity: O(n^2) for nearest feasible, O(n^3) for cheapest insertion

1. Nearest Feasible: greedily visit nearest location that maintains
   precedence (pickup before delivery).
2. Cheapest Pair Insertion: insert pickup-delivery pairs at cheapest
   feasible positions.

References:
    Savelsbergh, M.W.P. & Sol, M. (1995). The general pickup and delivery
    problem. Transportation Science, 29(1), 17-29.
    https://doi.org/10.1287/trsc.29.1.17
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


_inst = _load_mod("pdp_instance_h", os.path.join(_this_dir, "instance.py"))
PDPInstance = _inst.PDPInstance
PDPSolution = _inst.PDPSolution


def nearest_feasible(instance: PDPInstance) -> PDPSolution:
    """Construct a tour using nearest feasible neighbor.

    Starting from depot, greedily visit nearest unvisited location
    that maintains pickup-before-delivery precedence.

    Args:
        instance: A PDPInstance.

    Returns:
        PDPSolution.
    """
    n = instance.num_locations
    np_pairs = instance.num_pairs
    dist = instance.distance_matrix

    visited = [False] * n
    tour = [0]
    visited[0] = True
    picked_up = [False] * (np_pairs + 1)  # 1-indexed

    for _ in range(n - 1):
        current = tour[-1]
        best_loc = -1
        best_dist = float("inf")

        for loc in range(n):
            if visited[loc]:
                continue

            # Check precedence: if loc is a delivery, its pickup must be done
            if loc > np_pairs:
                pair = loc - np_pairs
                if not picked_up[pair]:
                    continue

            if dist[current][loc] < best_dist:
                best_dist = dist[current][loc]
                best_loc = loc

        tour.append(best_loc)
        visited[best_loc] = True
        if 1 <= best_loc <= np_pairs:
            picked_up[best_loc] = True

    return PDPSolution(
        tour=tour,
        distance=instance.tour_distance(tour),
        feasible=instance.precedence_feasible(tour),
    )


def cheapest_pair_insertion(instance: PDPInstance) -> PDPSolution:
    """Insert pickup-delivery pairs at cheapest feasible positions.

    Process pairs in order, for each pair find the cheapest positions
    to insert pickup then delivery (pickup must come before delivery).

    Args:
        instance: A PDPInstance.

    Returns:
        PDPSolution.
    """
    np_pairs = instance.num_pairs
    dist = instance.distance_matrix
    tour = [0]

    # Sort pairs by distance from depot (closest first)
    pair_order = sorted(
        range(1, np_pairs + 1),
        key=lambda p: dist[0][p] + dist[0][p + np_pairs],
    )

    for pair in pair_order:
        pickup = pair
        delivery = pair + np_pairs

        best_cost = float("inf")
        best_p_pos = -1
        best_d_pos = -1

        for p_pos in range(1, len(tour) + 1):
            for d_pos in range(p_pos + 1, len(tour) + 2):
                trial = tour[:p_pos] + [pickup] + tour[p_pos:]
                trial = trial[:d_pos] + [delivery] + trial[d_pos:]
                cost = _tour_distance_partial(dist, trial)
                if cost < best_cost:
                    best_cost = cost
                    best_p_pos = p_pos
                    best_d_pos = d_pos

        tour = tour[:best_p_pos] + [pickup] + tour[best_p_pos:]
        tour = tour[:best_d_pos] + [delivery] + tour[best_d_pos:]

    return PDPSolution(
        tour=tour,
        distance=instance.tour_distance(tour),
        feasible=instance.precedence_feasible(tour),
    )


def _tour_distance_partial(dist: np.ndarray, tour: list[int]) -> float:
    """Compute cyclic tour distance."""
    total = 0.0
    for i in range(len(tour)):
        total += dist[tour[i]][tour[(i + 1) % len(tour)]]
    return total


if __name__ == "__main__":
    inst = _inst.small_pdp_3()
    print(f"PDP: {inst.num_pairs} pairs")

    sol1 = nearest_feasible(inst)
    print(f"Nearest feasible: dist={sol1.distance:.1f}, feasible={sol1.feasible}")

    sol2 = cheapest_pair_insertion(inst)
    print(f"Cheapest insertion: dist={sol2.distance:.1f}, feasible={sol2.feasible}")
