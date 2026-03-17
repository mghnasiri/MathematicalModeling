"""
Insertion Heuristics — TSP construction by iteratively inserting cities.

Problem: TSP (Traveling Salesman Problem)
Complexity: O(n^2) per insertion, O(n^3) total for cheapest/farthest insertion

Three variants:
- Cheapest Insertion: Insert the city that increases tour cost the least.
- Farthest Insertion: Select the city farthest from the current tour,
  then insert at the cheapest position. Tends to build the tour shape early.
- Nearest Insertion: Select the city nearest to the current tour,
  then insert at the cheapest position.

Cheapest Insertion has a 2-approximation guarantee for metric TSP.

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

_inst = _load_module("tsp_instance_ci", os.path.join(_parent_dir, "instance.py"))
TSPInstance = _inst.TSPInstance
TSPSolution = _inst.TSPSolution


def _find_cheapest_position(
    tour: list[int], city: int, dist: np.ndarray
) -> tuple[int, float]:
    """Find the position in the tour where inserting city is cheapest.

    Args:
        tour: Current partial tour.
        city: City to insert.
        dist: Distance matrix.

    Returns:
        Tuple of (best_position, insertion_cost).
    """
    best_pos = 0
    best_cost = float("inf")
    m = len(tour)

    for i in range(m):
        j = (i + 1) % m
        cost = dist[tour[i]][city] + dist[city][tour[j]] - dist[tour[i]][tour[j]]
        if cost < best_cost:
            best_cost = cost
            best_pos = i + 1

    return best_pos, best_cost


def cheapest_insertion(instance: TSPInstance) -> TSPSolution:
    """Construct a tour using cheapest insertion.

    At each step, find the (city, position) pair that minimizes the
    increase in tour cost, and insert it.

    Args:
        instance: A TSPInstance.

    Returns:
        TSPSolution with the constructed tour.
    """
    n = instance.n
    dist = instance.distance_matrix

    if n <= 2:
        tour = list(range(n))
        return TSPSolution(tour=tour, distance=instance.tour_distance(tour))

    # Start with the two cities that are farthest apart
    i_max, j_max = 0, 1
    max_dist = dist[0][1]
    for i in range(n):
        for j in range(i + 1, n):
            if dist[i][j] > max_dist:
                max_dist = dist[i][j]
                i_max, j_max = i, j

    tour = [i_max, j_max]
    in_tour = set(tour)

    while len(tour) < n:
        # Find the cheapest (city, position) insertion
        best_city = -1
        best_pos = -1
        best_cost = float("inf")

        for city in range(n):
            if city in in_tour:
                continue
            pos, cost = _find_cheapest_position(tour, city, dist)
            if cost < best_cost:
                best_cost = cost
                best_city = city
                best_pos = pos

        tour.insert(best_pos, best_city)
        in_tour.add(best_city)

    return TSPSolution(tour=tour, distance=instance.tour_distance(tour))


def farthest_insertion(instance: TSPInstance) -> TSPSolution:
    """Construct a tour using farthest insertion.

    At each step, select the city farthest from the current tour,
    then insert it at the cheapest position.

    Args:
        instance: A TSPInstance.

    Returns:
        TSPSolution with the constructed tour.
    """
    n = instance.n
    dist = instance.distance_matrix

    if n <= 2:
        tour = list(range(n))
        return TSPSolution(tour=tour, distance=instance.tour_distance(tour))

    # Start with the two farthest cities
    i_max, j_max = 0, 1
    max_dist = dist[0][1]
    for i in range(n):
        for j in range(i + 1, n):
            if dist[i][j] > max_dist:
                max_dist = dist[i][j]
                i_max, j_max = i, j

    tour = [i_max, j_max]
    in_tour = set(tour)

    while len(tour) < n:
        # Select city farthest from tour
        farthest_city = -1
        farthest_dist = -1.0

        for city in range(n):
            if city in in_tour:
                continue
            min_d = min(dist[city][t] for t in tour)
            if min_d > farthest_dist:
                farthest_dist = min_d
                farthest_city = city

        # Insert at cheapest position
        best_pos, _ = _find_cheapest_position(tour, farthest_city, dist)
        tour.insert(best_pos, farthest_city)
        in_tour.add(farthest_city)

    return TSPSolution(tour=tour, distance=instance.tour_distance(tour))


def nearest_insertion(instance: TSPInstance) -> TSPSolution:
    """Construct a tour using nearest insertion.

    At each step, select the city nearest to the current tour,
    then insert it at the cheapest position.

    Args:
        instance: A TSPInstance.

    Returns:
        TSPSolution with the constructed tour.
    """
    n = instance.n
    dist = instance.distance_matrix

    if n <= 2:
        tour = list(range(n))
        return TSPSolution(tour=tour, distance=instance.tour_distance(tour))

    # Start with the two nearest cities
    i_min, j_min = 0, 1
    min_dist = dist[0][1]
    for i in range(n):
        for j in range(i + 1, n):
            if i != j and dist[i][j] < min_dist:
                min_dist = dist[i][j]
                i_min, j_min = i, j

    tour = [i_min, j_min]
    in_tour = set(tour)

    while len(tour) < n:
        # Select city nearest to tour
        nearest_city = -1
        nearest_dist = float("inf")

        for city in range(n):
            if city in in_tour:
                continue
            min_d = min(dist[city][t] for t in tour)
            if min_d < nearest_dist:
                nearest_dist = min_d
                nearest_city = city

        # Insert at cheapest position
        best_pos, _ = _find_cheapest_position(tour, nearest_city, dist)
        tour.insert(best_pos, nearest_city)
        in_tour.add(nearest_city)

    return TSPSolution(tour=tour, distance=instance.tour_distance(tour))


if __name__ == "__main__":
    from instance import small4, small5, gr17

    print("=== Insertion Heuristics ===\n")

    for name, inst_fn in [("small4", small4), ("small5", small5), ("gr17", gr17)]:
        inst = inst_fn()
        c_sol = cheapest_insertion(inst)
        f_sol = farthest_insertion(inst)
        n_sol = nearest_insertion(inst)
        print(f"{name}: cheapest={c_sol.distance:.1f}, "
              f"farthest={f_sol.distance:.1f}, nearest={n_sol.distance:.1f}")
