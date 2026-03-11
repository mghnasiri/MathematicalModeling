"""
Constructive Heuristics for TSPTW.

Problem: TSP with Time Windows
Complexity: O(n^2) for nearest feasible, O(n^3) for insertion

1. Nearest Feasible Neighbor: visit the nearest city whose time window
   can still be met.
2. Earliest Deadline Insertion: insert cities by deadline, at the
   cheapest feasible position.

References:
    Gendreau, M., Hertz, A., Laporte, G. & Stan, M. (1998). A
    generalized insertion heuristic for the traveling salesman problem
    with time windows. Operations Research, 46(3), 330-335.
    https://doi.org/10.1287/opre.46.3.330

    Solomon, M.M. (1987). Algorithms for the vehicle routing and
    scheduling problems with time window constraints. Operations
    Research, 35(2), 254-265.
    https://doi.org/10.1287/opre.35.2.254
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


_inst = _load_mod("tsptw_instance_h", os.path.join(_this_dir, "instance.py"))
TSPTWInstance = _inst.TSPTWInstance
TSPTWSolution = _inst.TSPTWSolution


def nearest_feasible(instance: TSPTWInstance) -> TSPTWSolution:
    """Construct a tour using nearest feasible neighbor.

    Starting from city 0, greedily visit the nearest unvisited city
    whose time window can be met. If no feasible city exists, visit
    the nearest unvisited city (may violate windows).

    Args:
        instance: A TSPTWInstance.

    Returns:
        TSPTWSolution with the constructed tour.
    """
    n = instance.n
    dist = instance.distance_matrix
    visited = [False] * n
    tour = [0]
    visited[0] = True
    current_time = instance.time_windows[0][0]

    for _ in range(n - 1):
        current = tour[-1]
        best_city = -1
        best_dist = float("inf")

        # Try feasible cities first
        for j in range(n):
            if visited[j]:
                continue
            arrival = current_time + instance.service_times[current] + dist[current][j]
            if arrival <= instance.time_windows[j][1] + 1e-10:
                if dist[current][j] < best_dist:
                    best_dist = dist[current][j]
                    best_city = j

        if best_city < 0:
            # No feasible city — pick nearest unvisited
            for j in range(n):
                if not visited[j] and dist[current][j] < best_dist:
                    best_dist = dist[current][j]
                    best_city = j

        tour.append(best_city)
        visited[best_city] = True
        arrival = current_time + instance.service_times[current] + dist[current][best_city]
        current_time = max(arrival, instance.time_windows[best_city][0])

    return TSPTWSolution(
        tour=tour,
        distance=instance.tour_distance(tour),
        feasible=instance.tour_feasible(tour),
    )


def earliest_deadline_insertion(instance: TSPTWInstance) -> TSPTWSolution:
    """Construct a tour by inserting cities ordered by deadline.

    Process cities in order of their latest arrival time (deadline).
    For each city, find the cheapest insertion position that maintains
    time window feasibility.

    Args:
        instance: A TSPTWInstance.

    Returns:
        TSPTWSolution.
    """
    n = instance.n

    # Sort non-depot cities by deadline
    cities = list(range(1, n))
    cities.sort(key=lambda i: instance.time_windows[i][1])

    tour = [0]

    for city in cities:
        best_pos = -1
        best_increase = float("inf")

        for pos in range(1, len(tour) + 1):
            trial = tour[:pos] + [city] + tour[pos:]
            if _check_partial_feasibility(instance, trial):
                increase = _insertion_cost(instance, tour, city, pos)
                if increase < best_increase:
                    best_increase = increase
                    best_pos = pos

        if best_pos >= 0:
            tour.insert(best_pos, city)
        else:
            # Insert at end if no feasible position
            tour.append(city)

    return TSPTWSolution(
        tour=tour,
        distance=instance.tour_distance(tour),
        feasible=instance.tour_feasible(tour),
    )


def _check_partial_feasibility(
    instance: TSPTWInstance, tour: list[int]
) -> bool:
    """Check if partial tour satisfies time windows."""
    if not tour:
        return True
    current_time = instance.time_windows[tour[0]][0]
    for k in range(1, len(tour)):
        prev, curr = tour[k - 1], tour[k]
        arrival = current_time + instance.service_times[prev] + instance.distance_matrix[prev][curr]
        if arrival > instance.time_windows[curr][1] + 1e-10:
            return False
        current_time = max(arrival, instance.time_windows[curr][0])
    return True


def _insertion_cost(
    instance: TSPTWInstance,
    tour: list[int],
    city: int,
    pos: int,
) -> float:
    """Compute distance increase for inserting city at position."""
    dist = instance.distance_matrix
    if pos == 0:
        return dist[city][tour[0]] - 0
    elif pos >= len(tour):
        return dist[tour[-1]][city]
    else:
        prev, nxt = tour[pos - 1], tour[pos]
        return dist[prev][city] + dist[city][nxt] - dist[prev][nxt]


if __name__ == "__main__":
    inst = _inst.small_tsptw_5()
    print(f"TSPTW: {inst.n} cities")

    sol1 = nearest_feasible(inst)
    print(f"Nearest feasible: dist={sol1.distance:.1f}, feasible={sol1.feasible}")

    sol2 = earliest_deadline_insertion(inst)
    print(f"EDT insertion: dist={sol2.distance:.1f}, feasible={sol2.feasible}")
