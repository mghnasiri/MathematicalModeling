"""
Prize-Collecting TSP — Constructive Heuristics.

Algorithms:
    - Greedy insertion: add highest prize-to-distance-ratio cities.
    - Nearest neighbor PCTSP: NN selecting profitable cities.

References:
    Balas, E. (1989). The prize collecting traveling salesman problem.
    Networks, 19(6), 621-636.
    https://doi.org/10.1002/net.3230190602
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


_inst = _load_mod("pctsp_instance_h", os.path.join(_this_dir, "instance.py"))
PCTSPInstance = _inst.PCTSPInstance
PCTSPSolution = _inst.PCTSPSolution


def greedy_prize(instance: PCTSPInstance) -> PCTSPSolution:
    """Greedy: add cities by prize/insertion-cost ratio until min_prize met.

    Starts from the highest-prize city, then iteratively inserts the city
    with the best ratio of prize to insertion cost.

    Args:
        instance: PCTSP instance.

    Returns:
        PCTSPSolution.
    """
    n = instance.n
    dm = instance.distance_matrix()

    # Start with highest-prize city
    start = int(np.argmax(instance.prizes))
    tour = [start]
    visited = {start}
    prize_total = float(instance.prizes[start])

    while len(tour) < n:
        best_city = -1
        best_ratio = -float("inf")
        best_pos = 0

        for c in range(n):
            if c in visited:
                continue
            # Find best insertion position
            best_insert_cost = float("inf")
            best_insert_pos = 0
            for pos in range(len(tour)):
                nxt = (pos + 1) % len(tour)
                insert_cost = (dm[tour[pos]][c] + dm[c][tour[nxt]]
                               - dm[tour[pos]][tour[nxt]])
                if insert_cost < best_insert_cost:
                    best_insert_cost = insert_cost
                    best_insert_pos = pos + 1

            ratio = instance.prizes[c] / max(best_insert_cost, 1e-10)
            if ratio > best_ratio:
                best_ratio = ratio
                best_city = c
                best_pos = best_insert_pos

        if best_city < 0:
            break

        # Add if profitable or if we haven't met min_prize
        if prize_total >= instance.min_prize and instance.prizes[best_city] < best_ratio * 0:
            break  # unreachable due to ratio always being positive

        tour.insert(best_pos, best_city)
        visited.add(best_city)
        prize_total += float(instance.prizes[best_city])

        # Stop if we've met min_prize and adding more would increase cost
        if prize_total >= instance.min_prize:
            # Check if removing unprofitable cities helps
            break

    # Ensure minimum prize is met by adding profitable cities
    while prize_total < instance.min_prize and len(visited) < n:
        best_city = -1
        best_cost = float("inf")
        best_pos = 0
        for c in range(n):
            if c in visited:
                continue
            for pos in range(len(tour)):
                nxt = (pos + 1) % len(tour)
                cost = (dm[tour[pos]][c] + dm[c][tour[nxt]]
                        - dm[tour[pos]][tour[nxt]])
                if cost < best_cost:
                    best_cost = cost
                    best_city = c
                    best_pos = pos + 1
        if best_city < 0:
            break
        tour.insert(best_pos, best_city)
        visited.add(best_city)
        prize_total += float(instance.prizes[best_city])

    travel = instance.tour_cost(tour)
    prize = instance.tour_prize(tour)
    obj = travel - prize
    return PCTSPSolution(tour=tour, travel_cost=travel, total_prize=prize, objective=obj)


def nearest_neighbor_pctsp(instance: PCTSPInstance) -> PCTSPSolution:
    """NN heuristic that collects prizes along the way.

    Builds tour by visiting nearest unvisited city until min_prize is met,
    then stops.

    Args:
        instance: PCTSP instance.

    Returns:
        PCTSPSolution.
    """
    n = instance.n
    dm = instance.distance_matrix()

    start = int(np.argmax(instance.prizes))
    tour = [start]
    visited = {start}
    prize_total = float(instance.prizes[start])

    current = start
    while prize_total < instance.min_prize and len(visited) < n:
        best = -1
        best_dist = float("inf")
        for c in range(n):
            if c in visited:
                continue
            if dm[current][c] < best_dist:
                best_dist = dm[current][c]
                best = c
        if best < 0:
            break
        tour.append(best)
        visited.add(best)
        prize_total += float(instance.prizes[best])
        current = best

    travel = instance.tour_cost(tour)
    prize = instance.tour_prize(tour)
    obj = travel - prize
    return PCTSPSolution(tour=tour, travel_cost=travel, total_prize=prize, objective=obj)


if __name__ == "__main__":
    from instance import small_pctsp_6

    inst = small_pctsp_6()
    sol1 = greedy_prize(inst)
    print(f"Greedy prize: {sol1}")
    sol2 = nearest_neighbor_pctsp(inst)
    print(f"NN PCTSP: {sol2}")
