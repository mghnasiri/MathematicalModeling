"""
Local Search for TSP — 2-opt and Or-opt neighborhoods.

Problem: TSP (Traveling Salesman Problem)

2-opt: Reverse a segment of the tour. O(n^2) per iteration.
    Removes two edges and reconnects the tour by reversing the path between.
Or-opt: Relocate a segment of 1-3 cities. O(n^2) per iteration.

References:
    Croes, G.A. (1958). A method for solving traveling salesman problems.
    Operations Research, 6(6), 791-812.
    https://doi.org/10.1287/opre.6.6.791

    Lin, S. (1965). Computer solutions of the traveling salesman problem.
    Bell System Technical Journal, 44(10), 2245-2269.
    https://doi.org/10.1002/j.1538-7305.1965.tb04146.x

    Or, I. (1976). Traveling salesman-type combinatorial problems and their
    relation to the logistics of regional blood banking. Ph.D. thesis,
    Northwestern University.
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

_inst = _load_module("tsp_instance_ls", os.path.join(_parent_dir, "instance.py"))
TSPInstance = _inst.TSPInstance
TSPSolution = _inst.TSPSolution


def two_opt(instance: TSPInstance, initial_tour: list[int] | None = None) -> TSPSolution:
    """Improve a tour using 2-opt local search.

    Args:
        instance: A TSPInstance.
        initial_tour: Starting tour. If None, uses identity permutation.

    Returns:
        TSPSolution with locally optimal tour (2-opt).
    """
    n = instance.n
    dist = instance.distance_matrix

    if n <= 3:
        tour = list(range(n)) if initial_tour is None else initial_tour[:]
        return TSPSolution(tour=tour, distance=instance.tour_distance(tour))

    tour = list(range(n)) if initial_tour is None else initial_tour[:]
    improved = True

    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue  # Would reverse entire tour
                # Cost of removing edges (i, i+1) and (j, j+1%n)
                # and adding edges (i, j) and (i+1, j+1%n)
                a, b = tour[i], tour[i + 1]
                c, d = tour[j], tour[(j + 1) % n]
                delta = (dist[a][c] + dist[b][d]) - (dist[a][b] + dist[c][d])
                if delta < -1e-10:
                    # Reverse segment [i+1..j]
                    tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                    improved = True

    return TSPSolution(tour=tour, distance=instance.tour_distance(tour))


def or_opt(instance: TSPInstance, initial_tour: list[int] | None = None) -> TSPSolution:
    """Improve a tour using Or-opt local search (relocate segments of 1-3 cities).

    Args:
        instance: A TSPInstance.
        initial_tour: Starting tour. If None, uses identity permutation.

    Returns:
        TSPSolution with locally optimal tour (Or-opt).
    """
    n = instance.n
    dist = instance.distance_matrix

    if n <= 3:
        tour = list(range(n)) if initial_tour is None else initial_tour[:]
        return TSPSolution(tour=tour, distance=instance.tour_distance(tour))

    tour = list(range(n)) if initial_tour is None else initial_tour[:]
    improved = True

    while improved:
        improved = False
        for seg_len in [1, 2, 3]:
            if improved:
                break
            for i in range(n):
                if improved:
                    break
                # Segment to relocate: tour[i..i+seg_len-1] (mod n)
                if i + seg_len > n:
                    continue  # Skip wrap-around for simplicity

                prev_i = (i - 1) % n
                next_seg = (i + seg_len) % n

                # Cost of removing segment
                remove_cost = (
                    dist[tour[prev_i]][tour[i]]
                    + dist[tour[i + seg_len - 1]][tour[next_seg]]
                )
                bridge_cost = dist[tour[prev_i]][tour[next_seg]]

                # Try inserting segment after each position
                for j in range(n):
                    if j >= i - 1 and j <= i + seg_len - 1:
                        continue
                    next_j = (j + 1) % n

                    # Cost of inserting segment between j and next_j
                    insert_cost = (
                        dist[tour[j]][tour[i]]
                        + dist[tour[i + seg_len - 1]][tour[next_j]]
                    )
                    old_edge = dist[tour[j]][tour[next_j]]

                    delta = (bridge_cost + insert_cost) - (remove_cost + old_edge)
                    if delta < -1e-10:
                        # Perform the move
                        segment = tour[i:i + seg_len]
                        new_tour = tour[:i] + tour[i + seg_len:]
                        # Find new position of j
                        if j < i:
                            ins_pos = j + 1
                        else:
                            ins_pos = j + 1 - seg_len
                        for k, city in enumerate(segment):
                            new_tour.insert(ins_pos + k, city)
                        tour = new_tour
                        improved = True
                        break

    return TSPSolution(tour=tour, distance=instance.tour_distance(tour))


def vnd(instance: TSPInstance, initial_tour: list[int] | None = None) -> TSPSolution:
    """Variable Neighborhood Descent combining 2-opt and Or-opt.

    Args:
        instance: A TSPInstance.
        initial_tour: Starting tour. If None, uses identity permutation.

    Returns:
        TSPSolution with locally optimal tour.
    """
    tour = list(range(instance.n)) if initial_tour is None else initial_tour[:]

    improved = True
    while improved:
        sol = two_opt(instance, tour)
        if sol.distance < instance.tour_distance(tour) - 1e-10:
            tour = sol.tour
        sol = or_opt(instance, tour)
        if sol.distance < instance.tour_distance(tour) - 1e-10:
            tour = sol.tour
            improved = True
        else:
            improved = False

    return TSPSolution(tour=tour, distance=instance.tour_distance(tour))


if __name__ == "__main__":
    from instance import small4, small5, gr17

    print("=== Local Search ===\n")

    for name, inst_fn in [("small4", small4), ("small5", small5), ("gr17", gr17)]:
        inst = inst_fn()
        sol_2opt = two_opt(inst)
        sol_oropt = or_opt(inst)
        sol_vnd = vnd(inst)
        print(f"{name}: 2-opt={sol_2opt.distance:.1f}, "
              f"or-opt={sol_oropt.distance:.1f}, vnd={sol_vnd.distance:.1f}")
