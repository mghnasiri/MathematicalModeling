"""
Constructive Heuristics for Open VRP.

Problem: OVRP
Complexity: O(n^2)

1. Nearest Neighbor OVRP: build routes greedily with nearest feasible
   customer; no return to depot.
2. Savings-based OVRP: modified Clarke-Wright where merge savings
   exclude return trip to depot.

References:
    Sariklis, D. & Powell, S. (2000). A heuristic method for the open
    vehicle routing problem. JORS, 51(5), 564-573.
    https://doi.org/10.1057/palgrave.jors.2600924
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


_inst = _load_mod("ovrp_instance_h", os.path.join(_this_dir, "instance.py"))
OVRPInstance = _inst.OVRPInstance
OVRPSolution = _inst.OVRPSolution


def nearest_neighbor_ovrp(instance: OVRPInstance) -> OVRPSolution:
    """Build routes using nearest neighbor for OVRP.

    Start from depot, greedily add nearest feasible customer.
    Open a new route when capacity is exhausted.

    Args:
        instance: An OVRPInstance.

    Returns:
        OVRPSolution.
    """
    n = instance.n
    dist = instance.distance_matrix
    visited = [False] * (n + 1)
    routes: list[list[int]] = []

    remaining = n
    while remaining > 0:
        route: list[int] = []
        load = 0.0
        current = 0  # start at depot

        while True:
            best_c = -1
            best_d = float("inf")
            for c in range(1, n + 1):
                if visited[c]:
                    continue
                if load + instance.demands[c] > instance.capacity + 1e-10:
                    continue
                if dist[current][c] < best_d:
                    best_d = dist[current][c]
                    best_c = c

            if best_c < 0:
                break

            route.append(best_c)
            visited[best_c] = True
            load += instance.demands[best_c]
            current = best_c
            remaining -= 1

        if route:
            routes.append(route)

    total_dist = sum(instance.route_distance(r) for r in routes)
    return OVRPSolution(routes=routes, total_distance=total_dist)


def savings_ovrp(instance: OVRPInstance) -> OVRPSolution:
    """Modified Clarke-Wright savings for OVRP.

    Savings s(i,j) = d(0,i) + d(0,j) - d(i,j) but route cost
    doesn't include return to depot.

    Args:
        instance: An OVRPInstance.

    Returns:
        OVRPSolution.
    """
    n = instance.n
    dist = instance.distance_matrix

    # Start with individual routes for each customer
    routes: list[list[int]] = [[c] for c in range(1, n + 1)]
    route_of: dict[int, int] = {c: c - 1 for c in range(1, n + 1)}

    # Compute savings
    savings = []
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            s = dist[0][i] + dist[0][j] - dist[i][j]
            savings.append((s, i, j))
    savings.sort(reverse=True)

    for s, i, j in savings:
        ri = route_of[i]
        rj = route_of[j]
        if ri == rj:
            continue

        # Can only merge if combined demand fits
        combined_demand = sum(instance.demands[c] for c in routes[ri]) + \
                          sum(instance.demands[c] for c in routes[rj])
        if combined_demand > instance.capacity + 1e-10:
            continue

        # Merge: i should be at end of ri, j at start of rj (or vice versa)
        if routes[ri][-1] == i and routes[rj][0] == j:
            merged = routes[ri] + routes[rj]
        elif routes[ri][0] == i and routes[rj][-1] == j:
            merged = routes[rj] + routes[ri]
        elif routes[ri][-1] == i and routes[rj][-1] == j:
            merged = routes[ri] + routes[rj][::-1]
        elif routes[ri][0] == i and routes[rj][0] == j:
            merged = routes[ri][::-1] + routes[rj]
        else:
            continue

        routes[ri] = merged
        routes[rj] = []
        for c in merged:
            route_of[c] = ri

    routes = [r for r in routes if r]
    total_dist = sum(instance.route_distance(r) for r in routes)
    return OVRPSolution(routes=routes, total_distance=total_dist)


if __name__ == "__main__":
    inst = _inst.small_ovrp_6()
    print(f"OVRP: {inst.n} customers")

    sol1 = nearest_neighbor_ovrp(inst)
    print(f"NN: dist={sol1.total_distance:.1f}, routes={len(sol1.routes)}")

    sol2 = savings_ovrp(inst)
    print(f"Savings: dist={sol2.total_distance:.1f}, routes={len(sol2.routes)}")
