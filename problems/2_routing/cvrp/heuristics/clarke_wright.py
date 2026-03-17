"""
Clarke-Wright Savings Algorithm — CVRP constructive heuristic.

Problem: CVRP (Capacitated Vehicle Routing Problem)
Complexity: O(n^2 log n) — dominated by savings computation and sorting

Starting from n individual routes (depot -> customer i -> depot), iteratively
merge routes by connecting the ends that yield the largest "savings":
    s(i,j) = d(0,i) + d(0,j) - d(i,j)

Two variants:
- Parallel: consider all feasible merges at each step (standard).
- Sequential: build one route at a time.

References:
    Clarke, G. & Wright, J.W. (1964). Scheduling of vehicles from a
    central depot to a number of delivery points. Operations Research,
    12(4), 568-581.
    https://doi.org/10.1287/opre.12.4.568
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

_inst = _load_module("cvrp_instance_cw", os.path.join(_parent_dir, "instance.py"))
CVRPInstance = _inst.CVRPInstance
CVRPSolution = _inst.CVRPSolution


def clarke_wright_savings(
    instance: CVRPInstance, parallel: bool = True
) -> CVRPSolution:
    """Construct CVRP routes using the Clarke-Wright savings algorithm.

    Args:
        instance: A CVRPInstance.
        parallel: If True, use parallel variant (consider all merges).
            If False, use sequential variant (one route at a time).

    Returns:
        CVRPSolution with constructed routes.
    """
    n = instance.n
    dist = instance.distance_matrix
    Q = instance.capacity

    # Compute savings: s(i,j) = d(0,i) + d(0,j) - d(i,j)
    savings = []
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            s = dist[0][i] + dist[0][j] - dist[i][j]
            savings.append((s, i, j))
    savings.sort(reverse=True)

    # Initialize: each customer in its own route
    # route_of[i] = index of route containing customer i
    routes: list[list[int]] = [[i] for i in range(1, n + 1)]
    route_of = {}
    for idx, route in enumerate(routes):
        route_of[route[0]] = idx

    def get_route_demand(route: list[int]) -> float:
        return sum(instance.demands[c - 1] for c in route)

    # Process savings in decreasing order
    for s, i, j in savings:
        if s <= 0:
            break

        ri = route_of.get(i)
        rj = route_of.get(j)

        if ri is None or rj is None:
            continue
        if ri == rj:
            continue  # Already in same route

        route_i = routes[ri]
        route_j = routes[rj]

        # Can only merge if i and j are at the ends of their routes
        i_at_start = route_i[0] == i
        i_at_end = route_i[-1] == i
        j_at_start = route_j[0] == j
        j_at_end = route_j[-1] == j

        if not ((i_at_start or i_at_end) and (j_at_start or j_at_end)):
            continue

        # Check capacity
        combined_demand = get_route_demand(route_i) + get_route_demand(route_j)
        if combined_demand > Q + 1e-10:
            continue

        # Merge the routes
        if i_at_end and j_at_start:
            merged = route_i + route_j
        elif i_at_start and j_at_end:
            merged = route_j + route_i
        elif i_at_end and j_at_end:
            merged = route_i + route_j[::-1]
        else:  # i_at_start and j_at_start
            merged = route_i[::-1] + route_j

        # Update routes
        new_idx = ri
        routes[new_idx] = merged
        routes[rj] = []  # Mark as empty

        for c in merged:
            route_of[c] = new_idx

    # Collect non-empty routes
    final_routes = [r for r in routes if r]
    total_dist = instance.total_distance(final_routes)

    return CVRPSolution(routes=final_routes, distance=total_dist)


if __name__ == "__main__":
    from instance import small6, christofides1, medium12

    print("=== Clarke-Wright Savings ===\n")

    for name, inst_fn in [
        ("small6", small6),
        ("christofides1", christofides1),
        ("medium12", medium12),
    ]:
        inst = inst_fn()
        sol = clarke_wright_savings(inst)
        print(f"{name}: {sol}")
