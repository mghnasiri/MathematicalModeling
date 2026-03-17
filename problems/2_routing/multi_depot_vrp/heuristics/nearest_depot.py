"""
Nearest Depot Assignment + Clarke-Wright Savings — MDVRP heuristic.

Problem: MDVRP (Multi-Depot Vehicle Routing Problem)
Complexity: O(n^2 log n) per depot — dominated by savings sort

Two-phase heuristic:
1. Assign each customer to its nearest depot.
2. For each depot, apply Clarke-Wright savings to build routes.

References:
    Clarke, G. & Wright, J.W. (1964). Scheduling of vehicles from a
    central depot to a number of delivery points. Operations Research,
    12(4), 568-581.
    https://doi.org/10.1287/opre.12.4.568

    Cordeau, J.-F., Gendreau, M. & Laporte, G. (1997). A tabu search
    heuristic for periodic and multi-depot vehicle routing problems.
    Networks, 30(2), 105-119.
    https://doi.org/10.1002/(SICI)1097-0037(199709)30:2<105::AID-NET5>3.0.CO;2-G
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


_inst = _load_module(
    "mdvrp_instance_nd", os.path.join(_parent_dir, "instance.py")
)
MDVRPInstance = _inst.MDVRPInstance
MDVRPSolution = _inst.MDVRPSolution


def _assign_customers_to_depots(
    instance: MDVRPInstance,
) -> dict[int, list[int]]:
    """Assign each customer to its nearest depot.

    Args:
        instance: MDVRP instance.

    Returns:
        Dict mapping depot_idx -> list of customer indices (0-based).
    """
    assignments: dict[int, list[int]] = {d: [] for d in range(instance.n_depots)}

    for c in range(instance.n_customers):
        c_node = instance.customer_node(c)
        best_depot = -1
        best_dist = float("inf")
        for d in range(instance.n_depots):
            d_node = instance.depot_node(d)
            dist = instance.distance_matrix[d_node][c_node]
            if dist < best_dist:
                best_dist = dist
                best_depot = d
        assignments[best_depot].append(c)

    return assignments


def _clarke_wright_for_depot(
    instance: MDVRPInstance,
    depot_idx: int,
    customers: list[int],
) -> list[list[int]]:
    """Apply Clarke-Wright savings for a single depot.

    Args:
        instance: MDVRP instance.
        depot_idx: Depot index.
        customers: List of customer indices assigned to this depot.

    Returns:
        List of routes (each route is a list of customer indices).
    """
    if not customers:
        return []

    d_node = instance.depot_node(depot_idx)
    dist = instance.distance_matrix
    Q = instance.capacity

    # Compute savings
    savings = []
    for i_idx, ci in enumerate(customers):
        ci_node = instance.customer_node(ci)
        for j_idx in range(i_idx + 1, len(customers)):
            cj = customers[j_idx]
            cj_node = instance.customer_node(cj)
            s = (
                dist[d_node][ci_node]
                + dist[d_node][cj_node]
                - dist[ci_node][cj_node]
            )
            savings.append((s, ci, cj))
    savings.sort(reverse=True)

    # Initialize: each customer in its own route
    routes: list[list[int]] = [[c] for c in customers]
    route_of: dict[int, int] = {c: i for i, c in enumerate(customers)}

    for s, ci, cj in savings:
        if s <= 0:
            break

        ri = route_of[ci]
        rj = route_of[cj]
        if ri == rj:
            continue

        # Can only merge if ci is at end of its route and cj at start (or vice versa)
        route_i = routes[ri]
        route_j = routes[rj]

        if not route_i or not route_j:
            continue

        # Check if ci at end and cj at start
        merge = None
        if route_i[-1] == ci and route_j[0] == cj:
            merge = (ri, rj, False)
        elif route_j[-1] == cj and route_i[0] == ci:
            merge = (rj, ri, False)
        elif route_i[-1] == ci and route_j[-1] == cj:
            merge = (ri, rj, True)
        elif route_i[0] == ci and route_j[0] == cj:
            merge = (rj, ri, True)
        else:
            continue

        src, dst, reverse_dst = merge
        src_route = routes[src]
        dst_route = routes[dst]

        # Check capacity
        total_demand = sum(instance.demands[c] for c in src_route + dst_route)
        if total_demand > Q + 1e-10:
            continue

        # Merge
        if reverse_dst:
            new_route = src_route + list(reversed(dst_route))
        else:
            new_route = src_route + dst_route

        routes[src] = new_route
        routes[dst] = []
        for c in new_route:
            route_of[c] = src

    return [r for r in routes if r]


def nearest_depot_cw(instance: MDVRPInstance) -> MDVRPSolution:
    """Solve MDVRP by nearest-depot assignment + Clarke-Wright savings.

    Phase 1: Assign each customer to the nearest depot.
    Phase 2: For each depot, construct routes using Clarke-Wright savings.

    Args:
        instance: An MDVRPInstance.

    Returns:
        MDVRPSolution with constructed routes per depot.
    """
    assignments = _assign_customers_to_depots(instance)

    depot_routes: dict[int, list[list[int]]] = {}
    for depot_idx in range(instance.n_depots):
        customers = assignments[depot_idx]
        routes = _clarke_wright_for_depot(instance, depot_idx, customers)
        depot_routes[depot_idx] = routes

    total_dist = instance.total_distance(depot_routes)
    return MDVRPSolution(
        depot_routes=depot_routes, distance=total_dist
    )


if __name__ == "__main__":
    from instance import small_mdvrp4, medium_mdvrp8

    print("=== Nearest Depot + Clarke-Wright ===\n")

    for name, inst_fn in [
        ("small_mdvrp4", small_mdvrp4),
        ("medium_mdvrp8", medium_mdvrp8),
    ]:
        inst = inst_fn()
        sol = nearest_depot_cw(inst)
        print(f"{name}: {sol}")
        for d, routes in sol.depot_routes.items():
            print(f"  Depot {d}: {routes}")
