"""
Cheapest Insertion Heuristic — VRPPD constructive heuristic.

Problem: VRPPD (VRP with Pickup and Delivery)
Complexity: O(n^2 * K) where K is the number of routes

Iteratively insert pickup-delivery pairs into routes. For each unassigned
request, find the cheapest feasible insertion of its pickup and delivery
(maintaining precedence: pickup before delivery). When no more pairs can
be inserted into the current route, start a new route.

References:
    Lu, Q. & Dessouky, M.M. (2004). An exact algorithm for the multiple
    vehicle pickup and delivery problem. Transportation Science, 38(4),
    503-514.
    https://doi.org/10.1287/trsc.1030.0040
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
    "vrppd_instance_ins", os.path.join(_parent_dir, "instance.py")
)
VRPPDInstance = _inst.VRPPDInstance
VRPPDSolution = _inst.VRPPDSolution


def _try_insert_pair(
    instance: VRPPDInstance,
    route: list[int],
    pickup: int,
    delivery: int,
) -> tuple[float, int, int]:
    """Find cheapest feasible insertion of a pickup-delivery pair into a route.

    The pickup must be inserted before the delivery. Checks capacity and
    precedence feasibility for each candidate position pair.

    Args:
        instance: VRPPD instance.
        route: Current route (list of node indices).
        pickup: Pickup node index.
        delivery: Delivery node index.

    Returns:
        Tuple of (cost_increase, pickup_pos, delivery_pos).
        cost_increase is inf if no feasible insertion exists.
    """
    best_cost = float("inf")
    best_p_pos = -1
    best_d_pos = -1
    req = instance.request_of_node(pickup)
    load = instance.loads[req]

    for p_pos in range(len(route) + 1):
        # Check capacity at pickup position
        running_load = 0.0
        capacity_ok = True
        for k in range(len(route)):
            if k == p_pos:
                running_load += load
                if running_load > instance.capacity + 1e-10:
                    capacity_ok = False
                    break
            node = route[k]
            if instance.is_pickup(node):
                running_load += instance.loads[instance.request_of_node(node)]
            elif instance.is_delivery(node):
                running_load -= instance.loads[instance.request_of_node(node)]
            if running_load > instance.capacity + 1e-10:
                capacity_ok = False
                break
        if not capacity_ok:
            continue
        # Also check when pickup is inserted at end
        if p_pos == len(route):
            temp_load = sum(
                instance.loads[instance.request_of_node(node)]
                for node in route
                if instance.is_pickup(node)
            ) - sum(
                instance.loads[instance.request_of_node(node)]
                for node in route
                if instance.is_delivery(node)
            ) + load
            if temp_load > instance.capacity + 1e-10:
                continue

        for d_pos in range(p_pos + 1, len(route) + 2):
            # Build trial route
            trial = route[:p_pos] + [pickup] + route[p_pos:]
            trial = trial[:d_pos] + [delivery] + trial[d_pos:]

            # Verify feasibility
            feasible, _ = instance.route_feasible(trial)
            if not feasible:
                continue

            cost = instance.route_distance(trial) - instance.route_distance(route)
            if cost < best_cost:
                best_cost = cost
                best_p_pos = p_pos
                best_d_pos = d_pos

    return best_cost, best_p_pos, best_d_pos


def cheapest_insertion_vrppd(
    instance: VRPPDInstance,
) -> VRPPDSolution:
    """Construct VRPPD routes using cheapest pair insertion.

    For each unassigned request, tries to insert the pickup-delivery pair
    into an existing route at the cheapest feasible positions. When no
    insertion is feasible, opens a new route.

    Args:
        instance: A VRPPDInstance.

    Returns:
        VRPPDSolution with constructed routes.
    """
    n = instance.n_requests
    unassigned = set(range(n))
    routes: list[list[int]] = []

    while unassigned:
        # Start a new route with the request farthest from depot
        seed_req = max(
            unassigned,
            key=lambda r: instance.distance_matrix[0][instance.pickup_of(r)],
        )
        p = instance.pickup_of(seed_req)
        d = instance.delivery_of(seed_req)
        routes.append([p, d])
        unassigned.remove(seed_req)

        # Try to insert remaining requests into the current route
        improved = True
        while improved and unassigned:
            improved = False
            best_cost = float("inf")
            best_req = -1
            best_p_pos = -1
            best_d_pos = -1

            for req in unassigned:
                pickup = instance.pickup_of(req)
                delivery = instance.delivery_of(req)
                cost, p_pos, d_pos = _try_insert_pair(
                    instance, routes[-1], pickup, delivery
                )
                if cost < best_cost:
                    best_cost = cost
                    best_req = req
                    best_p_pos = p_pos
                    best_d_pos = d_pos

            if best_req >= 0 and best_cost < float("inf"):
                pickup = instance.pickup_of(best_req)
                delivery = instance.delivery_of(best_req)
                route = routes[-1]
                route.insert(best_p_pos, pickup)
                route.insert(best_d_pos, delivery)
                unassigned.remove(best_req)
                improved = True

    total_dist = instance.total_distance(routes)
    return VRPPDSolution(routes=routes, distance=total_dist)


if __name__ == "__main__":
    from instance import small_vrppd3, medium_vrppd5

    print("=== Cheapest Insertion VRPPD ===\n")

    for name, inst_fn in [
        ("small_vrppd3", small_vrppd3),
        ("medium_vrppd5", medium_vrppd5),
    ]:
        inst = inst_fn()
        sol = cheapest_insertion_vrppd(inst)
        print(f"{name}: {sol}")
