"""
Cheapest Insertion Heuristic for the Dial-a-Ride Problem.

Problem: DARP (m vehicles | pickup-delivery, time windows | min distance)
Complexity: O(n^2 * m) per request insertion

Algorithm:
For each unserved request, find the cheapest position to insert the
pickup-delivery pair into any existing route. If no feasible insertion
exists (capacity/time), create a new route.

References:
    Jaw, J.J., Odoni, A.R., Psaraftis, H.N. & Wilson, N.H.M. (1986).
    A heuristic algorithm for the multi-vehicle advance request
    dial-a-ride problem with time windows. Transportation Research
    Part B, 20(3), 243-257.
    https://doi.org/10.1016/0191-2615(86)90020-2
"""

from __future__ import annotations

import os
import sys
import importlib.util

import numpy as np

_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_parent(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_inst = _load_parent("darp_instance_ins", os.path.join(_parent_dir, "instance.py"))
DARPInstance = _inst.DARPInstance
DARPSolution = _inst.DARPSolution
DARPRequest = _inst.DARPRequest


def _route_distance(instance: DARPInstance, route: list[int]) -> float:
    """Compute route distance."""
    if len(route) < 2:
        return 0.0
    return sum(
        instance.distance(route[i], route[i + 1])
        for i in range(len(route) - 1)
    )


def _insertion_cost(
    instance: DARPInstance,
    route: list[int],
    pickup: int,
    delivery: int,
    p_pos: int,
    d_pos: int,
) -> float:
    """Compute the cost increase of inserting pickup at p_pos and delivery at d_pos."""
    new_route = list(route)
    new_route.insert(p_pos, pickup)
    new_route.insert(d_pos, delivery)  # d_pos already adjusted since pickup was inserted
    return _route_distance(instance, new_route) - _route_distance(instance, route)


def _check_capacity(
    instance: DARPInstance,
    route: list[int],
    req: DARPRequest,
) -> bool:
    """Check if adding request to route doesn't exceed capacity at any point."""
    load = 0
    for node in route:
        for r in instance.requests:
            if node == r.pickup:
                load += r.load
            elif node == r.delivery:
                load -= r.load
        if load > instance.vehicle_capacity:
            return False
    return True


def cheapest_insertion_darp(
    instance: DARPInstance,
    seed: int | None = None,
) -> DARPSolution:
    """Solve DARP using cheapest insertion heuristic.

    For each request (sorted by earliest pickup time), find the best
    position to insert its pickup-delivery pair into any route.

    Args:
        instance: A DARPInstance.
        seed: Random seed (for tie-breaking).

    Returns:
        DARPSolution with all requests served.
    """
    rng = np.random.default_rng(seed)
    n = instance.n_requests

    # Sort requests by earliest pickup time
    order = sorted(
        range(n),
        key=lambda i: instance.requests[i].earliest_pickup,
    )

    # Initialize routes (one per vehicle, empty)
    routes: list[list[int]] = [
        [instance.depot_start, instance.depot_end]
        for _ in range(instance.n_vehicles)
    ]

    served = set()

    for req_idx in order:
        req = instance.requests[req_idx]
        p_node = instance.pickup_node(req_idx)
        d_node = instance.delivery_node(req_idx)

        best_cost = float("inf")
        best_route_idx = -1
        best_p_pos = -1
        best_d_pos = -1

        for r_idx, route in enumerate(routes):
            # Try all positions for pickup (after depot_start, before depot_end)
            for p_pos in range(1, len(route)):
                # Delivery must come after pickup
                for d_pos in range(p_pos + 1, len(route) + 1):
                    cost = _insertion_cost(
                        instance, route, p_node, d_node, p_pos, d_pos
                    )
                    if cost < best_cost:
                        # Check capacity feasibility
                        trial_route = list(route)
                        trial_route.insert(p_pos, p_node)
                        trial_route.insert(d_pos, d_node)
                        if _check_capacity(instance, trial_route, req):
                            best_cost = cost
                            best_route_idx = r_idx
                            best_p_pos = p_pos
                            best_d_pos = d_pos

        if best_route_idx >= 0:
            routes[best_route_idx].insert(best_p_pos, p_node)
            routes[best_route_idx].insert(best_d_pos, d_node)
            served.add(req_idx)
        else:
            # If no feasible insertion found, add a new route
            new_route = [instance.depot_start, p_node, d_node, instance.depot_end]
            routes.append(new_route)
            served.add(req_idx)

    # Remove empty routes
    routes = [r for r in routes if len(r) > 2]

    total_dist = sum(_route_distance(instance, r) for r in routes)

    return DARPSolution(
        routes=routes,
        total_distance=round(total_dist, 2),
        n_served=len(served),
    )


if __name__ == "__main__":
    _inst_mod = _load_parent(
        "darp_inst_main", os.path.join(_parent_dir, "instance.py")
    )
    inst = _inst_mod.small_darp3()
    sol = cheapest_insertion_darp(inst)
    print(f"DARP cheapest insertion: {sol}")
    for i, r in enumerate(sol.routes):
        print(f"  Route {i}: {r}")
