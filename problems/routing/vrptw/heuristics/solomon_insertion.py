"""
Solomon's Insertion Heuristic — VRPTW constructive heuristic.

Problem: VRPTW (Vehicle Routing Problem with Time Windows)
Complexity: O(n^2 * K) where K is the number of routes

Solomon's I1 heuristic: iteratively insert customers into routes based
on a composite criterion balancing distance increase and time urgency.
When no more customers can be inserted into the current route, a new
route is started with the farthest unrouted customer.

Criterion c1(i,u,j): cost of inserting u between i and j in current route.
    c1 = alpha1 * (d(i,u) + d(u,j) - mu * d(i,j))
       + alpha2 * (new_service_time_j - old_service_time_j)

Criterion c2(u): selects best unrouted customer for insertion.
    c2 = lambda * d(0,u) - c1*(i*,u,j*)  (urgency vs insertion cost)

References:
    Solomon, M.M. (1987). Algorithms for the vehicle routing and
    scheduling problems with time window constraints. Operations
    Research, 35(2), 254-265.
    https://doi.org/10.1287/opre.35.2.254
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


_inst = _load_module("vrptw_instance_si", os.path.join(_parent_dir, "instance.py"))
VRPTWInstance = _inst.VRPTWInstance
VRPTWSolution = _inst.VRPTWSolution


def _can_insert(
    instance: VRPTWInstance,
    route: list[int],
    customer: int,
    position: int,
) -> bool:
    """Check if inserting customer at position is feasible.

    Args:
        instance: VRPTW instance.
        route: Current route (1-indexed customers).
        customer: Customer to insert (1-indexed).
        position: Position in route to insert at (0 = before first).

    Returns:
        True if insertion is feasible (capacity + time windows).
    """
    # Check capacity
    demand = instance.route_demand(route) + instance.demands[customer - 1]
    if demand > instance.capacity + 1e-10:
        return False

    # Build trial route
    trial = route[:position] + [customer] + route[position:]

    # Check all time windows in trial route
    current_time = instance.time_windows[0][0]
    prev = 0
    for cust in trial:
        arrival = current_time + instance.travel_time(prev, cust)
        if arrival > instance.time_windows[cust][1] + 1e-10:
            return False
        start = max(arrival, instance.time_windows[cust][0])
        current_time = start + instance.service_times[cust]
        prev = cust

    # Check return to depot
    return_time = current_time + instance.travel_time(prev, 0)
    if return_time > instance.time_windows[0][1] + 1e-10:
        return False

    return True


def _insertion_cost(
    instance: VRPTWInstance,
    route: list[int],
    customer: int,
    position: int,
    alpha1: float = 1.0,
    alpha2: float = 0.0,
    mu: float = 1.0,
) -> float:
    """Compute the c1 insertion cost for Solomon's I1 heuristic.

    Args:
        instance: VRPTW instance.
        route: Current route.
        customer: Customer to insert.
        position: Position to insert at.
        alpha1: Weight for distance component.
        alpha2: Weight for time push-forward component.
        mu: Weight for removed edge in distance component.

    Returns:
        Insertion cost c1.
    """
    dist = instance.distance_matrix
    u = customer

    # Nodes before and after insertion point
    i_node = 0 if position == 0 else route[position - 1]
    j_node = 0 if position >= len(route) else route[position]

    # Distance component
    dist_cost = dist[i_node][u] + dist[u][j_node] - mu * dist[i_node][j_node]

    # Time push-forward component
    if alpha2 > 0 and j_node != 0:
        # Compute original arrival at j
        trial_without = route[:]
        orig_schedule = instance.route_schedule(trial_without)
        orig_j_time = orig_schedule[position] if position < len(orig_schedule) else 0.0

        # Compute new arrival at j after inserting u
        trial_with = route[:position] + [customer] + route[position:]
        new_schedule = instance.route_schedule(trial_with)
        new_j_time = new_schedule[position + 1] if position + 1 < len(new_schedule) else 0.0

        time_cost = new_j_time - orig_j_time
    else:
        time_cost = 0.0

    return alpha1 * dist_cost + alpha2 * time_cost


def solomon_insertion(
    instance: VRPTWInstance,
    alpha1: float = 1.0,
    alpha2: float = 0.0,
    mu: float = 1.0,
    lam: float = 1.0,
) -> VRPTWSolution:
    """Construct VRPTW routes using Solomon's I1 insertion heuristic.

    Args:
        instance: A VRPTWInstance.
        alpha1: Weight for distance cost in c1.
        alpha2: Weight for time push-forward in c1.
        mu: Weight for removed edge in c1.
        lam: Weight for depot distance in c2 selection.

    Returns:
        VRPTWSolution with constructed routes.
    """
    n = instance.n
    dist = instance.distance_matrix
    unrouted = set(range(1, n + 1))
    routes: list[list[int]] = []

    while unrouted:
        # Start new route: choose farthest unrouted customer from depot
        seed = max(unrouted, key=lambda c: dist[0][c])
        if not _can_insert(instance, [], seed, 0):
            # If even a single customer can't be routed, force it
            # (this shouldn't happen with valid instances)
            routes.append([seed])
            unrouted.remove(seed)
            continue

        route = [seed]
        unrouted.remove(seed)

        # Iteratively insert best customer
        improved = True
        while improved and unrouted:
            improved = False
            best_c2 = -float("inf")
            best_customer = -1
            best_pos = -1

            for customer in unrouted:
                # Find best insertion position for this customer
                best_c1 = float("inf")
                best_p = -1

                for pos in range(len(route) + 1):
                    if _can_insert(instance, route, customer, pos):
                        c1 = _insertion_cost(
                            instance, route, customer, pos,
                            alpha1, alpha2, mu,
                        )
                        if c1 < best_c1:
                            best_c1 = c1
                            best_p = pos

                if best_p >= 0:
                    # c2 criterion: prefer customers far from depot with low c1
                    c2 = lam * dist[0][customer] - best_c1
                    if c2 > best_c2:
                        best_c2 = c2
                        best_customer = customer
                        best_pos = best_p

            if best_customer >= 0:
                route.insert(best_pos, best_customer)
                unrouted.remove(best_customer)
                improved = True

        routes.append(route)

    total_dist = instance.total_distance(routes)
    return VRPTWSolution(routes=routes, distance=total_dist)


def nearest_neighbor_tw(instance: VRPTWInstance) -> VRPTWSolution:
    """Construct VRPTW routes using nearest neighbor with time window checks.

    At each step, visit the nearest feasible unrouted customer.
    Start a new route when no feasible extension exists.

    Args:
        instance: A VRPTWInstance.

    Returns:
        VRPTWSolution with constructed routes.
    """
    n = instance.n
    dist = instance.distance_matrix
    unrouted = set(range(1, n + 1))
    routes: list[list[int]] = []

    while unrouted:
        route: list[int] = []
        current_time = instance.time_windows[0][0]
        current_load = 0.0
        prev = 0

        while True:
            best_next = -1
            best_dist = float("inf")

            for cust in unrouted:
                # Check capacity
                if current_load + instance.demands[cust - 1] > instance.capacity + 1e-10:
                    continue
                # Check time window
                arrival = current_time + instance.travel_time(prev, cust)
                if arrival > instance.time_windows[cust][1] + 1e-10:
                    continue
                # Check return to depot after serving
                start = max(arrival, instance.time_windows[cust][0])
                depart = start + instance.service_times[cust]
                return_time = depart + instance.travel_time(cust, 0)
                if return_time > instance.time_windows[0][1] + 1e-10:
                    continue

                if dist[prev][cust] < best_dist:
                    best_dist = dist[prev][cust]
                    best_next = cust

            if best_next < 0:
                break

            route.append(best_next)
            unrouted.remove(best_next)
            arrival = current_time + instance.travel_time(prev, best_next)
            start = max(arrival, instance.time_windows[best_next][0])
            current_time = start + instance.service_times[best_next]
            current_load += instance.demands[best_next - 1]
            prev = best_next

        if route:
            routes.append(route)
        elif unrouted:
            # Force-assign a customer that can't fit anywhere
            cust = min(unrouted)
            routes.append([cust])
            unrouted.remove(cust)

    total_dist = instance.total_distance(routes)
    return VRPTWSolution(routes=routes, distance=total_dist)


if __name__ == "__main__":
    from instance import solomon_c101_mini, tight_tw5

    print("=== Solomon Insertion Heuristic ===\n")

    for name, inst_fn in [
        ("solomon_c101_mini", solomon_c101_mini),
        ("tight_tw5", tight_tw5),
    ]:
        inst = inst_fn()
        sol = solomon_insertion(inst)
        nn_sol = nearest_neighbor_tw(inst)
        print(f"{name}: solomon={sol.distance:.1f} ({sol.num_vehicles} vehicles), "
              f"nn={nn_sol.distance:.1f} ({nn_sol.num_vehicles} vehicles)")
