"""
Split Delivery VRP (SDVRP) — Constructive Heuristics.

Algorithms:
    - Route-first split-second: build routes greedily, split overflows.
    - Nearest neighbor with splitting: NN that splits when capacity hit.

References:
    Dror, M. & Trudeau, P. (1989). Savings by split delivery routing.
    Transportation Science, 23(2), 141-145.
    https://doi.org/10.1287/trsc.23.2.141
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


_inst = _load_mod("sdvrp_instance_h", os.path.join(_this_dir, "instance.py"))
SDVRPInstance = _inst.SDVRPInstance
SDVRPSolution = _inst.SDVRPSolution


def nearest_neighbor_split(instance: SDVRPInstance) -> SDVRPSolution:
    """Nearest neighbor heuristic with split deliveries.

    At each step, visit the nearest unfinished customer and deliver
    as much as capacity allows. Start a new route when empty.

    Args:
        instance: SDVRP instance.

    Returns:
        SDVRPSolution.
    """
    n = instance.n
    remaining = instance.demands.copy()
    routes: list[list[tuple[int, float]]] = []
    total_dist = 0.0

    while np.any(remaining > 1e-6):
        route: list[tuple[int, float]] = []
        load = 0.0
        current = 0  # depot

        while load < instance.capacity - 1e-6:
            # Find nearest customer with remaining demand
            best_cust = -1
            best_dist = float("inf")
            for j in range(n):
                if remaining[j] < 1e-6:
                    continue
                d = instance.distance(current, j + 1)
                if d < best_dist:
                    best_dist = d
                    best_cust = j

            if best_cust < 0:
                break

            available = instance.capacity - load
            deliver = min(remaining[best_cust], available)
            route.append((best_cust + 1, deliver))
            remaining[best_cust] -= deliver
            load += deliver
            total_dist += instance.distance(current, best_cust + 1)
            current = best_cust + 1

        if route:
            total_dist += instance.distance(current, 0)
            routes.append(route)

    return SDVRPSolution(routes=routes, total_distance=total_dist)


def savings_split(instance: SDVRPInstance) -> SDVRPSolution:
    """Clarke-Wright savings adapted for split deliveries.

    Initially creates one route per customer (potentially split across
    multiple routes if demand > capacity), then merges routes by savings.

    Args:
        instance: SDVRP instance.

    Returns:
        SDVRPSolution.
    """
    n = instance.n
    dm = instance.distance_matrix()

    # Initialize: one route per customer, split if demand > capacity
    routes: list[list[tuple[int, float]]] = []
    for j in range(n):
        rem = instance.demands[j]
        while rem > 1e-6:
            qty = min(rem, instance.capacity)
            routes.append([(j + 1, qty)])
            rem -= qty

    def _route_load(route):
        return sum(q for _, q in route)

    # Compute savings for merging route tails/heads
    improved = True
    while improved:
        improved = False
        best_saving = 0.0
        best_pair = (-1, -1)

        for i in range(len(routes)):
            for j in range(i + 1, len(routes)):
                li = _route_load(routes[i])
                lj = _route_load(routes[j])
                if li + lj > instance.capacity + 1e-6:
                    continue
                last_i = routes[i][-1][0]
                first_j = routes[j][0][0]
                saving = dm[last_i][0] + dm[0][first_j] - dm[last_i][first_j]
                if saving > best_saving + 1e-10:
                    best_saving = saving
                    best_pair = (i, j)

        if best_pair[0] >= 0:
            i, j = best_pair
            routes[i] = routes[i] + routes[j]
            routes.pop(j)
            improved = True

    # Compute total distance
    total_dist = 0.0
    for route in routes:
        nodes = [c for c, _ in route]
        total_dist += instance.route_distance(nodes)

    return SDVRPSolution(routes=routes, total_distance=total_dist)


if __name__ == "__main__":
    from instance import small_sdvrp_6

    inst = small_sdvrp_6()
    sol1 = nearest_neighbor_split(inst)
    print(f"NN Split: {sol1}")
    sol2 = savings_split(inst)
    print(f"Savings Split: {sol2}")
