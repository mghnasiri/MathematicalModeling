"""
Multi-Compartment VRP — Heuristics.

Algorithms:
    - Nearest neighbor with compartment capacity checks.
    - Clarke-Wright savings adapted for multi-compartment.

References:
    Derigs, U., Gottlieb, J. & Kalkoff, J. (2011). Vehicle routing with
    compartments: Applications, modelling and heuristics. OR Spectrum, 33(4),
    885-914. https://doi.org/10.1007/s00291-009-0175-6
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


_inst = _load_mod("mcvrp_instance_h", os.path.join(_this_dir, "instance.py"))
MCVRPInstance = _inst.MCVRPInstance
MCVRPSolution = _inst.MCVRPSolution


def nearest_neighbor_mcvrp(instance: MCVRPInstance) -> MCVRPSolution:
    """Nearest neighbor heuristic for MCVRP.

    Greedily assigns nearest customer that fits all compartments.

    Args:
        instance: MCVRP instance.

    Returns:
        MCVRPSolution.
    """
    unvisited = set(range(1, instance.n + 1))
    routes = []

    while unvisited:
        route = []
        loads = np.zeros(instance.num_compartments)
        current = 0

        while unvisited:
            best_cust = None
            best_dist = float("inf")
            for c in unvisited:
                new_loads = loads + instance.demands[c - 1]
                if np.all(new_loads <= instance.compartment_capacities + 1e-6):
                    d = instance.dist(current, c)
                    if d < best_dist:
                        best_dist = d
                        best_cust = c
            if best_cust is None:
                break
            route.append(best_cust)
            loads += instance.demands[best_cust - 1]
            current = best_cust
            unvisited.remove(best_cust)

        if route:
            routes.append(route)

    total_dist = sum(instance.route_distance(r) for r in routes)
    return MCVRPSolution(routes=routes, total_distance=total_dist)


def savings_mcvrp(instance: MCVRPInstance) -> MCVRPSolution:
    """Clarke-Wright savings heuristic adapted for MCVRP.

    Merges routes when combined compartment loads remain feasible.

    Args:
        instance: MCVRP instance.

    Returns:
        MCVRPSolution.
    """
    # Start with one route per customer
    routes = [[c] for c in range(1, instance.n + 1)]

    # Compute savings
    savings = []
    for i in range(1, instance.n + 1):
        for j in range(i + 1, instance.n + 1):
            s = (instance.dist(0, i) + instance.dist(0, j) -
                 instance.dist(i, j))
            savings.append((s, i, j))
    savings.sort(reverse=True)

    # Route lookup
    customer_route = {c: idx for idx, r in enumerate(routes) for c in r}

    for s, i, j in savings:
        ri = customer_route[i]
        rj = customer_route[j]
        if ri == rj:
            continue

        # Check if i is at end and j is at start (or vice versa)
        r1 = routes[ri]
        r2 = routes[rj]
        if not r1 or not r2:
            continue

        can_merge = False
        merged = None
        if r1[-1] == i and r2[0] == j:
            merged = r1 + r2
            can_merge = True
        elif r2[-1] == j and r1[0] == i:
            merged = r2 + r1
            can_merge = True
        elif r1[-1] == i and r2[-1] == j:
            merged = r1 + r2[::-1]
            can_merge = True
        elif r1[0] == i and r2[0] == j:
            merged = r1[::-1] + r2
            can_merge = True

        if not can_merge:
            continue

        # Check compartment feasibility
        loads = instance.route_loads(merged)
        if np.any(loads > instance.compartment_capacities + 1e-6):
            continue

        # Merge
        routes[ri] = merged
        routes[rj] = []
        for c in merged:
            customer_route[c] = ri

    routes = [r for r in routes if r]
    total_dist = sum(instance.route_distance(r) for r in routes)
    return MCVRPSolution(routes=routes, total_distance=total_dist)


if __name__ == "__main__":
    from instance import small_mcvrp_6

    inst = small_mcvrp_6()
    sol1 = nearest_neighbor_mcvrp(inst)
    print(f"NN: {sol1}")
    sol2 = savings_mcvrp(inst)
    print(f"Savings: {sol2}")
