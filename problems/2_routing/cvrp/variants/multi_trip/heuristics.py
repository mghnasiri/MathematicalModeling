"""
Multi-Trip VRP — Heuristics.

Algorithms:
    - Greedy bin-packing of routes to vehicles.

References:
    Taillard, E.D., Laporte, G. & Gendreau, M. (1996). Vehicle routeing with
    multiple use of vehicles. Journal of the Operational Research Society,
    47(8), 1065-1070. https://doi.org/10.1057/jors.1996.133
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


_inst = _load_mod("mtvrp_instance_h", os.path.join(_this_dir, "instance.py"))
MTVRPInstance = _inst.MTVRPInstance
MTVRPSolution = _inst.MTVRPSolution


def greedy_multi_trip(instance: MTVRPInstance) -> MTVRPSolution:
    """Build routes via NN, then assign routes to vehicles.

    First builds capacity-feasible routes, then assigns to vehicles
    using a round-robin approach.

    Args:
        instance: MTVRP instance.

    Returns:
        MTVRPSolution.
    """
    unvisited = set(range(1, instance.n + 1))
    routes = []

    while unvisited:
        route = []
        load = 0.0
        current = 0

        while unvisited:
            best = None
            best_dist = float("inf")
            for c in unvisited:
                if load + instance.demands[c - 1] > instance.vehicle_capacity + 1e-6:
                    continue
                d = instance.dist(current, c)
                if d < best_dist:
                    best_dist = d
                    best = c
            if best is None:
                break
            route.append(best)
            load += instance.demands[best - 1]
            current = best
            unvisited.remove(best)

        if route:
            routes.append(route)

    # Assign routes to vehicles (round-robin)
    vehicle_trips = [[] for _ in range(instance.num_vehicles)]
    for idx, route in enumerate(routes):
        v = idx % instance.num_vehicles
        vehicle_trips[v].append(route)

    total_dist = sum(instance.route_distance(r) for trips in vehicle_trips
                     for r in trips)
    return MTVRPSolution(vehicle_trips=vehicle_trips, total_distance=total_dist)


if __name__ == "__main__":
    from instance import small_mtvrp_8

    inst = small_mtvrp_8()
    sol = greedy_multi_trip(inst)
    print(f"Greedy: {sol}")
