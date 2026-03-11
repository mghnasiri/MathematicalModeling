"""
Electric Vehicle Routing Problem (EVRP) — Heuristics.

Algorithms:
    - Nearest neighbor with energy-aware station insertion.
    - Greedy savings adapted for EVRP.

References:
    Erdogan, S. & Miller-Hooks, E. (2012). A green vehicle routing problem.
    Transportation Research Part E, 48(1), 100-114.
    https://doi.org/10.1016/j.tre.2011.08.001
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


_inst = _load_mod("evrp_instance_h", os.path.join(_this_dir, "instance.py"))
EVRPInstance = _inst.EVRPInstance
EVRPSolution = _inst.EVRPSolution


def _find_nearest_station(instance: EVRPInstance, current: int) -> int | None:
    """Find nearest reachable charging station from current node."""
    best_station = None
    best_dist = float("inf")
    for s in instance.station_nodes:
        d = instance.dist(current, s)
        if instance.energy_cost(current, s) <= instance.battery_capacity and d < best_dist:
            best_dist = d
            best_station = s
    return best_station


def nearest_neighbor_evrp(instance: EVRPInstance) -> EVRPSolution:
    """Nearest neighbor heuristic with energy-aware station insertion.

    Builds routes greedily. When battery is low, inserts a station visit
    before the next customer. Starts a new route when capacity is full
    or no reachable customer exists.

    Args:
        instance: EVRP instance.

    Returns:
        EVRPSolution.
    """
    unvisited = set(range(1, instance.n + 1))
    routes = []

    while unvisited:
        route = []
        load = 0.0
        battery = instance.battery_capacity
        current = 0  # depot
        added_any = False

        for _ in range(instance.n * 3):  # safety bound
            if not unvisited:
                break

            # Find nearest customer reachable by energy and capacity
            candidates = []
            for c in unvisited:
                if instance.demands[c - 1] + load > instance.vehicle_capacity + 1e-6:
                    continue
                energy_needed = instance.energy_cost(current, c)
                if energy_needed <= battery + 1e-6:
                    # Also check we can get back to depot (directly or via station)
                    remaining = battery - energy_needed
                    can_return = instance.energy_cost(c, 0) <= remaining + 1e-6
                    if not can_return:
                        # Check if any station reachable from c
                        for s in instance.station_nodes:
                            if instance.energy_cost(c, s) <= remaining + 1e-6:
                                can_return = True
                                break
                    if can_return:
                        candidates.append((instance.dist(current, c), c))

            if not candidates:
                # Try recharging first
                if not added_any or battery < instance.battery_capacity * 0.5:
                    station = _find_nearest_station(instance, current)
                    if (station is not None and
                            instance.energy_cost(current, station) <= battery + 1e-6 and
                            battery < instance.battery_capacity - 1e-6):
                        route.append(station)
                        battery = instance.battery_capacity
                        current = station
                        continue
                break

            # Pick nearest
            candidates.sort()
            best_cust = candidates[0][1]
            battery -= instance.energy_cost(current, best_cust)
            load += instance.demands[best_cust - 1]
            route.append(best_cust)
            current = best_cust
            unvisited.remove(best_cust)
            added_any = True

        # Finalize route
        if route and any(1 <= node <= instance.n for node in route):
            # Insert station before returning if needed
            energy_back = instance.energy_cost(current, 0)
            if battery < energy_back - 1e-6:
                station = _find_nearest_station(instance, current)
                if station and instance.energy_cost(current, station) <= battery + 1e-6:
                    route.append(station)
            routes.append(route)
        elif unvisited:
            # Force-assign closest unvisited customer
            closest = min(unvisited, key=lambda c: instance.dist(0, c))
            routes.append([closest])
            unvisited.remove(closest)

    total_dist = sum(instance.route_distance(r) for r in routes)
    return EVRPSolution(routes=routes, total_distance=total_dist)


if __name__ == "__main__":
    from instance import small_evrp_6

    inst = small_evrp_6()
    sol = nearest_neighbor_evrp(inst)
    print(f"NN EVRP: {sol}")
    for i, r in enumerate(sol.routes):
        print(f"  Route {i}: {r}")
