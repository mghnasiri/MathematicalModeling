"""
Periodic VRP — Heuristics.

Algorithms:
    - Spread-then-route: spread visits evenly across days, then NN per day.

References:
    Christofides, N. & Beasley, J.E. (1984). The period routing problem.
    Networks, 14(2), 237-256.
    https://doi.org/10.1002/net.3230140205
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


_inst = _load_mod("pvrp_instance_h", os.path.join(_this_dir, "instance.py"))
PVRPInstance = _inst.PVRPInstance
PVRPSolution = _inst.PVRPSolution


def spread_then_route(instance: PVRPInstance) -> PVRPSolution:
    """Assign visit days evenly, then build NN routes per day.

    Args:
        instance: PVRPInstance.

    Returns:
        PVRPSolution.
    """
    n = instance.n
    T = instance.num_periods

    # Assign visit days: spread evenly
    day_customers: list[list[int]] = [[] for _ in range(T)]
    for c in range(1, n + 1):
        freq = int(instance.visit_freq[c - 1])
        # Spread visits evenly across days
        step = T / freq if freq > 0 else T
        for v in range(freq):
            day = int(v * step) % T
            day_customers[day].append(c)

    # Build NN routes for each day
    day_routes: list[list[list[int]]] = []
    total_dist = 0.0

    for t in range(T):
        unvisited = set(day_customers[t])
        routes: list[list[int]] = []

        while unvisited:
            route: list[int] = []
            load = 0.0
            current = 0

            while unvisited:
                best = -1
                best_d = float("inf")
                for c in unvisited:
                    if load + instance.demands[c - 1] > instance.capacity + 1e-6:
                        continue
                    d = instance.distance(current, c)
                    if d < best_d:
                        best_d = d
                        best = c
                if best < 0:
                    break
                route.append(best)
                unvisited.remove(best)
                load += instance.demands[best - 1]
                current = best

            if route:
                total_dist += instance.route_distance(route)
                routes.append(route)

        day_routes.append(routes)

    return PVRPSolution(day_routes=day_routes, total_distance=total_dist)


if __name__ == "__main__":
    from instance import small_pvrp_6

    inst = small_pvrp_6()
    sol = spread_then_route(inst)
    print(f"Spread-then-route: {sol}")
