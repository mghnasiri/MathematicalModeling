"""
VRPTW with Soft Time Windows — Heuristics.

Algorithms:
    - Nearest neighbor with TW awareness.
    - Sequential insertion.

References:
    Taillard, E. et al. (1997). A tabu search heuristic for the vehicle
    routing problem with soft time windows. Transportation Science, 31(2).
    https://doi.org/10.1287/trsc.31.2.170
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


_inst = _load_mod("softtw_instance_h", os.path.join(_this_dir, "instance.py"))
SoftTWInstance = _inst.SoftTWInstance
SoftTWSolution = _inst.SoftTWSolution


def nearest_neighbor_stw(instance: SoftTWInstance) -> SoftTWSolution:
    """Nearest neighbor heuristic for soft TW VRP.

    Builds routes greedily, choosing the nearest feasible customer
    (capacity-wise). Time window violations are allowed but penalized.

    Args:
        instance: SoftTWInstance.

    Returns:
        SoftTWSolution.
    """
    n = instance.n
    unvisited = set(range(1, n + 1))
    routes: list[list[int]] = []

    while unvisited:
        route: list[int] = []
        load = 0.0
        current = 0
        time = 0.0

        while unvisited:
            best = -1
            best_dist = float("inf")
            for c in unvisited:
                if load + instance.demands[c - 1] > instance.capacity + 1e-6:
                    continue
                d = instance.distance(current, c)
                if d < best_dist:
                    best_dist = d
                    best = c

            if best < 0:
                break

            route.append(best)
            unvisited.remove(best)
            load += instance.demands[best - 1]
            time += instance.distance(current, best)
            e = instance.time_windows[best][0]
            if time < e:
                time = e
            time += instance.service_times[best]
            current = best

        if route:
            routes.append(route)

    total_dist = 0.0
    total_pen = 0.0
    for route in routes:
        d, p, _ = instance.route_cost(route)
        total_dist += d
        total_pen += p

    return SoftTWSolution(
        routes=routes, total_distance=total_dist,
        total_penalty=total_pen, total_cost=total_dist + total_pen
    )


if __name__ == "__main__":
    from instance import small_softtw_6

    inst = small_softtw_6()
    sol = nearest_neighbor_stw(inst)
    print(f"NN: {sol}")
