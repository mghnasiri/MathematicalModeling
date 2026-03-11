"""
Cumulative VRP — Heuristics.

Algorithms:
    - Nearest neighbor (latency-aware).
    - Nearest-first insertion.

References:
    Ngueveu, S.U. et al. (2010). An effective memetic algorithm for the
    cumulative capacitated vehicle routing problem.
    https://doi.org/10.1016/j.cor.2009.06.014
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


_inst = _load_mod("cumvrp_instance_h", os.path.join(_this_dir, "instance.py"))
CumVRPInstance = _inst.CumVRPInstance
CumVRPSolution = _inst.CumVRPSolution


def nearest_neighbor_cum(instance: CumVRPInstance) -> CumVRPSolution:
    """Nearest neighbor heuristic prioritizing nearby customers.

    Args:
        instance: CumVRPInstance.

    Returns:
        CumVRPSolution.
    """
    n = instance.n
    unvisited = set(range(1, n + 1))
    routes: list[list[int]] = []

    while unvisited:
        route: list[int] = []
        load = 0.0
        current = 0

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
            current = best

        if route:
            routes.append(route)

    total_lat = sum(instance.route_latency(r) for r in routes)
    return CumVRPSolution(routes=routes, total_latency=total_lat)


if __name__ == "__main__":
    from instance import small_cumvrp_6

    inst = small_cumvrp_6()
    sol = nearest_neighbor_cum(inst)
    print(f"NN: {sol}")
