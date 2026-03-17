"""
VRP with Backhauls — Heuristics.

Algorithms:
    - Nearest neighbor with linehaul-first constraint.
    - Cluster-first, route-second.

References:
    Toth, P. & Vigo, D. (1999). A heuristic algorithm for the symmetric and
    asymmetric vehicle routing problem with backhauls. European Journal of
    Operational Research, 113(3), 528-543.
    https://doi.org/10.1016/S0377-2217(98)00012-6
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


_inst = _load_mod("vrpb_instance_h", os.path.join(_this_dir, "instance.py"))
VRPBInstance = _inst.VRPBInstance
VRPBSolution = _inst.VRPBSolution


def nearest_neighbor_vrpb(instance: VRPBInstance) -> VRPBSolution:
    """NN heuristic: serve all linehauls first, then backhauls per route.

    Args:
        instance: VRPB instance.

    Returns:
        VRPBSolution.
    """
    unvisited_lh = set(instance.linehaul_nodes)
    unvisited_bh = set(instance.backhaul_nodes)
    routes = []

    while unvisited_lh or unvisited_bh:
        route = []
        lh_load = 0.0
        bh_load = 0.0
        current = 0

        # Phase 1: linehaul customers
        while unvisited_lh:
            best = None
            best_dist = float("inf")
            for c in unvisited_lh:
                if lh_load + instance.demands[c - 1] > instance.vehicle_capacity + 1e-6:
                    continue
                d = instance.dist(current, c)
                if d < best_dist:
                    best_dist = d
                    best = c
            if best is None:
                break
            route.append(best)
            lh_load += instance.demands[best - 1]
            current = best
            unvisited_lh.remove(best)

        # Phase 2: backhaul customers
        while unvisited_bh:
            best = None
            best_dist = float("inf")
            for c in unvisited_bh:
                if bh_load + instance.demands[c - 1] > instance.vehicle_capacity + 1e-6:
                    continue
                d = instance.dist(current, c)
                if d < best_dist:
                    best_dist = d
                    best = c
            if best is None:
                break
            route.append(best)
            bh_load += instance.demands[best - 1]
            current = best
            unvisited_bh.remove(best)

        if route:
            routes.append(route)
        else:
            # Force assign if stuck
            if unvisited_lh:
                c = min(unvisited_lh, key=lambda x: instance.dist(0, x))
                routes.append([c])
                unvisited_lh.remove(c)
            elif unvisited_bh:
                c = min(unvisited_bh, key=lambda x: instance.dist(0, x))
                routes.append([c])
                unvisited_bh.remove(c)

    total_dist = sum(instance.route_distance(r) for r in routes)
    return VRPBSolution(routes=routes, total_distance=total_dist)


if __name__ == "__main__":
    from instance import small_vrpb_5

    inst = small_vrpb_5()
    sol = nearest_neighbor_vrpb(inst)
    print(f"NN VRPB: {sol}")
