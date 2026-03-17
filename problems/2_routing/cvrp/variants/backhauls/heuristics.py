"""
Constructive Heuristics for VRP with Backhauls.

Problem: VRPB
Complexity: O(n^2)

1. Cluster-first route-second: group linehaul+backhaul customers into
   routes respecting capacity, then sequence within each route.
2. Nearest neighbor with precedence: greedily add nearest customer
   (linehaul first, then backhaul).

References:
    Goetschalckx, M. & Jacobs-Blecha, C. (1989). The vehicle routing
    problem with backhauls. EJOR, 42(1), 39-51.
    https://doi.org/10.1016/0377-2217(89)90057-X
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
    """Build routes greedily: linehaul customers first, then backhaul.

    Args:
        instance: A VRPBInstance.

    Returns:
        VRPBSolution.
    """
    n = instance.n
    dist = instance.distance_matrix
    visited = [False] * (n + 1)
    routes: list[list[int]] = []

    linehaul_set = set(range(1, instance.n_linehaul + 1))
    backhaul_set = set(range(instance.n_linehaul + 1, n + 1))

    remaining_lh = set(linehaul_set)
    remaining_bh = set(backhaul_set)

    while remaining_lh or remaining_bh:
        route: list[int] = []
        lh_load = 0.0
        bh_load = 0.0
        current = 0

        # Phase 1: add linehaul customers
        while remaining_lh:
            best_c = -1
            best_d = float("inf")
            for c in remaining_lh:
                if lh_load + instance.demands[c] > instance.capacity + 1e-10:
                    continue
                if dist[current][c] < best_d:
                    best_d = dist[current][c]
                    best_c = c
            if best_c < 0:
                break
            route.append(best_c)
            remaining_lh.remove(best_c)
            lh_load += instance.demands[best_c]
            current = best_c

        # Phase 2: add backhaul customers
        while remaining_bh:
            best_c = -1
            best_d = float("inf")
            for c in remaining_bh:
                if bh_load + instance.demands[c] > instance.capacity + 1e-10:
                    continue
                if dist[current][c] < best_d:
                    best_d = dist[current][c]
                    best_c = c
            if best_c < 0:
                break
            route.append(best_c)
            remaining_bh.remove(best_c)
            bh_load += instance.demands[best_c]
            current = best_c

        if route:
            routes.append(route)
        else:
            break

    total_dist = sum(instance.route_distance(r) for r in routes)
    return VRPBSolution(routes=routes, total_distance=total_dist)


if __name__ == "__main__":
    inst = _inst.small_vrpb_4_3()
    sol = nearest_neighbor_vrpb(inst)
    print(f"NN-VRPB: dist={sol.total_distance:.1f}, routes={len(sol.routes)}")
