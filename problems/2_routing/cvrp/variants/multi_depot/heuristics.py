"""
Constructive Heuristic for Multi-Depot VRP.

Problem: MDVRP
Complexity: O(n^2 * D)

Nearest-depot assignment: assign each customer to nearest depot,
then build routes per depot using nearest neighbor.

References:
    Cordeau, J.-F., Gendreau, M. & Laporte, G. (1997). A tabu search
    for MDVRP. Networks, 30(2), 105-119.
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


_inst = _load_mod("mdvrp_instance_h", os.path.join(_this_dir, "instance.py"))
MDVRPInstance = _inst.MDVRPInstance
MDVRPSolution = _inst.MDVRPSolution


def nearest_depot_nn(instance: MDVRPInstance) -> MDVRPSolution:
    """Assign customers to nearest depot, then build routes with NN."""
    D = instance.num_depots
    n = instance.n
    dist = instance.distance_matrix

    # Assign each customer to nearest depot
    depot_customers: list[list[int]] = [[] for _ in range(D)]
    for c in range(n):
        node = instance.customer_idx(c)
        nearest_d = int(np.argmin([dist[d][node] for d in range(D)]))
        depot_customers[nearest_d].append(node)

    depot_routes: list[list[list[int]]] = [[] for _ in range(D)]

    for d in range(D):
        remaining = set(depot_customers[d])
        while remaining:
            route: list[int] = []
            load = 0.0
            current = d

            while remaining:
                best_c = -1
                best_d_val = float("inf")
                for c in remaining:
                    demand = instance.demands[c - D]
                    if load + demand > instance.capacity + 1e-10:
                        continue
                    if dist[current][c] < best_d_val:
                        best_d_val = dist[current][c]
                        best_c = c
                if best_c < 0:
                    break
                route.append(best_c)
                remaining.remove(best_c)
                load += instance.demands[best_c - D]
                current = best_c

            if route:
                depot_routes[d].append(route)

    total_dist = sum(
        instance.route_distance(d, r)
        for d in range(D) for r in depot_routes[d]
    )
    return MDVRPSolution(depot_routes=depot_routes, total_distance=total_dist)


if __name__ == "__main__":
    inst = _inst.small_mdvrp_2_6()
    sol = nearest_depot_nn(inst)
    print(f"NN: dist={sol.total_distance:.1f}")
