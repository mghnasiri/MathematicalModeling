"""
Simulated Annealing for VRP with Backhauls.

Problem: VRPB

Relocate/swap moves that maintain linehaul-before-backhaul precedence
within each route.

Warm-started with nearest neighbor VRPB.

References:
    Toth, P. & Vigo, D. (1999). A heuristic algorithm for the
    VRPB. EJOR, 113(3), 528-543.
    https://doi.org/10.1016/S0377-2217(98)00022-8
"""

from __future__ import annotations

import sys
import os
import math
import time
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


_inst = _load_mod("vrpb_instance_meta", os.path.join(_this_dir, "instance.py"))
VRPBInstance = _inst.VRPBInstance
VRPBSolution = _inst.VRPBSolution

_heur = _load_mod("vrpb_heur_meta", os.path.join(_this_dir, "heuristics.py"))
nearest_neighbor_vrpb = _heur.nearest_neighbor_vrpb


def _route_feasible(instance: VRPBInstance, route: list[int]) -> bool:
    """Check capacity and precedence feasibility."""
    if not instance.route_precedence_feasible(route):
        return False
    lh_demand = sum(instance.demands[c] for c in route if instance.is_linehaul(c))
    bh_demand = sum(instance.demands[c] for c in route if instance.is_backhaul(c))
    return (lh_demand <= instance.capacity + 1e-10 and
            bh_demand <= instance.capacity + 1e-10)


def simulated_annealing(
    instance: VRPBInstance,
    max_iterations: int = 30000,
    initial_temp: float | None = None,
    cooling_rate: float = 0.9995,
    time_limit: float | None = None,
    seed: int | None = None,
) -> VRPBSolution:
    """Solve VRPB using SA."""
    rng = np.random.default_rng(seed)
    start_time = time.time()

    init_sol = nearest_neighbor_vrpb(instance)
    routes = [r[:] for r in init_sol.routes]
    current_dist = init_sol.total_distance

    best_routes = [r[:] for r in routes]
    best_dist = current_dist

    if initial_temp is None:
        initial_temp = current_dist * 0.1

    temp = initial_temp

    for iteration in range(max_iterations):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        new_routes = [r[:] for r in routes]
        non_empty = [k for k in range(len(new_routes)) if new_routes[k]]
        if len(non_empty) < 1:
            break

        move = rng.integers(0, 3)

        if move == 0 and len(non_empty) >= 1:
            # Intra-route swap (within LH or within BH segment)
            k = rng.choice(non_empty)
            if len(new_routes[k]) >= 2:
                i, j = rng.choice(len(new_routes[k]), 2, replace=False)
                new_routes[k][i], new_routes[k][j] = new_routes[k][j], new_routes[k][i]
                if not _route_feasible(instance, new_routes[k]):
                    new_routes[k][i], new_routes[k][j] = new_routes[k][j], new_routes[k][i]
                    temp *= cooling_rate
                    continue

        elif move == 1 and len(non_empty) >= 2:
            # Inter-route relocate
            k1, k2 = rng.choice(non_empty, 2, replace=False)
            if not new_routes[k1]:
                temp *= cooling_rate
                continue
            idx = rng.integers(0, len(new_routes[k1]))
            cust = new_routes[k1].pop(idx)

            # Insert in correct segment of target route
            if instance.is_linehaul(cust):
                # Insert among linehaul (before backhaul)
                bh_start = len(new_routes[k2])
                for i, c in enumerate(new_routes[k2]):
                    if instance.is_backhaul(c):
                        bh_start = i
                        break
                pos = rng.integers(0, bh_start + 1)
            else:
                # Insert among backhaul (after linehaul)
                bh_start = len(new_routes[k2])
                for i, c in enumerate(new_routes[k2]):
                    if instance.is_backhaul(c):
                        bh_start = i
                        break
                pos = rng.integers(bh_start, len(new_routes[k2]) + 1)

            new_routes[k2].insert(pos, cust)

            if not _route_feasible(instance, new_routes[k2]):
                new_routes[k2].remove(cust)
                new_routes[k1].insert(idx, cust)
                temp *= cooling_rate
                continue

        elif move == 2 and len(non_empty) >= 2:
            # Inter-route swap (same type customers)
            k1, k2 = rng.choice(non_empty, 2, replace=False)
            if not new_routes[k1] or not new_routes[k2]:
                temp *= cooling_rate
                continue
            i1 = rng.integers(0, len(new_routes[k1]))
            i2 = rng.integers(0, len(new_routes[k2]))
            c1, c2 = new_routes[k1][i1], new_routes[k2][i2]

            new_routes[k1][i1], new_routes[k2][i2] = c2, c1
            if not (_route_feasible(instance, new_routes[k1]) and
                    _route_feasible(instance, new_routes[k2])):
                new_routes[k1][i1], new_routes[k2][i2] = c1, c2
                temp *= cooling_rate
                continue
        else:
            temp *= cooling_rate
            continue

        new_routes = [r for r in new_routes if r]
        new_dist = sum(instance.route_distance(r) for r in new_routes)

        delta = new_dist - current_dist
        if delta < 0 or (temp > 0 and rng.random() < math.exp(-delta / max(temp, 1e-10))):
            routes = new_routes
            current_dist = new_dist
            if current_dist < best_dist - 1e-10:
                best_dist = current_dist
                best_routes = [r[:] for r in routes]

        temp *= cooling_rate

    return VRPBSolution(routes=best_routes, total_distance=best_dist)


if __name__ == "__main__":
    inst = VRPBInstance.random(seed=42)
    nn_sol = nearest_neighbor_vrpb(inst)
    print(f"NN: dist={nn_sol.total_distance:.1f}")
    sa_sol = simulated_annealing(inst, seed=42)
    print(f"SA: dist={sa_sol.total_distance:.1f}")
