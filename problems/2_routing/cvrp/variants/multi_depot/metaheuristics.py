"""
Simulated Annealing for Multi-Depot VRP.

Problem: MDVRP

Neighborhoods: relocate customer between routes/depots, swap customers,
intra-route 2-opt.

Warm-started with nearest-depot NN.

References:
    Cordeau, J.-F., Gendreau, M. & Laporte, G. (1997). A tabu search
    for MDVRP. Networks, 30(2), 105-119.
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


_inst = _load_mod("mdvrp_instance_meta", os.path.join(_this_dir, "instance.py"))
MDVRPInstance = _inst.MDVRPInstance
MDVRPSolution = _inst.MDVRPSolution

_heur = _load_mod("mdvrp_heur_meta", os.path.join(_this_dir, "heuristics.py"))
nearest_depot_nn = _heur.nearest_depot_nn


def _total_distance(instance, depot_routes):
    return sum(
        instance.route_distance(d, r)
        for d in range(instance.num_depots) for r in depot_routes[d]
    )


def simulated_annealing(
    instance: MDVRPInstance,
    max_iterations: int = 30000,
    initial_temp: float | None = None,
    cooling_rate: float = 0.9995,
    time_limit: float | None = None,
    seed: int | None = None,
) -> MDVRPSolution:
    rng = np.random.default_rng(seed)
    D = instance.num_depots
    start_time = time.time()

    init_sol = nearest_depot_nn(instance)
    depot_routes = [[r[:] for r in rs] for rs in init_sol.depot_routes]
    current_dist = init_sol.total_distance

    best_routes = [[r[:] for r in rs] for rs in depot_routes]
    best_dist = current_dist

    if initial_temp is None:
        initial_temp = current_dist * 0.1
    temp = initial_temp

    for iteration in range(max_iterations):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        new_routes = [[r[:] for r in rs] for rs in depot_routes]

        # Flatten routes for selection
        all_routes = []
        for d in range(D):
            for k in range(len(new_routes[d])):
                if new_routes[d][k]:
                    all_routes.append((d, k))

        if not all_routes:
            break

        move = rng.integers(0, 3)

        if move == 0 and len(all_routes) >= 1:
            # Relocate
            d1, k1 = all_routes[rng.integers(0, len(all_routes))]
            if not new_routes[d1][k1]:
                temp *= cooling_rate
                continue
            idx = rng.integers(0, len(new_routes[d1][k1]))
            cust = new_routes[d1][k1].pop(idx)

            d2 = rng.integers(0, D)
            if not new_routes[d2]:
                new_routes[d2].append([cust])
            else:
                k2 = rng.integers(0, len(new_routes[d2]))
                if instance.route_demand(new_routes[d2][k2]) + instance.demands[cust - D] > instance.capacity + 1e-10:
                    new_routes[d1][k1].insert(idx, cust)
                    temp *= cooling_rate
                    continue
                pos = rng.integers(0, len(new_routes[d2][k2]) + 1)
                new_routes[d2][k2].insert(pos, cust)

        elif move == 1 and len(all_routes) >= 2:
            # Swap
            idx1, idx2 = rng.choice(len(all_routes), 2, replace=False)
            d1, k1 = all_routes[idx1]
            d2, k2 = all_routes[idx2]
            if not new_routes[d1][k1] or not new_routes[d2][k2]:
                temp *= cooling_rate
                continue
            i1 = rng.integers(0, len(new_routes[d1][k1]))
            i2 = rng.integers(0, len(new_routes[d2][k2]))
            c1, c2 = new_routes[d1][k1][i1], new_routes[d2][k2][i2]

            # Check capacity
            r1_demand = instance.route_demand(new_routes[d1][k1]) - instance.demands[c1 - D] + instance.demands[c2 - D]
            r2_demand = instance.route_demand(new_routes[d2][k2]) - instance.demands[c2 - D] + instance.demands[c1 - D]
            if r1_demand > instance.capacity + 1e-10 or r2_demand > instance.capacity + 1e-10:
                temp *= cooling_rate
                continue
            new_routes[d1][k1][i1], new_routes[d2][k2][i2] = c2, c1

        elif move == 2:
            # Intra-route 2-opt
            d1, k1 = all_routes[rng.integers(0, len(all_routes))]
            if len(new_routes[d1][k1]) >= 3:
                i = rng.integers(0, len(new_routes[d1][k1]) - 1)
                j = rng.integers(i + 1, len(new_routes[d1][k1]))
                new_routes[d1][k1][i:j + 1] = new_routes[d1][k1][i:j + 1][::-1]
            else:
                temp *= cooling_rate
                continue
        else:
            temp *= cooling_rate
            continue

        # Clean empty routes
        for d in range(D):
            new_routes[d] = [r for r in new_routes[d] if r]

        new_dist = _total_distance(instance, new_routes)
        delta = new_dist - current_dist

        if delta < 0 or (temp > 0 and rng.random() < math.exp(-delta / max(temp, 1e-10))):
            depot_routes = new_routes
            current_dist = new_dist
            if current_dist < best_dist - 1e-10:
                best_dist = current_dist
                best_routes = [[r[:] for r in rs] for rs in depot_routes]

        temp *= cooling_rate

    return MDVRPSolution(depot_routes=best_routes, total_distance=best_dist)


if __name__ == "__main__":
    inst = MDVRPInstance.random(seed=42)
    nn = nearest_depot_nn(inst)
    print(f"NN: {nn.total_distance:.1f}")
    sa = simulated_annealing(inst, seed=42)
    print(f"SA: {sa.total_distance:.1f}")
