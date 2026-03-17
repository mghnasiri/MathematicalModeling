"""
Simulated Annealing for Open VRP.

Problem: OVRP

Neighborhoods:
- Relocate: move a customer from one route to another
- Swap: exchange customers between routes
- 2-opt*: cross edges between two routes

All moves check capacity. Routes don't return to depot.
Warm-started with nearest neighbor OVRP.

References:
    Li, F., Golden, B. & Wasil, E. (2007). The open vehicle routing
    problem. C&OR, 34(10), 2918-2930.
    https://doi.org/10.1016/j.cor.2005.11.018
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


_inst = _load_mod("ovrp_instance_meta", os.path.join(_this_dir, "instance.py"))
OVRPInstance = _inst.OVRPInstance
OVRPSolution = _inst.OVRPSolution

_heur = _load_mod("ovrp_heur_meta", os.path.join(_this_dir, "heuristics.py"))
nearest_neighbor_ovrp = _heur.nearest_neighbor_ovrp


def simulated_annealing(
    instance: OVRPInstance,
    max_iterations: int = 30000,
    initial_temp: float | None = None,
    cooling_rate: float = 0.9995,
    time_limit: float | None = None,
    seed: int | None = None,
) -> OVRPSolution:
    """Solve OVRP using Simulated Annealing.

    Args:
        instance: An OVRPInstance.
        max_iterations: Maximum iterations.
        initial_temp: Initial temperature.
        cooling_rate: Geometric cooling factor.
        time_limit: Time limit in seconds.
        seed: Random seed.

    Returns:
        OVRPSolution.
    """
    rng = np.random.default_rng(seed)
    n = instance.n
    start_time = time.time()

    init_sol = nearest_neighbor_ovrp(instance)
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
        move = rng.integers(0, 3)

        non_empty = [k for k in range(len(new_routes)) if new_routes[k]]
        if not non_empty:
            break

        if move == 0 and len(non_empty) >= 1:
            # Relocate: move customer to another route
            k1 = rng.choice(non_empty)
            if not new_routes[k1]:
                temp *= cooling_rate
                continue
            idx = rng.integers(0, len(new_routes[k1]))
            cust = new_routes[k1].pop(idx)

            # Find target route
            k2 = rng.integers(0, len(new_routes))
            if instance.route_demand(new_routes[k2]) + instance.demands[cust] > instance.capacity + 1e-10:
                new_routes[k1].insert(idx, cust)
                temp *= cooling_rate
                continue
            pos = rng.integers(0, len(new_routes[k2]) + 1)
            new_routes[k2].insert(pos, cust)

        elif move == 1 and len(non_empty) >= 2:
            # Swap customers between routes
            k1, k2 = rng.choice(non_empty, 2, replace=False)
            if not new_routes[k1] or not new_routes[k2]:
                temp *= cooling_rate
                continue
            i1 = rng.integers(0, len(new_routes[k1]))
            i2 = rng.integers(0, len(new_routes[k2]))

            c1, c2 = new_routes[k1][i1], new_routes[k2][i2]
            d1 = instance.route_demand(new_routes[k1]) - instance.demands[c1] + instance.demands[c2]
            d2 = instance.route_demand(new_routes[k2]) - instance.demands[c2] + instance.demands[c1]
            if d1 > instance.capacity + 1e-10 or d2 > instance.capacity + 1e-10:
                temp *= cooling_rate
                continue
            new_routes[k1][i1], new_routes[k2][i2] = c2, c1

        elif move == 2:
            # Intra-route 2-opt
            k = rng.choice(non_empty)
            if len(new_routes[k]) >= 3:
                i = rng.integers(0, len(new_routes[k]) - 1)
                j = rng.integers(i + 1, len(new_routes[k]))
                new_routes[k][i:j + 1] = new_routes[k][i:j + 1][::-1]
            else:
                temp *= cooling_rate
                continue
        else:
            temp *= cooling_rate
            continue

        # Remove empty routes
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

    return OVRPSolution(routes=best_routes, total_distance=best_dist)


if __name__ == "__main__":
    inst = OVRPInstance.random(n=10, seed=42)
    print(f"OVRP: {inst.n} customers")

    nn_sol = nearest_neighbor_ovrp(inst)
    print(f"NN: dist={nn_sol.total_distance:.1f}")

    sa_sol = simulated_annealing(inst, seed=42)
    print(f"SA: dist={sa_sol.total_distance:.1f}")
