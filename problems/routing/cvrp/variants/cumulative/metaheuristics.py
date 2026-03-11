"""
Cumulative VRP — Metaheuristics.

Algorithms:
    - SA with relocate/swap/2-opt optimizing total latency.

References:
    Ngueveu, S.U. et al. (2010). An effective memetic algorithm for the
    cumulative capacitated vehicle routing problem.
    https://doi.org/10.1016/j.cor.2009.06.014
"""

from __future__ import annotations

import math
import sys
import os
import importlib.util
import time

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


_inst = _load_mod("cumvrp_instance_m", os.path.join(_this_dir, "instance.py"))
CumVRPInstance = _inst.CumVRPInstance
CumVRPSolution = _inst.CumVRPSolution

_heur = _load_mod("cumvrp_heuristics_m", os.path.join(_this_dir, "heuristics.py"))
nearest_neighbor_cum = _heur.nearest_neighbor_cum


def _route_load(instance, route):
    return sum(instance.demands[c - 1] for c in route)


def simulated_annealing(
    instance: CumVRPInstance,
    max_iterations: int = 50000,
    cooling_rate: float = 0.9995,
    seed: int | None = None,
    time_limit: float | None = None,
) -> CumVRPSolution:
    """SA for Cumulative VRP minimizing total latency."""
    rng = np.random.default_rng(seed)

    init = nearest_neighbor_cum(instance)
    routes = [list(r) for r in init.routes]
    cost = init.total_latency

    best_routes = [list(r) for r in routes]
    best_cost = cost

    temp = best_cost * 0.05 if best_cost > 0 else 10.0
    start = time.time()

    for it in range(max_iterations):
        if time_limit and time.time() - start > time_limit:
            break

        new_routes = [list(r) for r in routes]
        move = rng.integers(0, 3)

        if move == 0 and len(new_routes) > 1:
            # Inter-route relocate
            r1 = rng.integers(0, len(new_routes))
            if new_routes[r1]:
                idx = rng.integers(0, len(new_routes[r1]))
                cust = new_routes[r1][idx]
                r2 = rng.integers(0, len(new_routes))
                if r2 != r1:
                    new_routes[r1].pop(idx)
                    pos = rng.integers(0, len(new_routes[r2]) + 1)
                    new_routes[r2].insert(pos, cust)

        elif move == 1 and len(new_routes) > 1:
            # Inter-route swap
            r1, r2 = rng.choice(len(new_routes), 2, replace=False)
            if new_routes[r1] and new_routes[r2]:
                i1 = rng.integers(0, len(new_routes[r1]))
                i2 = rng.integers(0, len(new_routes[r2]))
                new_routes[r1][i1], new_routes[r2][i2] = (
                    new_routes[r2][i2], new_routes[r1][i1])

        elif move == 2:
            # Intra-route 2-opt
            r = rng.integers(0, len(new_routes))
            if len(new_routes[r]) >= 3:
                i = rng.integers(0, len(new_routes[r]) - 1)
                j = rng.integers(i + 1, len(new_routes[r]))
                new_routes[r][i:j + 1] = new_routes[r][i:j + 1][::-1]

        new_routes = [r for r in new_routes if r]

        feasible = all(
            _route_load(instance, r) <= instance.capacity + 1e-6
            for r in new_routes
        )
        if not feasible:
            continue

        new_cost = sum(instance.route_latency(r) for r in new_routes)
        delta = new_cost - cost

        if delta < 0 or rng.random() < math.exp(-delta / max(temp, 1e-15)):
            routes = new_routes
            cost = new_cost
            if cost < best_cost - 1e-10:
                best_cost = cost
                best_routes = [list(r) for r in routes]

        temp *= cooling_rate

    return CumVRPSolution(routes=best_routes, total_latency=best_cost)


if __name__ == "__main__":
    from instance import small_cumvrp_6

    inst = small_cumvrp_6()
    sol = simulated_annealing(inst, max_iterations=10000, seed=42)
    print(f"SA: {sol}")
