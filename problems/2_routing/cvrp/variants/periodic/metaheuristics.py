"""
Periodic VRP — Metaheuristics.

Algorithms:
    - SA with day-swap, intra-day relocate/2-opt.

References:
    Cordeau, J.F., Gendreau, M. & Laporte, G. (1997). A tabu search
    heuristic for periodic and multi-depot vehicle routing problems.
    Networks, 30(2), 105-119.
    https://doi.org/10.1002/(SICI)1097-0037(199709)30:2<105::AID-NET5>3.0.CO;2-G
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


_inst = _load_mod("pvrp_instance_m", os.path.join(_this_dir, "instance.py"))
PVRPInstance = _inst.PVRPInstance
PVRPSolution = _inst.PVRPSolution

_heur = _load_mod("pvrp_heuristics_m", os.path.join(_this_dir, "heuristics.py"))
spread_then_route = _heur.spread_then_route


def _total_distance(instance, day_routes):
    return sum(
        instance.route_distance(r)
        for routes in day_routes for r in routes
    )


def _check_capacity(instance, route):
    return sum(instance.demands[c - 1] for c in route) <= instance.capacity + 1e-6


def _visit_counts(instance, day_routes):
    counts = np.zeros(instance.n, dtype=int)
    for routes in day_routes:
        for route in routes:
            for c in route:
                counts[c - 1] += 1
    return counts


def simulated_annealing(
    instance: PVRPInstance,
    max_iterations: int = 50000,
    cooling_rate: float = 0.9995,
    seed: int | None = None,
    time_limit: float | None = None,
) -> PVRPSolution:
    """SA for Periodic VRP."""
    rng = np.random.default_rng(seed)
    T = instance.num_periods

    init = spread_then_route(instance)
    day_routes = [[list(r) for r in routes] for routes in init.day_routes]
    cost = init.total_distance

    best_dr = [[list(r) for r in routes] for routes in day_routes]
    best_cost = cost

    temp = best_cost * 0.05 if best_cost > 0 else 10.0
    start = time.time()

    for it in range(max_iterations):
        if time_limit and time.time() - start > time_limit:
            break

        new_dr = [[list(r) for r in routes] for routes in day_routes]
        move = rng.integers(0, 2)

        if move == 0:
            # Intra-day: relocate customer between routes
            t = rng.integers(0, T)
            if len(new_dr[t]) >= 1:
                r1 = rng.integers(0, len(new_dr[t]))
                if new_dr[t][r1]:
                    idx = rng.integers(0, len(new_dr[t][r1]))
                    cust = new_dr[t][r1].pop(idx)
                    r2 = rng.integers(0, len(new_dr[t]))
                    pos = rng.integers(0, len(new_dr[t][r2]) + 1)
                    new_dr[t][r2].insert(pos, cust)

        elif move == 1:
            # Intra-day: 2-opt on a route
            t = rng.integers(0, T)
            if new_dr[t]:
                r = rng.integers(0, len(new_dr[t]))
                if len(new_dr[t][r]) >= 3:
                    i = rng.integers(0, len(new_dr[t][r]) - 1)
                    j = rng.integers(i + 1, len(new_dr[t][r]))
                    new_dr[t][r][i:j + 1] = new_dr[t][r][i:j + 1][::-1]

        # Remove empty routes
        for t in range(T):
            new_dr[t] = [r for r in new_dr[t] if r]

        # Check capacity
        feasible = all(
            _check_capacity(instance, r)
            for routes in new_dr for r in routes
        )
        if not feasible:
            continue

        # Check visit frequencies
        counts = _visit_counts(instance, new_dr)
        if not np.array_equal(counts, instance.visit_freq):
            continue

        new_cost = _total_distance(instance, new_dr)
        delta = new_cost - cost

        if delta < 0 or rng.random() < math.exp(-delta / max(temp, 1e-15)):
            day_routes = new_dr
            cost = new_cost
            if cost < best_cost - 1e-10:
                best_cost = cost
                best_dr = [[list(r) for r in routes] for routes in day_routes]

        temp *= cooling_rate

    return PVRPSolution(day_routes=best_dr, total_distance=best_cost)


if __name__ == "__main__":
    from instance import small_pvrp_6

    inst = small_pvrp_6()
    sol = simulated_annealing(inst, max_iterations=10000, seed=42)
    print(f"SA: {sol}")
