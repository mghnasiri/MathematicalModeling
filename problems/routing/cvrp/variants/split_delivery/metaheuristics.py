"""
Split Delivery VRP (SDVRP) — Metaheuristics.

Algorithms:
    - Simulated Annealing with route-level moves (relocate, swap, resplit).

References:
    Archetti, C., Savelsbergh, M.W.P. & Speranza, M.G. (2006). Worst-case
    analysis for split delivery vehicle routing problems. Transportation
    Science, 40(2), 226-234. https://doi.org/10.1287/trsc.1050.0117
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


_inst = _load_mod("sdvrp_instance_m", os.path.join(_this_dir, "instance.py"))
SDVRPInstance = _inst.SDVRPInstance
SDVRPSolution = _inst.SDVRPSolution

_heur = _load_mod("sdvrp_heuristics_m", os.path.join(_this_dir, "heuristics.py"))
nearest_neighbor_split = _heur.nearest_neighbor_split


def simulated_annealing(
    instance: SDVRPInstance,
    max_iterations: int = 50000,
    cooling_rate: float = 0.9995,
    seed: int | None = None,
    time_limit: float | None = None,
) -> SDVRPSolution:
    """Simulated Annealing for SDVRP.

    Args:
        instance: SDVRP instance.
        max_iterations: Maximum iterations.
        cooling_rate: Geometric cooling factor.
        seed: Random seed.
        time_limit: Wall-clock time limit in seconds.

    Returns:
        Best SDVRPSolution found.
    """
    rng = np.random.default_rng(seed)

    current = nearest_neighbor_split(instance)
    routes = [list(r) for r in current.routes]
    cost = current.total_distance

    best_routes = [list(r) for r in routes]
    best_cost = cost

    # Auto-calibrate temperature
    temp = best_cost * 0.05
    start = time.time()

    for it in range(max_iterations):
        if time_limit and time.time() - start > time_limit:
            break

        new_routes = [list(r) for r in routes]
        move = rng.integers(0, 3)

        if move == 0 and len(new_routes) > 1:
            # Relocate: move a delivery from one route to another
            r1 = rng.integers(0, len(new_routes))
            if len(new_routes[r1]) > 0:
                idx = rng.integers(0, len(new_routes[r1]))
                cust, qty = new_routes[r1][idx]

                r2 = rng.integers(0, len(new_routes))
                while r2 == r1 and len(new_routes) > 1:
                    r2 = rng.integers(0, len(new_routes))

                load2 = sum(q for _, q in new_routes[r2])
                available = instance.capacity - load2
                if available > 1e-6:
                    transfer = min(qty, available)
                    new_routes[r2].append((cust, transfer))
                    remaining = qty - transfer
                    if remaining > 1e-6:
                        new_routes[r1][idx] = (cust, remaining)
                    else:
                        new_routes[r1].pop(idx)

        elif move == 1:
            # Intra-route swap
            r = rng.integers(0, len(new_routes))
            if len(new_routes[r]) >= 2:
                i, j = rng.choice(len(new_routes[r]), 2, replace=False)
                new_routes[r][i], new_routes[r][j] = new_routes[r][j], new_routes[r][i]

        elif move == 2 and len(new_routes) > 1:
            # Inter-route swap
            r1, r2 = rng.choice(len(new_routes), 2, replace=False)
            if new_routes[r1] and new_routes[r2]:
                i1 = rng.integers(0, len(new_routes[r1]))
                i2 = rng.integers(0, len(new_routes[r2]))
                new_routes[r1][i1], new_routes[r2][i2] = new_routes[r2][i2], new_routes[r1][i1]

        # Remove empty routes
        new_routes = [r for r in new_routes if r]
        if not new_routes:
            continue

        # Check capacity
        feasible = True
        for r in new_routes:
            if sum(q for _, q in r) > instance.capacity + 1e-6:
                feasible = False
                break
        if not feasible:
            continue

        # Check demand coverage
        delivered = np.zeros(instance.n)
        for r in new_routes:
            for c, q in r:
                delivered[c - 1] += q
        if not np.allclose(delivered, instance.demands, atol=1e-4):
            continue

        new_cost = sum(
            instance.route_distance([c for c, _ in r]) for r in new_routes
        )
        delta = new_cost - cost

        if delta < 0 or rng.random() < math.exp(-delta / max(temp, 1e-15)):
            routes = new_routes
            cost = new_cost
            if cost < best_cost - 1e-10:
                best_cost = cost
                best_routes = [list(r) for r in routes]

        temp *= cooling_rate

    return SDVRPSolution(routes=best_routes, total_distance=best_cost)


if __name__ == "__main__":
    from instance import small_sdvrp_6

    inst = small_sdvrp_6()
    sol = simulated_annealing(inst, max_iterations=10000, seed=42)
    print(f"SA: {sol}")
