"""
Multi-Trip VRP — Metaheuristics.

Algorithms:
    - Simulated Annealing with trip reassignment and intra-trip moves.

References:
    Olivera, A. & Viera, O. (2007). Adaptive memory programming for the
    vehicle routing problem with multiple trips. Computers & Operations
    Research, 34(1), 28-47. https://doi.org/10.1016/j.cor.2005.02.044
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


_inst = _load_mod("mtvrp_instance_m", os.path.join(_this_dir, "instance.py"))
MTVRPInstance = _inst.MTVRPInstance
MTVRPSolution = _inst.MTVRPSolution

_heur = _load_mod("mtvrp_heuristics_m", os.path.join(_this_dir, "heuristics.py"))
greedy_multi_trip = _heur.greedy_multi_trip


def _flatten_routes(vehicle_trips):
    """Get list of (vehicle, trip_idx, route) tuples."""
    result = []
    for v, trips in enumerate(vehicle_trips):
        for t, route in enumerate(trips):
            result.append((v, t, route))
    return result


def simulated_annealing(
    instance: MTVRPInstance,
    max_iterations: int = 50000,
    cooling_rate: float = 0.9995,
    seed: int | None = None,
    time_limit: float | None = None,
) -> MTVRPSolution:
    """SA for Multi-Trip VRP.

    Moves: relocate customer between trips, swap customers, 2-opt within trip.

    Args:
        instance: MTVRP instance.
        max_iterations: Maximum iterations.
        cooling_rate: Temperature decay factor.
        seed: Random seed.
        time_limit: Time limit in seconds.

    Returns:
        MTVRPSolution.
    """
    rng = np.random.default_rng(seed)

    init = greedy_multi_trip(instance)
    vt = [[list(r) for r in trips] for trips in init.vehicle_trips]
    cost = init.total_distance

    best_vt = [[list(r) for r in trips] for trips in vt]
    best_cost = cost

    temp = best_cost * 0.1
    start = time.time()

    for it in range(max_iterations):
        if time_limit and time.time() - start > time_limit:
            break

        new_vt = [[list(r) for r in trips] for trips in vt]
        all_trips = _flatten_routes(new_vt)
        non_empty = [(v, t, r) for v, t, r in all_trips if r]

        if not non_empty:
            break

        move = rng.integers(0, 3)

        if move == 0 and len(non_empty) >= 1:
            # Relocate customer between trips
            src_idx = int(rng.integers(0, len(non_empty)))
            v1, t1, r1 = non_empty[src_idx]
            if not r1:
                temp *= cooling_rate
                continue
            c_idx = int(rng.integers(0, len(r1)))
            cust = r1[c_idx]

            # Pick destination trip (or create new)
            dst_v = int(rng.integers(0, instance.num_vehicles))
            if not new_vt[dst_v]:
                new_vt[dst_v].append([])
            dst_t = int(rng.integers(0, len(new_vt[dst_v])))

            new_vt[v1][t1].remove(cust)
            pos = int(rng.integers(0, len(new_vt[dst_v][dst_t]) + 1))
            new_vt[dst_v][dst_t].insert(pos, cust)

        elif move == 1 and len(non_empty) >= 2:
            # Swap two customers between trips
            i1 = int(rng.integers(0, len(non_empty)))
            i2 = int(rng.integers(0, len(non_empty) - 1))
            if i2 >= i1:
                i2 += 1
            v1, t1, r1 = non_empty[i1]
            v2, t2, r2 = non_empty[i2]
            if r1 and r2:
                c1 = int(rng.integers(0, len(r1)))
                c2 = int(rng.integers(0, len(r2)))
                new_vt[v1][t1][c1], new_vt[v2][t2][c2] = \
                    new_vt[v2][t2][c2], new_vt[v1][t1][c1]

        elif move == 2 and non_empty:
            # 2-opt within a trip
            idx = int(rng.integers(0, len(non_empty)))
            v, t, r = non_empty[idx]
            if len(r) >= 3:
                i = int(rng.integers(0, len(r) - 1))
                j = int(rng.integers(i + 1, len(r)))
                new_vt[v][t][i:j + 1] = reversed(new_vt[v][t][i:j + 1])

        # Remove empty trips
        for v in range(len(new_vt)):
            new_vt[v] = [r for r in new_vt[v] if r]

        # Check capacity
        feasible = True
        for trips in new_vt:
            for route in trips:
                if instance.route_load(route) > instance.vehicle_capacity + 1e-6:
                    feasible = False
                    break
            if not feasible:
                break

        if not feasible:
            temp *= cooling_rate
            continue

        new_cost = sum(instance.route_distance(r) for trips in new_vt for r in trips)
        delta = new_cost - cost

        if delta < 0 or rng.random() < math.exp(-delta / max(temp, 1e-15)):
            vt = new_vt
            cost = new_cost
            if cost < best_cost:
                best_cost = cost
                best_vt = [[list(r) for r in trips] for trips in vt]

        temp *= cooling_rate

    return MTVRPSolution(vehicle_trips=best_vt, total_distance=best_cost)


if __name__ == "__main__":
    from instance import small_mtvrp_8

    inst = small_mtvrp_8()
    sol = simulated_annealing(inst, max_iterations=10000, seed=42)
    print(f"SA: {sol}")
