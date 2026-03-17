"""
Electric Vehicle Routing Problem (EVRP) — Metaheuristics.

Algorithms:
    - Simulated Annealing with relocate, swap, station insertion/removal.

References:
    Schneider, M., Stenger, A. & Goeke, D. (2014). The electric vehicle
    routing problem with time windows and recharging stations. Transportation
    Science, 48(4), 500-520. https://doi.org/10.1287/trsc.2013.0490
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


_inst = _load_mod("evrp_instance_m", os.path.join(_this_dir, "instance.py"))
EVRPInstance = _inst.EVRPInstance
EVRPSolution = _inst.EVRPSolution
validate_solution = _inst.validate_solution

_heur = _load_mod("evrp_heuristics_m", os.path.join(_this_dir, "heuristics.py"))
nearest_neighbor_evrp = _heur.nearest_neighbor_evrp


def _total_distance(instance: EVRPInstance, routes: list[list[int]]) -> float:
    return sum(instance.route_distance(r) for r in routes)


def _all_routes_feasible(instance: EVRPInstance, routes: list[list[int]]) -> bool:
    for route in routes:
        feasible, _ = instance.route_feasible(route)
        if not feasible:
            return False
    return True


def simulated_annealing(
    instance: EVRPInstance,
    max_iterations: int = 50000,
    cooling_rate: float = 0.9995,
    seed: int | None = None,
    time_limit: float | None = None,
) -> EVRPSolution:
    """Simulated Annealing for EVRP.

    Neighborhoods: intra-route relocate, inter-route relocate,
    inter-route swap, station insertion/removal.

    Args:
        instance: EVRP instance.
        max_iterations: Maximum iterations.
        cooling_rate: Temperature decay factor.
        seed: Random seed.
        time_limit: Time limit in seconds.

    Returns:
        EVRPSolution.
    """
    rng = np.random.default_rng(seed)

    init = nearest_neighbor_evrp(instance)
    routes = [list(r) for r in init.routes]
    cost = _total_distance(instance, routes)

    best_routes = [list(r) for r in routes]
    best_cost = cost

    temp = best_cost * 0.1
    start = time.time()

    stations = set(instance.station_nodes)

    for it in range(max_iterations):
        if time_limit and time.time() - start > time_limit:
            break

        new_routes = [list(r) for r in routes]
        move = rng.integers(0, 4)

        if move == 0 and len(new_routes) > 0:
            # Intra-route relocate (customer only)
            ri = int(rng.integers(0, len(new_routes)))
            route = new_routes[ri]
            customers = [k for k, node in enumerate(route) if node not in stations]
            if len(customers) >= 2:
                idx = customers[int(rng.integers(0, len(customers)))]
                node = route.pop(idx)
                pos = int(rng.integers(0, len(route) + 1))
                route.insert(pos, node)

        elif move == 1 and len(new_routes) >= 2:
            # Inter-route relocate
            ri = int(rng.integers(0, len(new_routes)))
            rj = int(rng.integers(0, len(new_routes) - 1))
            if rj >= ri:
                rj += 1
            r1 = new_routes[ri]
            r2 = new_routes[rj]
            customers_r1 = [k for k, node in enumerate(r1) if node not in stations]
            if customers_r1:
                idx = customers_r1[int(rng.integers(0, len(customers_r1)))]
                node = r1.pop(idx)
                pos = int(rng.integers(0, len(r2) + 1))
                r2.insert(pos, node)
                if not r1 or all(n in stations for n in r1):
                    new_routes[ri] = []

        elif move == 2 and len(new_routes) >= 2:
            # Inter-route swap
            ri = int(rng.integers(0, len(new_routes)))
            rj = int(rng.integers(0, len(new_routes) - 1))
            if rj >= ri:
                rj += 1
            r1 = new_routes[ri]
            r2 = new_routes[rj]
            c1 = [k for k, node in enumerate(r1) if node not in stations]
            c2 = [k for k, node in enumerate(r2) if node not in stations]
            if c1 and c2:
                i1 = c1[int(rng.integers(0, len(c1)))]
                i2 = c2[int(rng.integers(0, len(c2)))]
                r1[i1], r2[i2] = r2[i2], r1[i1]

        elif move == 3 and len(new_routes) > 0:
            # Station insertion/removal
            ri = int(rng.integers(0, len(new_routes)))
            route = new_routes[ri]
            station_positions = [k for k, node in enumerate(route)
                                 if node in stations]
            if station_positions and rng.random() < 0.5:
                # Remove a station
                idx = station_positions[int(rng.integers(0, len(station_positions)))]
                route.pop(idx)
            else:
                # Insert a random station
                s = instance.station_nodes[int(rng.integers(0, len(instance.station_nodes)))]
                pos = int(rng.integers(0, len(route) + 1))
                route.insert(pos, s)

        # Remove empty routes
        new_routes = [r for r in new_routes if r and
                      any(node not in stations for node in r)]

        if not new_routes:
            continue

        if not _all_routes_feasible(instance, new_routes):
            temp *= cooling_rate
            continue

        # Check all customers still visited
        visited = set()
        for r in new_routes:
            for node in r:
                if 1 <= node <= instance.n:
                    visited.add(node)
        if len(visited) != instance.n:
            temp *= cooling_rate
            continue

        new_cost = _total_distance(instance, new_routes)
        delta = new_cost - cost

        if delta < 0 or rng.random() < math.exp(-delta / max(temp, 1e-15)):
            routes = new_routes
            cost = new_cost
            if cost < best_cost:
                best_cost = cost
                best_routes = [list(r) for r in routes]

        temp *= cooling_rate

    return EVRPSolution(routes=best_routes, total_distance=best_cost)


if __name__ == "__main__":
    from instance import small_evrp_6

    inst = small_evrp_6()
    sol = simulated_annealing(inst, max_iterations=10000, seed=42)
    print(f"SA EVRP: {sol}")
