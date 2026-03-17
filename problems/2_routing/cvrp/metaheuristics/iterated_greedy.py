"""
Iterated Greedy for the Capacitated Vehicle Routing Problem (CVRP).

Iteratively removes d random customers from their routes and reinserts
them at the cheapest feasible positions. Uses Boltzmann acceptance.

Warm-started with Clarke-Wright savings heuristic.

Complexity: O(iterations * d * n * K) where K = number of routes.

References:
    Ruiz, R. & Stützle, T. (2007). A simple and effective iterated
    greedy algorithm for the permutation flowshop scheduling problem.
    European Journal of Operational Research, 177(3), 2033-2049.
    https://doi.org/10.1016/j.ejor.2005.12.009

    Ropke, S. & Pisinger, D. (2006). An adaptive large neighborhood
    search heuristic for the pickup and delivery problem with time
    windows. Transportation Science, 40(4), 455-472.
    https://doi.org/10.1287/trsc.1050.0135
"""

from __future__ import annotations

import sys
import os
import math
import time
import importlib.util

import numpy as np

_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_mod(name, filepath):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_inst = _load_mod("cvrp_instance_ig", os.path.join(_parent_dir, "instance.py"))
CVRPInstance = _inst.CVRPInstance
CVRPSolution = _inst.CVRPSolution
validate_solution = _inst.validate_solution

_cw = _load_mod(
    "cvrp_cw_ig",
    os.path.join(_parent_dir, "heuristics", "clarke_wright.py"),
)
clarke_wright_savings = _cw.clarke_wright_savings


def iterated_greedy(
    instance: CVRPInstance,
    max_iterations: int = 3000,
    d: int | None = None,
    temperature_factor: float = 0.1,
    time_limit: float | None = None,
    seed: int | None = None,
) -> CVRPSolution:
    """Solve CVRP using Iterated Greedy.

    Args:
        instance: A CVRPInstance.
        max_iterations: Maximum number of iterations.
        d: Number of customers to remove. Default: max(2, n//5).
        temperature_factor: Controls acceptance probability.
        time_limit: Maximum wall-clock time in seconds.
        seed: Random seed for reproducibility.

    Returns:
        CVRPSolution with the best routes found.
    """
    rng = np.random.default_rng(seed)
    n = instance.n
    start_time = time.time()

    if d is None:
        d = max(2, n // 5)
    d = min(d, n)

    dist = instance.distance_matrix

    # Warm-start
    init_sol = clarke_wright_savings(instance)
    routes = [list(r) for r in init_sol.routes if r]
    current_dist = instance.total_distance(routes)

    best_routes = [r[:] for r in routes]
    best_dist = current_dist

    # Temperature
    avg_edge = current_dist / max(n, 1)
    temperature = temperature_factor * avg_edge

    for iteration in range(max_iterations):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        # Flatten routes to get all customers
        all_customers = [c for r in routes for c in r]
        if len(all_customers) < d:
            continue

        # Destruction: remove d random customers
        to_remove = set(rng.choice(all_customers, size=d, replace=False).tolist())
        new_routes = []
        for r in routes:
            new_r = [c for c in r if c not in to_remove]
            if new_r:
                new_routes.append(new_r)

        # Reconstruction: greedily insert each removed customer at cheapest position
        removed_list = list(to_remove)
        rng.shuffle(removed_list)

        for cust in removed_list:
            best_cost = float("inf")
            best_route_idx = -1
            best_pos = 0

            for ri, route in enumerate(new_routes):
                route_demand = sum(instance.demands[c - 1] for c in route)
                if route_demand + instance.demands[cust - 1] > instance.capacity + 1e-10:
                    continue

                for pos in range(len(route) + 1):
                    if pos == 0:
                        prev = 0
                        nxt = route[0] if route else 0
                    elif pos == len(route):
                        prev = route[-1]
                        nxt = 0
                    else:
                        prev = route[pos - 1]
                        nxt = route[pos]

                    cost = dist[prev][cust] + dist[cust][nxt] - dist[prev][nxt]
                    if cost < best_cost:
                        best_cost = cost
                        best_route_idx = ri
                        best_pos = pos

            # Also try creating a new route
            new_route_cost = dist[0][cust] + dist[cust][0]
            if new_route_cost < best_cost:
                new_routes.append([cust])
            elif best_route_idx >= 0:
                new_routes[best_route_idx].insert(best_pos, cust)
            else:
                new_routes.append([cust])

        new_dist = instance.total_distance(new_routes)

        # Acceptance
        delta = new_dist - current_dist
        if delta < 0 or (temperature > 0 and
                         rng.random() < math.exp(-delta / max(temperature, 1e-10))):
            routes = new_routes
            current_dist = new_dist

            if current_dist < best_dist:
                best_dist = current_dist
                best_routes = [r[:] for r in routes]

    return CVRPSolution(
        routes=best_routes,
        distance=best_dist,
    )


if __name__ == "__main__":
    inst = CVRPInstance.random(n=15, seed=42)
    print(f"CVRP: {inst.n} customers, capacity={inst.capacity}")

    cw_sol = clarke_wright_savings(inst)
    print(f"CW: distance={cw_sol.distance:.2f}, vehicles={len(cw_sol.routes)}")

    ig_sol = iterated_greedy(inst, seed=42)
    print(f"IG: distance={ig_sol.distance:.2f}, vehicles={len(ig_sol.routes)}")
