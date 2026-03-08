"""
Simulated Annealing for CVRP.

Problem: CVRP (Capacitated Vehicle Routing Problem)

Neighborhoods:
- Relocate: move a customer from one route to another
- Swap: swap customers between two routes
- 2-opt*: exchange tails between two routes

Warm-started with Clarke-Wright savings heuristic.
Geometric cooling schedule with Boltzmann acceptance.

References:
    Osman, I.H. (1993). Metastrategy simulated annealing and tabu
    search algorithms for the vehicle routing problem. Annals of
    Operations Research, 41(4), 421-451.
    https://doi.org/10.1007/BF02023004
"""

from __future__ import annotations

import os
import math
import numpy as np

_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _load_module(name, filepath):
    import importlib.util
    import sys as _sys
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    _sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_inst = _load_module("cvrp_instance_sa", os.path.join(_parent_dir, "instance.py"))
CVRPInstance = _inst.CVRPInstance
CVRPSolution = _inst.CVRPSolution


def _copy_routes(routes: list[list[int]]) -> list[list[int]]:
    return [r[:] for r in routes]


def _total_distance(instance: CVRPInstance, routes: list[list[int]]) -> float:
    return instance.total_distance(routes)


def _route_demand(instance: CVRPInstance, route: list[int]) -> float:
    return sum(instance.demands[c - 1] for c in route)


def simulated_annealing(
    instance: CVRPInstance,
    initial_routes: list[list[int]] | None = None,
    max_iterations: int = 50_000,
    initial_temp: float | None = None,
    cooling_rate: float = 0.9995,
    seed: int | None = None,
) -> CVRPSolution:
    """Solve CVRP using simulated annealing.

    Args:
        instance: A CVRPInstance.
        initial_routes: Starting routes. If None, uses Clarke-Wright.
        max_iterations: Maximum number of iterations.
        initial_temp: Initial temperature. If None, auto-calibrated.
        cooling_rate: Geometric cooling factor.
        seed: Random seed for reproducibility.

    Returns:
        CVRPSolution with the best routes found.
    """
    rng = np.random.default_rng(seed)
    n = instance.n
    Q = instance.capacity

    # Initialize
    if initial_routes is None:
        _cw_mod = _load_module(
            "cvrp_cw_sa", os.path.join(_parent_dir, "heuristics", "clarke_wright.py"))
        init_sol = _cw_mod.clarke_wright_savings(instance)
        routes = _copy_routes(init_sol.routes)
    else:
        routes = _copy_routes(initial_routes)

    current_cost = _total_distance(instance, routes)
    best_routes = _copy_routes(routes)
    best_cost = current_cost

    # Auto-calibrate temperature
    if initial_temp is None:
        initial_temp = best_cost * 0.05

    temp = initial_temp

    for iteration in range(max_iterations):
        # Select a random neighborhood move
        move_type = rng.integers(0, 3)
        new_routes = _copy_routes(routes)

        # Filter non-empty routes
        non_empty = [i for i, r in enumerate(new_routes) if r]
        if len(non_empty) < 1:
            continue

        if move_type == 0 and len(non_empty) >= 1:
            # Relocate: move a customer to another route (or a new route)
            src_idx = rng.choice(non_empty)
            src_route = new_routes[src_idx]
            if not src_route:
                continue
            cust_pos = rng.integers(0, len(src_route))
            customer = src_route[cust_pos]
            demand = instance.demands[customer - 1]

            # Choose destination
            other_routes = [i for i in range(len(new_routes)) if i != src_idx]
            if not other_routes:
                # Create a new route
                new_routes.append([])
                dst_idx = len(new_routes) - 1
            else:
                dst_idx = rng.choice(other_routes)

            dst_route = new_routes[dst_idx]
            if _route_demand(instance, dst_route) + demand > Q + 1e-10:
                continue

            src_route.pop(cust_pos)
            ins_pos = rng.integers(0, len(dst_route) + 1)
            dst_route.insert(ins_pos, customer)

        elif move_type == 1 and len(non_empty) >= 2:
            # Swap: exchange two customers between different routes
            r1_idx, r2_idx = rng.choice(non_empty, size=2, replace=False)
            r1 = new_routes[r1_idx]
            r2 = new_routes[r2_idx]
            if not r1 or not r2:
                continue

            p1 = rng.integers(0, len(r1))
            p2 = rng.integers(0, len(r2))

            c1, c2 = r1[p1], r2[p2]
            d1, d2 = instance.demands[c1 - 1], instance.demands[c2 - 1]

            # Check capacity after swap
            r1_demand = _route_demand(instance, r1) - d1 + d2
            r2_demand = _route_demand(instance, r2) - d2 + d1
            if r1_demand > Q + 1e-10 or r2_demand > Q + 1e-10:
                continue

            r1[p1], r2[p2] = c2, c1

        elif move_type == 2 and len(non_empty) >= 2:
            # 2-opt*: exchange tails between two routes
            r1_idx, r2_idx = rng.choice(non_empty, size=2, replace=False)
            r1 = new_routes[r1_idx]
            r2 = new_routes[r2_idx]
            if not r1 or not r2:
                continue

            p1 = rng.integers(0, len(r1))
            p2 = rng.integers(0, len(r2))

            # Swap tails
            new_r1 = r1[:p1 + 1] + r2[p2 + 1:]
            new_r2 = r2[:p2 + 1] + r1[p1 + 1:]

            if (_route_demand(instance, new_r1) > Q + 1e-10 or
                    _route_demand(instance, new_r2) > Q + 1e-10):
                continue

            new_routes[r1_idx] = new_r1
            new_routes[r2_idx] = new_r2
        else:
            continue

        # Remove empty routes
        new_routes = [r for r in new_routes if r]

        new_cost = _total_distance(instance, new_routes)
        delta = new_cost - current_cost

        # Acceptance criterion
        if delta < 0 or (temp > 0 and rng.random() < math.exp(-delta / temp)):
            routes = new_routes
            current_cost = new_cost

            if current_cost < best_cost:
                best_cost = current_cost
                best_routes = _copy_routes(routes)

        temp *= cooling_rate

    # Clean up empty routes
    best_routes = [r for r in best_routes if r]
    return CVRPSolution(
        routes=best_routes,
        distance=_total_distance(instance, best_routes),
    )


if __name__ == "__main__":
    from instance import small6, christofides1, medium12

    print("=== Simulated Annealing for CVRP ===\n")

    for name, inst_fn in [
        ("small6", small6),
        ("christofides1", christofides1),
        ("medium12", medium12),
    ]:
        inst = inst_fn()
        sol = simulated_annealing(inst, seed=42)
        print(f"{name}: {sol}")
