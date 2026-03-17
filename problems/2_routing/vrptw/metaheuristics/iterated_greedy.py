"""
Iterated Greedy for VRPTW.

Problem: VRPTW (Vehicle Routing Problem with Time Windows)

Iterated Greedy repeatedly destroys and reconstructs solutions:
    1. Destroy: remove d random customers from routes
    2. Repair: reinsert each removed customer at the cheapest feasible position
    3. Accept: Boltzmann-based acceptance criterion

Warm-started with Solomon's insertion heuristic.

Complexity: O(iterations * d * n * K) per run.

References:
    Ruiz, R. & Stuetzle, T. (2007). A simple and effective iterated
    greedy algorithm for the permutation flowshop scheduling problem.
    European Journal of Operational Research, 177(3), 2033-2049.
    https://doi.org/10.1016/j.ejor.2005.12.009

    Solomon, M.M. (1987). Algorithms for the vehicle routing and
    scheduling problems with time window constraints. Operations
    Research, 35(2), 254-265.
    https://doi.org/10.1287/opre.35.2.254
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


_inst = _load_mod("vrptw_instance_ig", os.path.join(_parent_dir, "instance.py"))
VRPTWInstance = _inst.VRPTWInstance
VRPTWSolution = _inst.VRPTWSolution


def iterated_greedy(
    instance: VRPTWInstance,
    max_iterations: int = 5000,
    d: int | None = None,
    temperature_factor: float = 0.1,
    time_limit: float | None = None,
    seed: int | None = None,
) -> VRPTWSolution:
    """Solve VRPTW using Iterated Greedy.

    Args:
        instance: A VRPTWInstance.
        max_iterations: Maximum number of iterations.
        d: Number of customers to remove per iteration. Defaults to max(1, n//4).
        temperature_factor: Temperature as fraction of initial cost.
        time_limit: Maximum wall-clock time in seconds.
        seed: Random seed for reproducibility.

    Returns:
        VRPTWSolution with the best routes found.
    """
    rng = np.random.default_rng(seed)
    n = instance.n
    start_time = time.time()

    if d is None:
        d = max(1, n // 4)

    # Warm-start with Solomon insertion
    _si = _load_mod(
        "vrptw_si_ig",
        os.path.join(_parent_dir, "heuristics", "solomon_insertion.py"),
    )
    init_sol = _si.solomon_insertion(instance)
    routes = [r[:] for r in init_sol.routes]
    routes = [r for r in routes if r]

    current_cost = instance.total_distance(routes)
    best_routes = [r[:] for r in routes]
    best_cost = current_cost

    temperature = temperature_factor * current_cost if current_cost > 0 else 1.0

    for iteration in range(max_iterations):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        # Destroy: remove d random customers
        all_customers = []
        for ri, route in enumerate(routes):
            for ci, cust in enumerate(route):
                all_customers.append((ri, ci, cust))

        if len(all_customers) < d:
            d_actual = len(all_customers)
        else:
            d_actual = d

        if d_actual == 0:
            break

        indices = rng.choice(len(all_customers), size=d_actual, replace=False)
        removed = []
        # Sort in reverse to remove from end first
        remove_info = sorted(
            [all_customers[i] for i in indices],
            key=lambda x: (x[0], x[1]),
            reverse=True,
        )
        for ri, ci, cust in remove_info:
            routes[ri].pop(ci)
            removed.append(cust)

        # Remove empty routes
        routes = [r for r in routes if r]

        # Repair: reinsert each removed customer at cheapest feasible position
        rng.shuffle(removed)
        for cust in removed:
            best_pos = None
            best_increase = float("inf")

            for ri in range(len(routes)):
                route = routes[ri]
                for pos in range(len(route) + 1):
                    route.insert(pos, cust)
                    if instance.route_feasible(route):
                        new_dist = instance.route_distance(route)
                        old_route = route[:pos] + route[pos + 1:]
                        old_dist = instance.route_distance(old_route)
                        increase = new_dist - old_dist
                        if increase < best_increase:
                            best_increase = increase
                            best_pos = (ri, pos)
                    route.pop(pos)

            if best_pos is None:
                # Open a new route
                new_route = [cust]
                if instance.route_feasible(new_route):
                    routes.append(new_route)
                else:
                    # Force insert in least-bad position (shouldn't happen with valid instances)
                    routes.append([cust])
            else:
                ri, pos = best_pos
                routes[ri].insert(pos, cust)

        routes = [r for r in routes if r]
        new_cost = instance.total_distance(routes)

        # Acceptance criterion
        delta = new_cost - current_cost
        if delta < 0 or (temperature > 0 and rng.random() < math.exp(-delta / temperature)):
            current_cost = new_cost
            if current_cost < best_cost - 1e-10:
                best_cost = current_cost
                best_routes = [r[:] for r in routes]
        else:
            # Revert to current best working solution
            routes = [r[:] for r in best_routes]
            current_cost = best_cost

    best_routes = [r for r in best_routes if r]
    return VRPTWSolution(
        routes=best_routes,
        distance=instance.total_distance(best_routes),
    )


if __name__ == "__main__":
    inst = VRPTWInstance.random(n=10, seed=42)
    print(f"VRPTW: {inst.n} customers, capacity={inst.capacity}")

    sol = iterated_greedy(inst, seed=42)
    print(f"IG: distance={sol.distance:.2f}, vehicles={sol.num_vehicles}")

    _si = _load_mod(
        "vrptw_si_ig_main",
        os.path.join(_parent_dir, "heuristics", "solomon_insertion.py"),
    )
    si_sol = _si.solomon_insertion(inst)
    print(f"Solomon I1: distance={si_sol.distance:.2f}, vehicles={si_sol.num_vehicles}")
