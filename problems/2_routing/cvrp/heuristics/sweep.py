"""
Sweep Algorithm — CVRP constructive heuristic based on angular sweep.

Problem: CVRP (Capacitated Vehicle Routing Problem)
Complexity: O(n log n) — dominated by angular sorting

Assign customers to routes by sweeping a ray centered at the depot.
Customers are sorted by their angle relative to the depot. A new route
is started whenever adding the next customer would exceed capacity.

Requires coordinate data. Simple and fast, works well when customers
are geographically clustered.

References:
    Gillett, B.E. & Miller, L.R. (1974). A heuristic algorithm for
    the vehicle-dispatch problem. Operations Research, 22(2), 340-349.
    https://doi.org/10.1287/opre.22.2.340
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

_inst = _load_module("cvrp_instance_sw", os.path.join(_parent_dir, "instance.py"))
CVRPInstance = _inst.CVRPInstance
CVRPSolution = _inst.CVRPSolution


def sweep(instance: CVRPInstance, start_angle: float = 0.0) -> CVRPSolution:
    """Construct CVRP routes using the sweep algorithm.

    Args:
        instance: A CVRPInstance (must have coords).
        start_angle: Starting angle in radians for the sweep.

    Returns:
        CVRPSolution with constructed routes.

    Raises:
        ValueError: If instance has no coordinate data.
    """
    if instance.coords is None:
        raise ValueError("Sweep algorithm requires coordinate data")

    n = instance.n
    Q = instance.capacity
    depot = instance.coords[0]

    # Compute angles of customers relative to depot
    angles = []
    for i in range(1, n + 1):
        dx = instance.coords[i][0] - depot[0]
        dy = instance.coords[i][1] - depot[1]
        angle = math.atan2(dy, dx)
        # Adjust relative to start_angle
        angle = (angle - start_angle) % (2 * math.pi)
        angles.append((angle, i))

    # Sort by angle
    angles.sort()

    # Build routes by sweeping
    routes: list[list[int]] = []
    current_route: list[int] = []
    current_demand = 0.0

    for _, customer in angles:
        demand = instance.demands[customer - 1]
        if current_demand + demand > Q + 1e-10:
            if current_route:
                routes.append(current_route)
            current_route = [customer]
            current_demand = demand
        else:
            current_route.append(customer)
            current_demand += demand

    if current_route:
        routes.append(current_route)

    total_dist = instance.total_distance(routes)
    return CVRPSolution(routes=routes, distance=total_dist)


def sweep_multistart(instance: CVRPInstance, num_starts: int = 12) -> CVRPSolution:
    """Run sweep from multiple starting angles and return the best solution.

    Args:
        instance: A CVRPInstance (must have coords).
        num_starts: Number of evenly spaced starting angles to try.

    Returns:
        Best CVRPSolution across all starting angles.
    """
    best_sol = None
    for k in range(num_starts):
        angle = 2 * math.pi * k / num_starts
        sol = sweep(instance, start_angle=angle)
        if best_sol is None or sol.distance < best_sol.distance:
            best_sol = sol
    return best_sol


if __name__ == "__main__":
    from instance import small6, medium12

    print("=== Sweep Algorithm ===\n")

    for name, inst_fn in [("small6", small6), ("medium12", medium12)]:
        inst = inst_fn()
        sol = sweep(inst)
        sol_multi = sweep_multistart(inst)
        print(f"{name}: single={sol.distance:.1f}, multi={sol_multi.distance:.1f}")
        print(f"  routes: {sol_multi.routes}")
