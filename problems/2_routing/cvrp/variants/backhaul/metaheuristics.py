"""
VRP with Backhauls — Metaheuristics.

Algorithms:
    - Simulated Annealing with linehaul-first constraint preservation.

References:
    Toth, P. & Vigo, D. (1999). A heuristic algorithm for the symmetric and
    asymmetric vehicle routing problem with backhauls. European Journal of
    Operational Research, 113(3), 528-543.
    https://doi.org/10.1016/S0377-2217(98)00012-6
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


_inst = _load_mod("vrpb_instance_m", os.path.join(_this_dir, "instance.py"))
VRPBInstance = _inst.VRPBInstance
VRPBSolution = _inst.VRPBSolution
validate_solution = _inst.validate_solution

_heur = _load_mod("vrpb_heuristics_m", os.path.join(_this_dir, "heuristics.py"))
nearest_neighbor_vrpb = _heur.nearest_neighbor_vrpb


def _route_valid(instance: VRPBInstance, route: list[int]) -> bool:
    """Check if route respects linehaul-first and capacity constraints."""
    seen_backhaul = False
    lh_load = 0.0
    bh_load = 0.0
    for node in route:
        if instance.is_backhaul(node):
            seen_backhaul = True
            bh_load += instance.demands[node - 1]
        elif instance.is_linehaul(node):
            if seen_backhaul:
                return False
            lh_load += instance.demands[node - 1]
    return (lh_load <= instance.vehicle_capacity + 1e-6 and
            bh_load <= instance.vehicle_capacity + 1e-6)


def simulated_annealing(
    instance: VRPBInstance,
    max_iterations: int = 50000,
    cooling_rate: float = 0.9995,
    seed: int | None = None,
    time_limit: float | None = None,
) -> VRPBSolution:
    """SA for VRPB with constraint-preserving neighborhoods.

    Args:
        instance: VRPB instance.
        max_iterations: Maximum iterations.
        cooling_rate: Temperature decay factor.
        seed: Random seed.
        time_limit: Time limit in seconds.

    Returns:
        VRPBSolution.
    """
    rng = np.random.default_rng(seed)

    init = nearest_neighbor_vrpb(instance)
    routes = [list(r) for r in init.routes]
    cost = init.total_distance

    best_routes = [list(r) for r in routes]
    best_cost = cost

    temp = best_cost * 0.1
    start = time.time()

    for it in range(max_iterations):
        if time_limit and time.time() - start > time_limit:
            break

        new_routes = [list(r) for r in routes]
        move = rng.integers(0, 3)

        if move == 0 and len(new_routes) > 0:
            # Intra-route: swap two nodes of same type
            ri = int(rng.integers(0, len(new_routes)))
            r = new_routes[ri]
            if len(r) >= 2:
                i = int(rng.integers(0, len(r)))
                j = int(rng.integers(0, len(r) - 1))
                if j >= i:
                    j += 1
                # Only swap same type
                if ((instance.is_linehaul(r[i]) == instance.is_linehaul(r[j])) or
                        (instance.is_backhaul(r[i]) == instance.is_backhaul(r[j]))):
                    r[i], r[j] = r[j], r[i]

        elif move == 1 and len(new_routes) >= 2:
            # Inter-route relocate (same type)
            ri = int(rng.integers(0, len(new_routes)))
            rj = int(rng.integers(0, len(new_routes) - 1))
            if rj >= ri:
                rj += 1
            r1 = new_routes[ri]
            r2 = new_routes[rj]
            if r1:
                idx = int(rng.integers(0, len(r1)))
                node = r1[idx]
                # Insert in correct position (linehaul before backhaul)
                if instance.is_linehaul(node):
                    # Insert before first backhaul in r2
                    pos = 0
                    for p, n in enumerate(r2):
                        if instance.is_backhaul(n):
                            break
                        pos = p + 1
                    pos = min(pos, int(rng.integers(0, pos + 1)) if pos > 0 else 0)
                else:
                    # Insert after last linehaul in r2
                    pos = len(r2)
                    for p in range(len(r2) - 1, -1, -1):
                        if instance.is_linehaul(r2[p]):
                            pos = p + 1
                            break
                    if pos < len(r2):
                        pos = int(rng.integers(pos, len(r2) + 1))

                r1.pop(idx)
                r2.insert(pos, node)

        elif move == 2 and len(new_routes) >= 2:
            # Inter-route swap (same type)
            ri = int(rng.integers(0, len(new_routes)))
            rj = int(rng.integers(0, len(new_routes) - 1))
            if rj >= ri:
                rj += 1
            r1 = new_routes[ri]
            r2 = new_routes[rj]
            if r1 and r2:
                i1 = int(rng.integers(0, len(r1)))
                i2 = int(rng.integers(0, len(r2)))
                n1, n2 = r1[i1], r2[i2]
                # Only swap same type
                if (instance.is_linehaul(n1) == instance.is_linehaul(n2)):
                    r1[i1], r2[i2] = n2, n1

        # Remove empty routes
        new_routes = [r for r in new_routes if r]

        # Validate all routes
        feasible = all(_route_valid(instance, r) for r in new_routes)
        if not feasible:
            temp *= cooling_rate
            continue

        # Check all customers visited
        visited = set()
        for r in new_routes:
            visited.update(r)
        if len(visited) != instance.n_total:
            temp *= cooling_rate
            continue

        new_cost = sum(instance.route_distance(r) for r in new_routes)
        delta = new_cost - cost

        if delta < 0 or rng.random() < math.exp(-delta / max(temp, 1e-15)):
            routes = new_routes
            cost = new_cost
            if cost < best_cost:
                best_cost = cost
                best_routes = [list(r) for r in routes]

        temp *= cooling_rate

    return VRPBSolution(routes=best_routes, total_distance=best_cost)


if __name__ == "__main__":
    from instance import small_vrpb_5

    inst = small_vrpb_5()
    sol = simulated_annealing(inst, max_iterations=10000, seed=42)
    print(f"SA VRPB: {sol}")
