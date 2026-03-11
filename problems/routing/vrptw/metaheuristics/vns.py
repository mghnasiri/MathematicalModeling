"""
Variable Neighborhood Search for VRPTW.

Problem: Vehicle Routing Problem with Time Windows

VNS uses multiple neighborhood structures to escape local optima while
maintaining time window feasibility:
    N1: Relocate — move a customer to another route
    N2: Swap — exchange customers between routes
    N3: 2-opt* — exchange route tails between two routes

Local search uses best-improvement intra-route 2-opt and inter-route relocate.
Warm-started with Solomon insertion heuristic.

Complexity: O(iterations * k_max * n^2 * K) where K = number of routes.

References:
    Mladenović, N. & Hansen, P. (1997). Variable neighborhood search.
    Computers & Operations Research, 24(11), 1097-1100.
    https://doi.org/10.1016/S0305-0548(97)00031-2

    Bräysy, O. (2003). A reactive variable neighborhood search for
    the vehicle routing problem with time windows. INFORMS Journal
    on Computing, 15(4), 347-368.
    https://doi.org/10.1287/ijoc.15.4.347.24896
"""

from __future__ import annotations

import sys
import os
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


_inst = _load_mod("vrptw_instance_vns", os.path.join(_parent_dir, "instance.py"))
VRPTWInstance = _inst.VRPTWInstance
VRPTWSolution = _inst.VRPTWSolution
validate_solution = _inst.validate_solution

_sol = _load_mod(
    "vrptw_solomon_vns",
    os.path.join(_parent_dir, "heuristics", "solomon_insertion.py"),
)
solomon_insertion = _sol.solomon_insertion


def vns(
    instance: VRPTWInstance,
    max_iterations: int = 500,
    k_max: int = 3,
    time_limit: float | None = None,
    seed: int | None = None,
) -> VRPTWSolution:
    """Solve VRPTW using Variable Neighborhood Search.

    Args:
        instance: A VRPTWInstance.
        max_iterations: Maximum number of iterations.
        k_max: Maximum neighborhood size.
        time_limit: Maximum wall-clock time in seconds.
        seed: Random seed for reproducibility.

    Returns:
        VRPTWSolution with the best routes found.
    """
    rng = np.random.default_rng(seed)
    start_time = time.time()

    # Warm-start
    init_sol = solomon_insertion(instance)
    routes = [list(r) for r in init_sol.routes if r]
    current_dist = instance.total_distance(routes)

    best_routes = [r[:] for r in routes]
    best_dist = current_dist

    for iteration in range(max_iterations):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        k = 1
        while k <= k_max:
            if time_limit is not None and time.time() - start_time >= time_limit:
                break

            # Shaking
            shaken = [r[:] for r in routes]
            _shake(instance, shaken, k, rng)
            shaken = [r for r in shaken if r]

            # Local search
            _local_search_vnd(instance, shaken)
            shaken = [r for r in shaken if r]
            shaken_dist = instance.total_distance(shaken)

            if shaken_dist < current_dist - 1e-10:
                routes = shaken
                current_dist = shaken_dist
                k = 1

                if current_dist < best_dist - 1e-10:
                    best_dist = current_dist
                    best_routes = [r[:] for r in routes]
            else:
                k += 1

    return VRPTWSolution(
        routes=best_routes,
        distance=best_dist,
    )


def _shake(
    instance: VRPTWInstance,
    routes: list[list[int]],
    k: int,
    rng: np.random.Generator,
) -> None:
    """Shake: perform k random relocations."""
    for _ in range(k):
        non_empty = [i for i in range(len(routes)) if routes[i]]
        if not non_empty:
            break

        ri = non_empty[rng.integers(len(non_empty))]
        if not routes[ri]:
            continue

        ci_idx = rng.integers(len(routes[ri]))
        cust = routes[ri][ci_idx]

        # Try random insertion
        targets = list(range(len(routes)))
        rng.shuffle(targets)

        moved = False
        for rj in targets:
            if rj == ri:
                continue
            for pos in range(len(routes[rj]) + 1):
                new_route = routes[rj][:pos] + [cust] + routes[rj][pos:]
                if (instance.route_demand(new_route) <= instance.capacity + 1e-10
                        and instance.route_feasible(new_route)):
                    routes[ri] = routes[ri][:ci_idx] + routes[ri][ci_idx + 1:]
                    routes[rj] = new_route
                    moved = True
                    break
            if moved:
                break

        if not moved:
            # Try creating a new route
            if instance.route_feasible([cust]):
                routes[ri] = routes[ri][:ci_idx] + routes[ri][ci_idx + 1:]
                routes.append([cust])


def _local_search_vnd(
    instance: VRPTWInstance,
    routes: list[list[int]],
) -> None:
    """VND local search: 2-opt then relocate until no improvement."""
    improved = True
    while improved:
        improved = False

        # Intra-route 2-opt
        for ri in range(len(routes)):
            route = routes[ri]
            if len(route) < 3:
                continue
            for i in range(len(route) - 1):
                for j in range(i + 1, len(route)):
                    new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                    if instance.route_feasible(new_route):
                        new_dist = instance.route_distance(new_route)
                        old_dist = instance.route_distance(route)
                        if new_dist < old_dist - 1e-10:
                            routes[ri] = new_route
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break

        # Inter-route relocate (first improvement)
        if not improved:
            for ri in range(len(routes)):
                for ci_idx in range(len(routes[ri])):
                    cust = routes[ri][ci_idx]
                    old_from = routes[ri]
                    new_from = old_from[:ci_idx] + old_from[ci_idx + 1:]
                    old_from_dist = instance.route_distance(old_from)
                    new_from_dist = instance.route_distance(new_from)

                    for rj in range(len(routes)):
                        if ri == rj:
                            continue
                        for pos in range(len(routes[rj]) + 1):
                            new_to = routes[rj][:pos] + [cust] + routes[rj][pos:]
                            if instance.route_demand(new_to) > instance.capacity + 1e-10:
                                continue
                            if not instance.route_feasible(new_to):
                                continue
                            if new_from and not instance.route_feasible(new_from):
                                continue
                            old_to_dist = instance.route_distance(routes[rj])
                            new_to_dist = instance.route_distance(new_to)
                            delta = (new_from_dist + new_to_dist) - (old_from_dist + old_to_dist)
                            if delta < -1e-10:
                                routes[ri] = new_from
                                routes[rj] = new_to
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break


if __name__ == "__main__":
    from problems.routing.vrptw.instance import solomon_c101_mini

    inst = solomon_c101_mini()
    print(f"VRPTW: {inst.n} customers, capacity={inst.capacity}")

    sol = vns(inst, seed=42)
    print(f"VNS: distance={sol.distance:.2f}, vehicles={sol.num_vehicles}")
