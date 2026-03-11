"""
Local Search for VRPTW.

Problem: Vehicle Routing Problem with Time Windows

Iterative improvement using relocate, swap, and 2-opt neighborhoods,
all with time window feasibility checks. Includes random perturbation
for escaping local optima.

Neighborhoods:
    - Intra-route 2-opt: reverse a segment within a route
    - Inter-route relocate: move a customer between routes
    - Inter-route swap: exchange customers between routes

Warm-started with Solomon insertion heuristic.

Complexity: O(iterations * n^2 * K) where K = number of routes.

References:
    Solomon, M.M. (1987). Algorithms for the vehicle routing and
    scheduling problems with time window constraints. Operations
    Research, 35(2), 254-265.
    https://doi.org/10.1287/opre.35.2.254

    Bräysy, O. & Gendreau, M. (2005). Vehicle routing problem with
    time windows, Part I: Route construction and local search algorithms.
    Transportation Science, 39(1), 104-118.
    https://doi.org/10.1287/trsc.1030.0056
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


_inst = _load_mod("vrptw_instance_ls", os.path.join(_parent_dir, "instance.py"))
VRPTWInstance = _inst.VRPTWInstance
VRPTWSolution = _inst.VRPTWSolution
validate_solution = _inst.validate_solution

_sol = _load_mod(
    "vrptw_solomon_ls",
    os.path.join(_parent_dir, "heuristics", "solomon_insertion.py"),
)
solomon_insertion = _sol.solomon_insertion


def local_search(
    instance: VRPTWInstance,
    max_iterations: int = 1000,
    time_limit: float | None = None,
    seed: int | None = None,
) -> VRPTWSolution:
    """Solve VRPTW using Local Search.

    Args:
        instance: A VRPTWInstance.
        max_iterations: Maximum number of iterations.
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

    no_improve = 0

    for iteration in range(max_iterations):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

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
                            current_dist += new_dist - old_dist
                            improved = True
                            break
                if improved:
                    break

        # Inter-route relocate (best-improvement)
        best_delta = 0.0
        best_relocate = None

        for ri in range(len(routes)):
            for ci_idx in range(len(routes[ri])):
                cust = routes[ri][ci_idx]
                old_route = routes[ri]
                new_from = old_route[:ci_idx] + old_route[ci_idx + 1:]
                old_from_dist = instance.route_distance(old_route)
                new_from_dist = instance.route_distance(new_from)

                for rj in range(len(routes)):
                    if ri == rj:
                        continue
                    for pos in range(len(routes[rj]) + 1):
                        new_to = routes[rj][:pos] + [cust] + routes[rj][pos:]
                        # Check capacity
                        if instance.route_demand(new_to) > instance.capacity + 1e-10:
                            continue
                        if not instance.route_feasible(new_to):
                            continue
                        if new_from and not instance.route_feasible(new_from):
                            continue
                        old_to_dist = instance.route_distance(routes[rj])
                        new_to_dist = instance.route_distance(new_to)
                        delta = (new_from_dist + new_to_dist) - (old_from_dist + old_to_dist)
                        if delta < best_delta - 1e-10:
                            best_delta = delta
                            best_relocate = (ri, ci_idx, rj, pos)

        if best_relocate is not None:
            ri, ci_idx, rj, pos = best_relocate
            cust = routes[ri][ci_idx]
            routes[ri] = routes[ri][:ci_idx] + routes[ri][ci_idx + 1:]
            # Adjust pos if same route (shouldn't be, but safety)
            routes[rj] = routes[rj][:pos] + [cust] + routes[rj][pos:]
            # Remove empty routes
            routes = [r for r in routes if r]
            current_dist = instance.total_distance(routes)
            improved = True

        if improved:
            no_improve = 0
            if current_dist < best_dist - 1e-10:
                best_dist = current_dist
                best_routes = [r[:] for r in routes]
        else:
            no_improve += 1
            if no_improve >= 5:
                # Perturbation: random relocate
                _perturb(instance, routes, rng)
                routes = [r for r in routes if r]
                current_dist = instance.total_distance(routes)
                no_improve = 0

    return VRPTWSolution(
        routes=best_routes,
        distance=best_dist,
    )


def _perturb(
    instance: VRPTWInstance,
    routes: list[list[int]],
    rng: np.random.Generator,
) -> None:
    """Random perturbation: move a random customer to a random feasible position."""
    if not routes:
        return

    non_empty = [i for i, r in enumerate(routes) if r]
    if not non_empty:
        return

    ri = non_empty[rng.integers(len(non_empty))]
    if not routes[ri]:
        return

    ci_idx = rng.integers(len(routes[ri]))
    cust = routes[ri][ci_idx]
    routes[ri] = routes[ri][:ci_idx] + routes[ri][ci_idx + 1:]

    # Try to insert into a random route at the best feasible position
    targets = list(range(len(routes)))
    rng.shuffle(targets)

    for rj in targets:
        for pos in range(len(routes[rj]) + 1):
            new_route = routes[rj][:pos] + [cust] + routes[rj][pos:]
            if (instance.route_demand(new_route) <= instance.capacity + 1e-10
                    and instance.route_feasible(new_route)):
                routes[rj] = new_route
                return

    # If can't insert anywhere, create a new route
    if instance.route_feasible([cust]):
        routes.append([cust])
    else:
        # Put back in original position
        routes[ri].insert(ci_idx, cust)


if __name__ == "__main__":
    from problems.routing.vrptw.instance import solomon_c101_mini

    inst = solomon_c101_mini()
    print(f"VRPTW: {inst.n} customers, capacity={inst.capacity}")

    sol = local_search(inst, seed=42)
    print(f"LS: distance={sol.distance:.2f}, vehicles={sol.num_vehicles}")
