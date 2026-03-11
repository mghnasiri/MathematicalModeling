"""
Variable Neighborhood Search for CVRP.

Problem: CVRP (Capacitated Vehicle Routing Problem)

VNS systematically changes neighborhoods during the search to escape local
optima. Uses three neighborhood structures for shaking and VND-style local
search for intensification:

Neighborhoods:
    N1: Relocate — move a customer to another route
    N2: Swap — exchange customers between two routes
    N3: 2-opt* — exchange tails between two routes

Warm-started with Clarke-Wright savings heuristic.

Complexity: O(iterations * n^2) per run.

References:
    Mladenović, N. & Hansen, P. (1997). Variable neighborhood search.
    Computers & Operations Research, 24(11), 1097-1100.
    https://doi.org/10.1016/S0305-0548(97)00031-2

    Hansen, P., Mladenović, N. & Moreno Pérez, J.A. (2010). Variable
    neighbourhood search: methods and applications. Annals of Operations
    Research, 175(1), 367-407.
    https://doi.org/10.1007/s10479-009-0657-6
"""

from __future__ import annotations

import os
import time
import numpy as np

_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_module(name, filepath):
    import importlib.util
    import sys as _sys
    if name in _sys.modules:
        return _sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    _sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_inst = _load_module("cvrp_instance_vns", os.path.join(_parent_dir, "instance.py"))
CVRPInstance = _inst.CVRPInstance
CVRPSolution = _inst.CVRPSolution


def _copy_routes(routes: list[list[int]]) -> list[list[int]]:
    return [r[:] for r in routes]


def _route_demand(instance: CVRPInstance, route: list[int]) -> float:
    return sum(instance.demands[c - 1] for c in route)


def _total_distance(instance: CVRPInstance, routes: list[list[int]]) -> float:
    return instance.total_distance(routes)


def vns(
    instance: CVRPInstance,
    max_iterations: int = 500,
    k_max: int = 3,
    time_limit: float | None = None,
    seed: int | None = None,
) -> CVRPSolution:
    """Solve CVRP using Variable Neighborhood Search.

    Args:
        instance: A CVRPInstance.
        max_iterations: Maximum number of VNS iterations.
        k_max: Number of neighborhood structures (1-3).
        time_limit: Maximum wall-clock time in seconds.
        seed: Random seed for reproducibility.

    Returns:
        CVRPSolution with the best routes found.
    """
    rng = np.random.default_rng(seed)
    n = instance.n
    Q = instance.capacity
    start_time = time.time()

    # Warm-start with Clarke-Wright
    _cw_mod = _load_module(
        "cvrp_cw_vns", os.path.join(_parent_dir, "heuristics", "clarke_wright.py")
    )
    init_sol = _cw_mod.clarke_wright_savings(instance)
    routes = _copy_routes(init_sol.routes)
    current_cost = _total_distance(instance, routes)

    best_routes = _copy_routes(routes)
    best_cost = current_cost

    for iteration in range(max_iterations):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        k = 1
        while k <= k_max:
            if time_limit is not None and time.time() - start_time >= time_limit:
                break

            # Shaking: random move in neighborhood k
            shaken = _shake(instance, routes, k, rng, Q)
            if shaken is None:
                k += 1
                continue

            # Local search (VND)
            improved = _local_search(instance, shaken, Q, rng)
            improved_cost = _total_distance(instance, improved)

            # Move or not
            if improved_cost < current_cost - 1e-10:
                routes = improved
                current_cost = improved_cost
                k = 1  # restart from first neighborhood

                if current_cost < best_cost - 1e-10:
                    best_cost = current_cost
                    best_routes = _copy_routes(routes)
            else:
                k += 1

    best_routes = [r for r in best_routes if r]
    return CVRPSolution(
        routes=best_routes,
        distance=_total_distance(instance, best_routes),
    )


def _shake(
    instance: CVRPInstance,
    routes: list[list[int]],
    k: int,
    rng: np.random.Generator,
    Q: float,
) -> list[list[int]] | None:
    """Perform random perturbation in neighborhood k."""
    new_routes = _copy_routes(routes)
    non_empty = [i for i, r in enumerate(new_routes) if r]

    if k == 1:
        # N1: Relocate a random customer
        if len(non_empty) < 1:
            return None
        src_idx = rng.choice(non_empty)
        src = new_routes[src_idx]
        if not src:
            return None
        pos = rng.integers(0, len(src))
        cust = src[pos]
        demand = instance.demands[cust - 1]

        # Find feasible destination
        candidates = [i for i in range(len(new_routes))
                      if i != src_idx and _route_demand(instance, new_routes[i]) + demand <= Q + 1e-10]
        if not candidates:
            # Create a new route
            src.pop(pos)
            new_routes.append([cust])
        else:
            dst_idx = rng.choice(candidates)
            src.pop(pos)
            ins_pos = rng.integers(0, len(new_routes[dst_idx]) + 1)
            new_routes[dst_idx].insert(ins_pos, cust)

    elif k == 2:
        # N2: Swap two customers between different routes
        if len(non_empty) < 2:
            return None
        r1_idx, r2_idx = rng.choice(non_empty, size=2, replace=False)
        r1, r2 = new_routes[r1_idx], new_routes[r2_idx]
        if not r1 or not r2:
            return None
        p1, p2 = rng.integers(0, len(r1)), rng.integers(0, len(r2))
        c1, c2 = r1[p1], r2[p2]
        d1, d2 = instance.demands[c1 - 1], instance.demands[c2 - 1]

        new_r1_demand = _route_demand(instance, r1) - d1 + d2
        new_r2_demand = _route_demand(instance, r2) - d2 + d1
        if new_r1_demand > Q + 1e-10 or new_r2_demand > Q + 1e-10:
            return None
        r1[p1], r2[p2] = c2, c1

    elif k == 3:
        # N3: 2-opt* — exchange tails
        if len(non_empty) < 2:
            return None
        r1_idx, r2_idx = rng.choice(non_empty, size=2, replace=False)
        r1, r2 = new_routes[r1_idx], new_routes[r2_idx]
        if not r1 or not r2:
            return None
        p1, p2 = rng.integers(0, len(r1)), rng.integers(0, len(r2))
        new_r1 = r1[:p1 + 1] + r2[p2 + 1:]
        new_r2 = r2[:p2 + 1] + r1[p1 + 1:]
        if (_route_demand(instance, new_r1) > Q + 1e-10 or
                _route_demand(instance, new_r2) > Q + 1e-10):
            return None
        new_routes[r1_idx] = new_r1
        new_routes[r2_idx] = new_r2

    new_routes = [r for r in new_routes if r]
    return new_routes


def _local_search(
    instance: CVRPInstance,
    routes: list[list[int]],
    Q: float,
    rng: np.random.Generator,
) -> list[list[int]]:
    """Apply best-improvement relocate local search."""
    improved = True
    while improved:
        improved = False
        non_empty = [i for i, r in enumerate(routes) if r]

        for src_idx in non_empty:
            src = routes[src_idx]
            for pos in range(len(src)):
                cust = src[pos]
                demand = instance.demands[cust - 1]

                # Current cost contribution of customer in source
                old_cost = _removal_saving(instance, src, pos)

                best_delta = 0.0
                best_dst_idx = -1
                best_ins_pos = -1

                for dst_idx in range(len(routes)):
                    if dst_idx == src_idx:
                        continue
                    dst = routes[dst_idx]
                    if _route_demand(instance, dst) + demand > Q + 1e-10:
                        continue

                    for ins in range(len(dst) + 1):
                        insert_cost = _insertion_cost(instance, dst, ins, cust)
                        delta = insert_cost - old_cost
                        if delta < best_delta - 1e-10:
                            best_delta = delta
                            best_dst_idx = dst_idx
                            best_ins_pos = ins

                if best_dst_idx >= 0:
                    cust = src.pop(pos)
                    routes[best_dst_idx].insert(best_ins_pos, cust)
                    improved = True
                    break
            if improved:
                break

    routes = [r for r in routes if r]
    return routes


def _removal_saving(instance: CVRPInstance, route: list[int], pos: int) -> float:
    """Cost change (negative = saving) of removing customer at pos from route."""
    dist = instance.distance_matrix
    cust = route[pos]
    prev_node = 0 if pos == 0 else route[pos - 1]
    next_node = 0 if pos == len(route) - 1 else route[pos + 1]

    old_cost = dist[prev_node][cust] + dist[cust][next_node]
    new_cost = dist[prev_node][next_node]
    return -(old_cost - new_cost)  # negative = removing saves distance


def _insertion_cost(instance: CVRPInstance, route: list[int], pos: int, cust: int) -> float:
    """Extra cost of inserting customer at position pos in route."""
    dist = instance.distance_matrix
    prev_node = 0 if pos == 0 else route[pos - 1]
    next_node = 0 if pos >= len(route) else route[pos]

    return dist[prev_node][cust] + dist[cust][next_node] - dist[prev_node][next_node]


if __name__ == "__main__":
    from instance import small6, christofides1, medium12

    print("=== Variable Neighborhood Search for CVRP ===\n")

    for name, inst_fn in [
        ("small6", small6),
        ("christofides1", christofides1),
        ("medium12", medium12),
    ]:
        inst = inst_fn()
        sol = vns(inst, seed=42)
        print(f"{name}: {sol}")
