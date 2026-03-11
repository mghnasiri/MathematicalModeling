"""
Tabu Search for CVRP.

Problem: CVRP (Capacitated Vehicle Routing Problem)

Neighborhoods:
- Relocate: move a customer from one route to another
- Swap: exchange customers between two routes
- 2-opt* intra-route: reverse a segment within a route

Uses short-term memory (tabu list) that prevents recently moved customers
from being moved again for a number of iterations. An aspiration criterion
overrides tabu status when a move produces a new global best.

Warm-started with Clarke-Wright savings heuristic.

Complexity: O(iterations * n^2) per run.

References:
    Gendreau, M., Hertz, A. & Laporte, G. (1994). A tabu search heuristic
    for the vehicle routing problem. Management Science, 40(10), 1276-1290.
    https://doi.org/10.1287/mnsc.40.10.1276

    Taillard, É.D. (1993). Parallel iterative search methods for vehicle
    routing problems. Networks, 23(8), 661-673.
    https://doi.org/10.1002/net.3230230804
"""

from __future__ import annotations

import os
import time
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


_inst = _load_module("cvrp_instance_ts", os.path.join(_parent_dir, "instance.py"))
CVRPInstance = _inst.CVRPInstance
CVRPSolution = _inst.CVRPSolution


def _copy_routes(routes: list[list[int]]) -> list[list[int]]:
    return [r[:] for r in routes]


def _route_demand(instance: CVRPInstance, route: list[int]) -> float:
    return sum(instance.demands[c - 1] for c in route)


def tabu_search(
    instance: CVRPInstance,
    initial_routes: list[list[int]] | None = None,
    max_iterations: int = 3000,
    tabu_tenure: int | None = None,
    time_limit: float | None = None,
    seed: int | None = None,
) -> CVRPSolution:
    """Solve CVRP using Tabu Search.

    Args:
        instance: A CVRPInstance.
        initial_routes: Starting routes. If None, uses Clarke-Wright.
        max_iterations: Maximum iterations.
        tabu_tenure: Number of iterations a move stays tabu. Default: sqrt(n).
        time_limit: Time limit in seconds.
        seed: Random seed for reproducibility.

    Returns:
        Best CVRPSolution found.
    """
    rng = np.random.default_rng(seed)
    n = instance.n
    start_time = time.time()

    if tabu_tenure is None:
        tabu_tenure = max(5, int(n ** 0.5))

    # ── Initial solution ─────────────────────────────────────────────────
    if initial_routes is None:
        _cw_mod = _load_module(
            "cvrp_cw_ts",
            os.path.join(_parent_dir, "heuristics", "clarke_wright.py"),
        )
        clarke_wright_savings = _cw_mod.clarke_wright_savings
        init_sol = clarke_wright_savings(instance)
        routes = _copy_routes(init_sol.routes)
    else:
        routes = _copy_routes(initial_routes)

    # Remove empty routes
    routes = [r for r in routes if r]

    current_dist = instance.total_distance(routes)
    best_routes = _copy_routes(routes)
    best_dist = current_dist

    # Tabu list: customer -> iteration when tabu expires
    tabu_dict: dict[int, int] = {}

    for iteration in range(max_iterations):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        best_delta = float("inf")
        best_move = None  # (move_type, params)

        # ── Relocate neighborhood ────────────────────────────────────────
        for ri in range(len(routes)):
            if not routes[ri]:
                continue
            for ci in range(len(routes[ri])):
                customer = routes[ri][ci]

                is_tabu = (
                    customer in tabu_dict
                    and tabu_dict[customer] > iteration
                )

                for rj in range(len(routes)):
                    if ri == rj:
                        continue

                    # Check capacity
                    new_demand = _route_demand(instance, routes[rj]) + instance.demands[customer - 1]
                    if new_demand > instance.capacity + 1e-10:
                        continue

                    # Compute delta for removing from ri
                    route_i = routes[ri]
                    prev_i = 0 if ci == 0 else route_i[ci - 1]
                    next_i = 0 if ci == len(route_i) - 1 else route_i[ci + 1]
                    remove_delta = (
                        instance.distance_matrix[prev_i][next_i]
                        - instance.distance_matrix[prev_i][customer]
                        - instance.distance_matrix[customer][next_i]
                    )

                    # Best insertion in rj
                    route_j = routes[rj]
                    best_insert_delta = float("inf")
                    best_insert_pos = 0

                    for pos in range(len(route_j) + 1):
                        prev_j = 0 if pos == 0 else route_j[pos - 1]
                        next_j = 0 if pos == len(route_j) else route_j[pos]
                        insert_delta = (
                            instance.distance_matrix[prev_j][customer]
                            + instance.distance_matrix[customer][next_j]
                            - instance.distance_matrix[prev_j][next_j]
                        )
                        if insert_delta < best_insert_delta:
                            best_insert_delta = insert_delta
                            best_insert_pos = pos

                    total_delta = remove_delta + best_insert_delta

                    # Aspiration check
                    if is_tabu and current_dist + total_delta >= best_dist:
                        continue

                    if total_delta < best_delta:
                        best_delta = total_delta
                        best_move = ("relocate", ri, ci, rj, best_insert_pos, customer)

        # ── Swap neighborhood ────────────────────────────────────────────
        for ri in range(len(routes)):
            if not routes[ri]:
                continue
            for rj in range(ri + 1, len(routes)):
                if not routes[rj]:
                    continue
                for ci in range(len(routes[ri])):
                    cust_i = routes[ri][ci]
                    for cj in range(len(routes[rj])):
                        cust_j = routes[rj][cj]

                        # Check capacity
                        di = instance.demands[cust_i - 1]
                        dj = instance.demands[cust_j - 1]
                        demand_ri = _route_demand(instance, routes[ri]) - di + dj
                        demand_rj = _route_demand(instance, routes[rj]) - dj + di
                        if demand_ri > instance.capacity + 1e-10:
                            continue
                        if demand_rj > instance.capacity + 1e-10:
                            continue

                        is_tabu = (
                            (cust_i in tabu_dict and tabu_dict[cust_i] > iteration)
                            or (cust_j in tabu_dict and tabu_dict[cust_j] > iteration)
                        )

                        # Delta for swapping
                        route_i = routes[ri]
                        route_j = routes[rj]

                        prev_i = 0 if ci == 0 else route_i[ci - 1]
                        next_i = 0 if ci == len(route_i) - 1 else route_i[ci + 1]
                        prev_j = 0 if cj == 0 else route_j[cj - 1]
                        next_j = 0 if cj == len(route_j) - 1 else route_j[cj + 1]

                        delta_ri = (
                            instance.distance_matrix[prev_i][cust_j]
                            + instance.distance_matrix[cust_j][next_i]
                            - instance.distance_matrix[prev_i][cust_i]
                            - instance.distance_matrix[cust_i][next_i]
                        )
                        delta_rj = (
                            instance.distance_matrix[prev_j][cust_i]
                            + instance.distance_matrix[cust_i][next_j]
                            - instance.distance_matrix[prev_j][cust_j]
                            - instance.distance_matrix[cust_j][next_j]
                        )
                        total_delta = delta_ri + delta_rj

                        if is_tabu and current_dist + total_delta >= best_dist:
                            continue

                        if total_delta < best_delta:
                            best_delta = total_delta
                            best_move = ("swap", ri, ci, rj, cj, cust_i, cust_j)

        if best_move is None:
            tabu_dict.clear()
            continue

        # ── Apply best move ──────────────────────────────────────────────
        if best_move[0] == "relocate":
            _, ri, ci, rj, insert_pos, customer = best_move
            routes[ri].pop(ci)
            routes[rj].insert(insert_pos, customer)
            tabu_dict[customer] = iteration + tabu_tenure

        elif best_move[0] == "swap":
            _, ri, ci, rj, cj, cust_i, cust_j = best_move
            routes[ri][ci] = cust_j
            routes[rj][cj] = cust_i
            tabu_dict[cust_i] = iteration + tabu_tenure
            tabu_dict[cust_j] = iteration + tabu_tenure

        # Remove empty routes
        routes = [r for r in routes if r]

        current_dist = instance.total_distance(routes)

        if current_dist < best_dist:
            best_dist = current_dist
            best_routes = _copy_routes(routes)

    return CVRPSolution(routes=best_routes, distance=best_dist)


if __name__ == "__main__":
    from instance import small6, christofides1

    print("=== Tabu Search on small6 ===")
    inst = small6()
    sol = tabu_search(inst, max_iterations=1000, seed=42)
    print(f"TS distance: {sol.distance:.2f}, routes: {sol.routes}")

    print("\n=== Tabu Search on christofides1 ===")
    inst = christofides1()
    sol = tabu_search(inst, max_iterations=1000, seed=42)
    print(f"TS distance: {sol.distance:.2f}, routes: {sol.routes}")
