"""
Tabu Search for VRPTW.

Problem: VRPTW (Vehicle Routing Problem with Time Windows)

Neighborhoods:
- Relocate: move a customer from one route to another (TW-feasible)
- Swap: exchange two customers between routes (TW-feasible)

Uses short-term memory (tabu list) preventing recently moved customers
from being moved again for a number of iterations. Aspiration criterion
overrides tabu when a move yields a new global best.

Warm-started with Solomon's I1 insertion heuristic.

Complexity: O(iterations * n^2) per run.

References:
    Cordeau, J.-F., Laporte, G. & Mercier, A. (2001). A unified tabu
    search heuristic for vehicle routing problems with time windows.
    Journal of the Operational Research Society, 52(8), 928-936.
    https://doi.org/10.1057/palgrave.jors.2601163

    Taillard, É.D., Badeau, P., Gendreau, M., Guertin, F. & Potvin,
    J.-Y. (1997). A tabu search heuristic for the vehicle routing
    problem with soft time windows. Transportation Science, 31(2),
    170-186.
    https://doi.org/10.1287/trsc.31.2.170
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


_inst = _load_module("vrptw_instance_ts", os.path.join(_parent_dir, "instance.py"))
VRPTWInstance = _inst.VRPTWInstance
VRPTWSolution = _inst.VRPTWSolution


def _copy_routes(routes: list[list[int]]) -> list[list[int]]:
    return [r[:] for r in routes]


def tabu_search(
    instance: VRPTWInstance,
    initial_routes: list[list[int]] | None = None,
    max_iterations: int = 3000,
    tabu_tenure: int | None = None,
    time_limit: float | None = None,
    seed: int | None = None,
) -> VRPTWSolution:
    """Solve VRPTW using Tabu Search.

    Args:
        instance: A VRPTWInstance.
        initial_routes: Starting routes. If None, uses Solomon insertion.
        max_iterations: Maximum iterations.
        tabu_tenure: Number of iterations a move stays tabu. Default: sqrt(n).
        time_limit: Time limit in seconds.
        seed: Random seed for reproducibility.

    Returns:
        Best VRPTWSolution found.
    """
    rng = np.random.default_rng(seed)
    n = instance.n
    start_time = time.time()

    if tabu_tenure is None:
        tabu_tenure = max(5, int(n ** 0.5))

    # Initial solution
    if initial_routes is None:
        _si_mod = _load_module(
            "vrptw_si_ts",
            os.path.join(_parent_dir, "heuristics", "solomon_insertion.py"),
        )
        init_sol = _si_mod.solomon_insertion(instance)
        routes = _copy_routes(init_sol.routes)
    else:
        routes = _copy_routes(initial_routes)

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
        best_move = None

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

                    # Try inserting customer into route rj
                    test_src = routes[ri][:ci] + routes[ri][ci + 1:]

                    # Check capacity
                    new_demand = (
                        instance.route_demand(routes[rj])
                        + instance.demands[customer - 1]
                    )
                    if new_demand > instance.capacity + 1e-10:
                        continue

                    # Try best insertion position
                    best_insert_delta = float("inf")
                    best_insert_pos = 0

                    for pos in range(len(routes[rj]) + 1):
                        test_dst = list(routes[rj])
                        test_dst.insert(pos, customer)

                        if not instance.route_feasible(test_dst):
                            continue

                        # Compute delta
                        old_dist = (
                            instance.route_distance(routes[ri])
                            + instance.route_distance(routes[rj])
                        )
                        new_dist = (
                            instance.route_distance(test_src)
                            + instance.route_distance(test_dst)
                        )
                        delta = new_dist - old_dist

                        if delta < best_insert_delta:
                            best_insert_delta = delta
                            best_insert_pos = pos

                    if best_insert_delta < float("inf"):
                        total_delta = best_insert_delta

                        # Aspiration check
                        if is_tabu and current_dist + total_delta >= best_dist:
                            continue

                        if total_delta < best_delta:
                            best_delta = total_delta
                            best_move = (
                                "relocate", ri, ci, rj,
                                best_insert_pos, customer,
                            )

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
                        demand_ri = (
                            instance.route_demand(routes[ri]) - di + dj
                        )
                        demand_rj = (
                            instance.route_demand(routes[rj]) - dj + di
                        )
                        if demand_ri > instance.capacity + 1e-10:
                            continue
                        if demand_rj > instance.capacity + 1e-10:
                            continue

                        is_tabu = (
                            (cust_i in tabu_dict
                             and tabu_dict[cust_i] > iteration)
                            or (cust_j in tabu_dict
                                and tabu_dict[cust_j] > iteration)
                        )

                        # Test swap feasibility
                        test_ri = list(routes[ri])
                        test_rj = list(routes[rj])
                        test_ri[ci] = cust_j
                        test_rj[cj] = cust_i

                        if (not instance.route_feasible(test_ri)
                                or not instance.route_feasible(test_rj)):
                            continue

                        old_dist = (
                            instance.route_distance(routes[ri])
                            + instance.route_distance(routes[rj])
                        )
                        new_dist = (
                            instance.route_distance(test_ri)
                            + instance.route_distance(test_rj)
                        )
                        total_delta = new_dist - old_dist

                        if is_tabu and current_dist + total_delta >= best_dist:
                            continue

                        if total_delta < best_delta:
                            best_delta = total_delta
                            best_move = (
                                "swap", ri, ci, rj, cj, cust_i, cust_j,
                            )

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

        routes = [r for r in routes if r]
        current_dist = instance.total_distance(routes)

        if current_dist < best_dist:
            best_dist = current_dist
            best_routes = _copy_routes(routes)

    best_routes = [r for r in best_routes if r]
    return VRPTWSolution(
        routes=best_routes,
        distance=instance.total_distance(best_routes),
    )


if __name__ == "__main__":
    from instance import solomon_c101_mini, tight_tw5

    print("=== Tabu Search for VRPTW ===\n")

    for name, inst_fn in [
        ("solomon_c101_mini", solomon_c101_mini),
        ("tight_tw5", tight_tw5),
    ]:
        inst = inst_fn()
        sol = tabu_search(inst, max_iterations=500, seed=42)
        print(f"{name}: {sol}")
