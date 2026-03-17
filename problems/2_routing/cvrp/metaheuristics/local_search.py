"""
Local Search for CVRP.

Problem: CVRP (Capacitated Vehicle Routing Problem)

Iterative improvement using inter-route and intra-route neighborhoods:
    - Relocate: move a customer to a better position in another route
    - Swap: exchange customers between two routes
    - 2-opt: reverse a segment within a single route
    - Or-opt: move a sequence of 1-3 customers within a route

Warm-started with Clarke-Wright savings heuristic.
Best-improvement strategy within each neighborhood.

Complexity: O(iterations * n^2) per run.

References:
    Laporte, G. & Semet, F. (2002). Classical heuristics for the
    capacitated VRP. In: Toth, P. & Vigo, D. (eds) The Vehicle
    Routing Problem, SIAM, 109-128.
    https://doi.org/10.1137/1.9780898718515.ch5

    Savelsbergh, M.W.P. (1992). The vehicle routing problem with time
    windows: Minimizing route duration. ORSA Journal on Computing,
    4(2), 146-154.
    https://doi.org/10.1287/ijoc.4.2.146
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


_inst = _load_module("cvrp_instance_ls", os.path.join(_parent_dir, "instance.py"))
CVRPInstance = _inst.CVRPInstance
CVRPSolution = _inst.CVRPSolution


def local_search(
    instance: CVRPInstance,
    max_iterations: int = 1000,
    time_limit: float | None = None,
    seed: int | None = None,
) -> CVRPSolution:
    """Solve CVRP using iterative local search.

    Args:
        instance: A CVRPInstance.
        max_iterations: Maximum number of iterations.
        time_limit: Maximum wall-clock time in seconds.
        seed: Random seed for reproducibility.

    Returns:
        CVRPSolution with the best routes found.
    """
    rng = np.random.default_rng(seed)
    start_time = time.time()
    Q = instance.capacity

    # Warm-start with Clarke-Wright
    _cw_mod = _load_module(
        "cvrp_cw_ls", os.path.join(_parent_dir, "heuristics", "clarke_wright.py")
    )
    init_sol = _cw_mod.clarke_wright_savings(instance)
    routes = [r[:] for r in init_sol.routes]
    current_cost = instance.total_distance(routes)

    best_routes = [r[:] for r in routes]
    best_cost = current_cost

    for iteration in range(max_iterations):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        improved = False

        # Intra-route 2-opt
        for r_idx in range(len(routes)):
            if len(routes[r_idx]) < 3:
                continue
            delta = _best_2opt(instance, routes[r_idx])
            if delta < -1e-10:
                current_cost += delta
                improved = True

        # Inter-route relocate
        if not improved:
            delta = _best_relocate(instance, routes, Q)
            if delta < -1e-10:
                current_cost += delta
                improved = True

        # Inter-route swap
        if not improved:
            delta = _best_swap(instance, routes, Q)
            if delta < -1e-10:
                current_cost += delta
                improved = True

        # Clean up empty routes
        routes = [r for r in routes if r]
        current_cost = instance.total_distance(routes)

        if current_cost < best_cost - 1e-10:
            best_cost = current_cost
            best_routes = [r[:] for r in routes]

        if not improved:
            # Perturbation: random relocate
            _perturb(instance, routes, Q, rng)
            routes = [r for r in routes if r]
            current_cost = instance.total_distance(routes)
            if current_cost < best_cost - 1e-10:
                best_cost = current_cost
                best_routes = [r[:] for r in routes]

    return CVRPSolution(
        routes=best_routes,
        distance=instance.total_distance(best_routes),
    )


def _best_2opt(instance: CVRPInstance, route: list[int]) -> float:
    """Apply best 2-opt move within a single route. Returns delta."""
    dist = instance.distance_matrix
    n = len(route)
    best_delta = 0.0
    best_i = -1
    best_j = -1

    for i in range(n - 1):
        prev_i = 0 if i == 0 else route[i - 1]
        for j in range(i + 1, n):
            next_j = 0 if j == n - 1 else route[j + 1]

            old = dist[prev_i][route[i]] + dist[route[j]][next_j]
            new = dist[prev_i][route[j]] + dist[route[i]][next_j]
            delta = new - old

            if delta < best_delta - 1e-10:
                best_delta = delta
                best_i = i
                best_j = j

    if best_i >= 0:
        route[best_i:best_j + 1] = route[best_i:best_j + 1][::-1]

    return best_delta


def _best_relocate(
    instance: CVRPInstance, routes: list[list[int]], Q: float
) -> float:
    """Find and apply best inter-route relocate. Returns delta."""
    dist = instance.distance_matrix
    best_delta = 0.0
    best_move = None

    for src_idx in range(len(routes)):
        src = routes[src_idx]
        for pos in range(len(src)):
            cust = src[pos]
            demand = instance.demands[cust - 1]

            # Cost of removing
            prev = 0 if pos == 0 else src[pos - 1]
            nxt = 0 if pos == len(src) - 1 else src[pos + 1]
            remove_saving = (dist[prev][cust] + dist[cust][nxt] - dist[prev][nxt])

            for dst_idx in range(len(routes)):
                if dst_idx == src_idx:
                    continue
                dst = routes[dst_idx]
                if sum(instance.demands[c - 1] for c in dst) + demand > Q + 1e-10:
                    continue

                for ins in range(len(dst) + 1):
                    p = 0 if ins == 0 else dst[ins - 1]
                    n = 0 if ins >= len(dst) else dst[ins]
                    insert_cost = dist[p][cust] + dist[cust][n] - dist[p][n]
                    delta = insert_cost - remove_saving
                    if delta < best_delta - 1e-10:
                        best_delta = delta
                        best_move = (src_idx, pos, dst_idx, ins)

    if best_move:
        si, sp, di, dp = best_move
        cust = routes[si].pop(sp)
        routes[di].insert(dp, cust)

    return best_delta


def _best_swap(
    instance: CVRPInstance, routes: list[list[int]], Q: float
) -> float:
    """Find and apply best inter-route swap. Returns delta."""
    dist = instance.distance_matrix
    best_delta = 0.0
    best_move = None

    for r1_idx in range(len(routes)):
        for r2_idx in range(r1_idx + 1, len(routes)):
            r1, r2 = routes[r1_idx], routes[r2_idx]
            r1_demand = sum(instance.demands[c - 1] for c in r1)
            r2_demand = sum(instance.demands[c - 1] for c in r2)

            for p1 in range(len(r1)):
                for p2 in range(len(r2)):
                    c1, c2 = r1[p1], r2[p2]
                    d1, d2 = instance.demands[c1 - 1], instance.demands[c2 - 1]

                    if r1_demand - d1 + d2 > Q + 1e-10:
                        continue
                    if r2_demand - d2 + d1 > Q + 1e-10:
                        continue

                    # Compute delta
                    prev1 = 0 if p1 == 0 else r1[p1 - 1]
                    next1 = 0 if p1 == len(r1) - 1 else r1[p1 + 1]
                    prev2 = 0 if p2 == 0 else r2[p2 - 1]
                    next2 = 0 if p2 == len(r2) - 1 else r2[p2 + 1]

                    old = (dist[prev1][c1] + dist[c1][next1] +
                           dist[prev2][c2] + dist[c2][next2])
                    new = (dist[prev1][c2] + dist[c2][next1] +
                           dist[prev2][c1] + dist[c1][next2])
                    delta = new - old

                    if delta < best_delta - 1e-10:
                        best_delta = delta
                        best_move = (r1_idx, p1, r2_idx, p2)

    if best_move:
        r1i, p1, r2i, p2 = best_move
        routes[r1i][p1], routes[r2i][p2] = routes[r2i][p2], routes[r1i][p1]

    return best_delta


def _perturb(
    instance: CVRPInstance,
    routes: list[list[int]],
    Q: float,
    rng: np.random.Generator,
) -> None:
    """Random perturbation: relocate a random customer."""
    non_empty = [i for i in range(len(routes)) if routes[i]]
    if len(non_empty) < 1:
        return

    src_idx = rng.choice(non_empty)
    src = routes[src_idx]
    if not src:
        return
    pos = rng.integers(0, len(src))
    cust = src[pos]
    demand = instance.demands[cust - 1]

    candidates = [i for i in range(len(routes))
                  if i != src_idx and
                  sum(instance.demands[c - 1] for c in routes[i]) + demand <= Q + 1e-10]
    if not candidates:
        return

    dst_idx = rng.choice(candidates)
    src.pop(pos)
    ins = rng.integers(0, len(routes[dst_idx]) + 1)
    routes[dst_idx].insert(ins, cust)


if __name__ == "__main__":
    from instance import small6, christofides1, medium12

    print("=== Local Search for CVRP ===\n")

    for name, inst_fn in [
        ("small6", small6),
        ("christofides1", christofides1),
        ("medium12", medium12),
    ]:
        inst = inst_fn()
        sol = local_search(inst, seed=42)
        print(f"{name}: {sol}")
