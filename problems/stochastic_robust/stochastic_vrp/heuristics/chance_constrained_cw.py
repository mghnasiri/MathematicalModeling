"""
Chance-Constrained Clarke-Wright Savings for Stochastic VRP

Adapts the classic Clarke-Wright savings heuristic to stochastic demands.
Route merges are only accepted if the merged route's overflow probability
remains below the threshold alpha.

Complexity: O(n^2 log n + n^2 * S) — savings computation + feasibility checks.

References:
    - Clarke, G. & Wright, J.W. (1964). Scheduling of vehicles from a central
      depot to a number of delivery points. Oper. Res., 12(4), 568-581.
      https://doi.org/10.1287/opre.12.4.568
    - Gendreau, M., Laporte, G. & Séguin, R. (1996). Stochastic vehicle
      routing. EJOR, 88(1), 3-12.
"""
from __future__ import annotations

import sys
import os
import importlib.util

import numpy as np

def _load_parent(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_inst = _load_parent("svrp_instance", os.path.join(os.path.dirname(__file__), "..", "instance.py"))
StochasticVRPInstance = _inst.StochasticVRPInstance
StochasticVRPSolution = _inst.StochasticVRPSolution


def chance_constrained_savings(
    instance: StochasticVRPInstance,
) -> StochasticVRPSolution:
    """Clarke-Wright savings with chance constraint on route capacity.

    1. Initialize one route per customer.
    2. Compute savings s(i,j) = d(0,i) + d(0,j) - d(i,j).
    3. Sort savings descending; merge routes if overflow prob <= alpha.

    Args:
        instance: StochasticVRPInstance.

    Returns:
        StochasticVRPSolution.
    """
    n = instance.n_customers
    dm = instance.distance_matrix()

    # Initialize: one route per customer
    routes: list[list[int]] = [[c + 1] for c in range(n)]
    route_of: dict[int, int] = {c + 1: c for c in range(n)}  # customer -> route index

    # Compute savings
    savings = []
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            s = dm[0, i] + dm[0, j] - dm[i, j]
            savings.append((s, i, j))
    savings.sort(reverse=True)

    for s_val, i, j in savings:
        if s_val <= 0:
            break

        ri = route_of[i]
        rj = route_of[j]

        if ri == rj:
            continue  # same route

        route_i = routes[ri]
        route_j = routes[rj]

        if not route_i or not route_j:
            continue

        # Only merge if i is at the end of its route and j at the start (or vice versa)
        can_merge = False
        merged = None

        if route_i[-1] == i and route_j[0] == j:
            merged = route_i + route_j
            can_merge = True
        elif route_j[-1] == j and route_i[0] == i:
            merged = route_j + route_i
            can_merge = True
        elif route_i[-1] == i and route_j[-1] == j:
            merged = route_i + route_j[::-1]
            can_merge = True
        elif route_i[0] == i and route_j[0] == j:
            merged = route_i[::-1] + route_j
            can_merge = True

        if not can_merge:
            continue

        # Check chance constraint
        overflow_prob = instance.route_overflow_probability(merged)
        if overflow_prob > instance.alpha + 1e-9:
            continue

        # Check vehicle limit
        active_routes = sum(1 for r in routes if r)
        if active_routes - 1 < 0:
            continue

        # Merge: put merged into ri, clear rj
        routes[ri] = merged
        routes[rj] = []
        for c in merged:
            route_of[c] = ri

    # Collect non-empty routes
    final_routes = [r for r in routes if r]

    # Compute solution metrics
    total_dist = instance.solution_total_distance(final_routes)
    recourse = instance.expected_recourse_cost(final_routes)
    max_overflow = max(
        (instance.route_overflow_probability(r) for r in final_routes),
        default=0.0,
    )

    return StochasticVRPSolution(
        routes=final_routes,
        total_distance=total_dist,
        expected_total_cost=total_dist + recourse,
        max_overflow_prob=max_overflow,
        n_routes=len(final_routes),
    )


def mean_demand_savings(instance: StochasticVRPInstance) -> StochasticVRPSolution:
    """Clarke-Wright using mean demands as deterministic proxy.

    Merge routes as long as mean total demand <= capacity.

    Args:
        instance: StochasticVRPInstance.

    Returns:
        StochasticVRPSolution.
    """
    n = instance.n_customers
    dm = instance.distance_matrix()
    mean_d = instance.mean_demands

    routes: list[list[int]] = [[c + 1] for c in range(n)]
    route_of: dict[int, int] = {c + 1: c for c in range(n)}
    route_load: dict[int, float] = {c: mean_d[c] for c in range(n)}

    savings = []
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            s = dm[0, i] + dm[0, j] - dm[i, j]
            savings.append((s, i, j))
    savings.sort(reverse=True)

    for s_val, i, j in savings:
        if s_val <= 0:
            break

        ri = route_of[i]
        rj = route_of[j]
        if ri == rj:
            continue

        route_i = routes[ri]
        route_j = routes[rj]
        if not route_i or not route_j:
            continue

        combined_load = route_load[ri] + route_load[rj]
        if combined_load > instance.vehicle_capacity + 1e-9:
            continue

        # Merge at endpoints
        if route_i[-1] == i and route_j[0] == j:
            merged = route_i + route_j
        elif route_j[-1] == j and route_i[0] == i:
            merged = route_j + route_i
        elif route_i[-1] == i and route_j[-1] == j:
            merged = route_i + route_j[::-1]
        elif route_i[0] == i and route_j[0] == j:
            merged = route_i[::-1] + route_j
        else:
            continue

        routes[ri] = merged
        routes[rj] = []
        route_load[ri] = combined_load
        for c in merged:
            route_of[c] = ri

    final_routes = [r for r in routes if r]
    total_dist = instance.solution_total_distance(final_routes)
    recourse = instance.expected_recourse_cost(final_routes)
    max_overflow = max(
        (instance.route_overflow_probability(r) for r in final_routes),
        default=0.0,
    )

    return StochasticVRPSolution(
        routes=final_routes,
        total_distance=total_dist,
        expected_total_cost=total_dist + recourse,
        max_overflow_prob=max_overflow,
        n_routes=len(final_routes),
    )
