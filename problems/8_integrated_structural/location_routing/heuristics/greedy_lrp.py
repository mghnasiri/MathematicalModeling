"""
Sequential Greedy Heuristic for the Location-Routing Problem (LRP).

Problem: LRP (Location-Routing Problem)
Complexity: O(m*n + n^2) -- depot selection O(m*n), route building O(n^2)

Three-phase sequential approach:
  Phase 1 (Location): Open cheapest depots that cover all customer demand,
      assigning each customer to the nearest open depot.
  Phase 2 (Routing): For each open depot, build vehicle routes for its
      assigned customers using nearest-neighbor insertion.
  Phase 3 (Improvement): Reassign border customers between depots if it
      reduces total cost without violating capacity constraints.

References:
    Salhi, S. & Rand, G.K. (1989). The effect of ignoring routes when
    locating depots. European Journal of Operational Research, 39(2),
    150-156.
    https://doi.org/10.1016/0377-2217(89)90188-4

    Nagy, G. & Salhi, S. (2007). Location-routing: Issues, models and
    methods. European Journal of Operational Research, 177(2), 649-672.
    https://doi.org/10.1016/j.ejor.2006.04.004

    Prins, C., Prodhon, C. & Wolfler Calvo, R. (2006). Solving the
    capacitated location-routing problem by a cooperative Lagrangean
    relaxation-granular tabu search heuristic. Transportation Science,
    40(1), 18-32.
    https://doi.org/10.1287/trsc.1050.0126
"""

from __future__ import annotations

import os
import sys
import importlib.util

import numpy as np

_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_mod(name: str, filepath: str):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_inst = _load_mod("lrp_instance_gr", os.path.join(_parent_dir, "instance.py"))
LRPInstance = _inst.LRPInstance
LRPSolution = _inst.LRPSolution
compute_cost = _inst.compute_cost


def _build_nn_routes(
    instance: LRPInstance,
    depot_idx: int,
    customers: list[int],
) -> list[list[int]]:
    """Build nearest-neighbor routes for customers assigned to a depot.

    Creates multiple routes as needed to respect vehicle capacity Q.

    Args:
        instance: The LRP instance.
        depot_idx: Index of the depot (0-based).
        customers: List of customer indices (0-based) assigned to this depot.

    Returns:
        List of routes, each a list of customer indices.
    """
    if not customers:
        return []

    Q = instance.vehicle_capacity
    dist = instance.distance_matrix
    unvisited = set(customers)
    routes: list[list[int]] = []

    while unvisited:
        route: list[int] = []
        load = 0.0
        depot_node = instance.depot_node(depot_idx)

        # Start with nearest unvisited customer to depot
        nearest = min(
            unvisited,
            key=lambda c: dist[depot_node][instance.customer_node(c)],
        )
        route.append(nearest)
        load += instance.demands[nearest]
        unvisited.remove(nearest)

        # Extend route with nearest feasible customer
        while unvisited:
            last_node = instance.customer_node(route[-1])
            best_cust = None
            best_dist = float("inf")
            for c in unvisited:
                if load + instance.demands[c] <= Q + 1e-10:
                    d = dist[last_node][instance.customer_node(c)]
                    if d < best_dist:
                        best_dist = d
                        best_cust = c
            if best_cust is None:
                break
            route.append(best_cust)
            load += instance.demands[best_cust]
            unvisited.remove(best_cust)

        routes.append(route)

    return routes


def greedy_lrp(instance: LRPInstance) -> LRPSolution:
    """Solve LRP using a sequential greedy heuristic.

    Phase 1: Open depots greedily based on cost efficiency (fixed cost
        per unit demand that can be covered).
    Phase 2: Build nearest-neighbor routes per depot.
    Phase 3: Try reassigning border customers to improve total cost.

    Args:
        instance: An LRPInstance.

    Returns:
        LRPSolution with open depots, routes, and total cost.
    """
    m = instance.m
    n = instance.n
    dist = instance.distance_matrix

    # --- Phase 1: Open depots to cover all demand ---
    # Score depots by: fixed_cost / (capacity * average proximity to customers)
    open_depots: list[int] = []
    assigned: dict[int, int] = {}  # customer -> depot
    remaining_cap = np.zeros(m)
    covered_demand = 0.0
    total_demand = float(instance.demands.sum())

    # Compute average distance from each depot to all customers
    avg_dist_to_custs = np.zeros(m)
    for j in range(m):
        dnode = instance.depot_node(j)
        dists = [
            dist[dnode][instance.customer_node(i)]
            for i in range(n)
        ]
        avg_dist_to_custs[j] = np.mean(dists) if dists else float("inf")

    # Rank depots by efficiency: lower score is better
    depot_scores = np.zeros(m)
    for j in range(m):
        if avg_dist_to_custs[j] > 0:
            depot_scores[j] = instance.fixed_costs[j] / (
                instance.capacities[j] * (1.0 / avg_dist_to_custs[j])
            )
        else:
            depot_scores[j] = instance.fixed_costs[j]

    depot_order = np.argsort(depot_scores)

    # Open depots until all demand can be served
    total_open_cap = 0.0
    for j in depot_order:
        open_depots.append(int(j))
        remaining_cap[j] = instance.capacities[j]
        total_open_cap += instance.capacities[j]
        if total_open_cap >= total_demand:
            break

    # If still not enough capacity, open remaining depots
    if total_open_cap < total_demand:
        for j in range(m):
            if j not in open_depots:
                open_depots.append(j)
                remaining_cap[j] = instance.capacities[j]
                total_open_cap += instance.capacities[j]
                if total_open_cap >= total_demand:
                    break

    open_set = set(open_depots)

    # Assign each customer to nearest open depot with available capacity
    # Sort customers by distance to their nearest depot (farthest first)
    cust_nearest_dist = []
    for i in range(n):
        cnode = instance.customer_node(i)
        min_d = min(
            dist[instance.depot_node(j)][cnode]
            for j in open_depots
        )
        cust_nearest_dist.append((min_d, i))
    cust_nearest_dist.sort(reverse=True)

    for _, i in cust_nearest_dist:
        cnode = instance.customer_node(i)
        best_depot = None
        best_d = float("inf")
        for j in open_depots:
            if remaining_cap[j] >= instance.demands[i] - 1e-10:
                d = dist[instance.depot_node(j)][cnode]
                if d < best_d:
                    best_d = d
                    best_depot = j
        if best_depot is None:
            # Fallback: assign to depot with most remaining capacity
            best_depot = max(open_depots, key=lambda j: remaining_cap[j])
        assigned[i] = best_depot
        remaining_cap[best_depot] -= instance.demands[i]

    # --- Phase 2: Build routes per depot ---
    depot_customers: dict[int, list[int]] = {j: [] for j in open_depots}
    for i in range(n):
        depot_customers[assigned[i]].append(i)

    routes: dict[int, list[list[int]]] = {}
    for j in open_depots:
        routes[j] = _build_nn_routes(instance, j, depot_customers[j])

    # --- Phase 3: Border customer reassignment ---
    improved = True
    while improved:
        improved = False
        for i in range(n):
            current_depot = assigned[i]
            cnode = instance.customer_node(i)

            for j in open_depots:
                if j == current_depot:
                    continue
                # Check depot capacity
                current_j_demand = sum(
                    instance.demands[c]
                    for c in depot_customers[j]
                )
                if (
                    current_j_demand + instance.demands[i]
                    > instance.capacities[j] + 1e-10
                ):
                    continue

                # Estimate cost change: compare distances to depots
                old_d = dist[instance.depot_node(current_depot)][cnode]
                new_d = dist[instance.depot_node(j)][cnode]
                if new_d < old_d * 0.8:  # Significant improvement threshold
                    # Reassign
                    depot_customers[current_depot].remove(i)
                    depot_customers[j].append(i)
                    assigned[i] = j

                    # Rebuild routes for affected depots
                    routes[current_depot] = _build_nn_routes(
                        instance, current_depot,
                        depot_customers[current_depot],
                    )
                    routes[j] = _build_nn_routes(
                        instance, j, depot_customers[j],
                    )
                    improved = True
                    break

    # Remove depots with no customers
    final_depots = [j for j in open_depots if depot_customers[j]]
    final_routes = {j: routes[j] for j in final_depots}

    sol = LRPSolution(
        open_depots=sorted(final_depots),
        routes=final_routes,
        cost=0.0,
    )
    total, _, _ = compute_cost(instance, sol)
    sol.cost = total

    return sol


if __name__ == "__main__":
    from instance import small_lrp_3_8, medium_lrp_5_15

    print("=== Greedy LRP ===\n")

    for name, inst_fn in [
        ("small_3_8", small_lrp_3_8),
        ("medium_5_15", medium_lrp_5_15),
    ]:
        inst = inst_fn()
        sol = greedy_lrp(inst)
        print(f"{name}: {sol}")
        for d, rs in sol.routes.items():
            for r_idx, route in enumerate(rs):
                demand = inst.route_demand(route)
                dist_val = inst.route_distance(d, route)
                print(
                    f"  Depot {d}, Route {r_idx}: "
                    f"{route} (demand={demand:.0f}, dist={dist_val:.1f})"
                )
        print()
