"""
Simulated Annealing for the Location-Routing Problem (LRP).

Problem: LRP (Location-Routing Problem)

Neighborhoods:
- Toggle depot: open a closed depot or close an open one (if feasible)
- Reassign customer: move a customer to a different open depot
- Swap customers: exchange two customers between different depot routes
- 2-opt intra-route: reverse a segment within a single route

Warm-started with the sequential greedy heuristic.

References:
    Yu, V.F., Lin, S.W., Lee, W. & Ting, C.J. (2010). A simulated
    annealing heuristic for the capacitated location routing problem.
    Computers & Industrial Engineering, 58(2), 288-299.
    https://doi.org/10.1016/j.cie.2009.10.007

    Prins, C., Prodhon, C. & Wolfler Calvo, R. (2006). Solving the
    capacitated location-routing problem by a cooperative Lagrangean
    relaxation-granular tabu search heuristic. Transportation Science,
    40(1), 18-32.
    https://doi.org/10.1287/trsc.1050.0126

    Kirkpatrick, S., Gelatt, C.D. & Vecchi, M.P. (1983). Optimization
    by simulated annealing. Science, 220(4598), 671-680.
    https://doi.org/10.1126/science.220.4598.671
"""

from __future__ import annotations

import os
import sys
import math
import copy
import importlib.util

import numpy as np

_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_mod(name: str, filepath: str):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_inst = _load_mod("lrp_instance_sa", os.path.join(_parent_dir, "instance.py"))
LRPInstance = _inst.LRPInstance
LRPSolution = _inst.LRPSolution
compute_cost = _inst.compute_cost


def _total_cost(
    instance: LRPInstance,
    open_depots: list[int],
    routes: dict[int, list[list[int]]],
) -> float:
    """Compute total cost of a given solution state."""
    fixed = float(sum(instance.fixed_costs[d] for d in open_depots))
    routing = 0.0
    for d, depot_routes in routes.items():
        for route in depot_routes:
            routing += instance.route_distance(d, route)
    return fixed + routing


def _depot_demand(
    instance: LRPInstance,
    routes: dict[int, list[list[int]]],
    depot_idx: int,
) -> float:
    """Total demand assigned to a depot."""
    depot_routes = routes.get(depot_idx, [])
    return sum(instance.route_demand(r) for r in depot_routes)


def _rebuild_routes_nn(
    instance: LRPInstance,
    depot_idx: int,
    customers: list[int],
) -> list[list[int]]:
    """Build nearest-neighbor routes for a depot's customers."""
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

        nearest = min(
            unvisited,
            key=lambda c: dist[depot_node][instance.customer_node(c)],
        )
        route.append(nearest)
        load += instance.demands[nearest]
        unvisited.remove(nearest)

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


def _get_all_customers(
    routes: dict[int, list[list[int]]],
    depot_idx: int,
) -> list[int]:
    """Get all customers assigned to a depot."""
    result: list[int] = []
    for route in routes.get(depot_idx, []):
        result.extend(route)
    return result


def _two_opt(route: list[int], instance: LRPInstance, depot_idx: int) -> list[int]:
    """Apply 2-opt improvement to a single route.

    Args:
        route: List of customer indices (0-based).
        instance: The LRP instance.
        depot_idx: Depot index.

    Returns:
        Improved route.
    """
    if len(route) < 3:
        return route

    dist = instance.distance_matrix
    depot_node = instance.depot_node(depot_idx)
    improved = True
    best = list(route)

    while improved:
        improved = False
        for i in range(len(best) - 1):
            for j in range(i + 2, len(best)):
                # Nodes involved
                ni = instance.customer_node(best[i])
                ni1 = instance.customer_node(best[i + 1])
                nj = instance.customer_node(best[j])

                if j + 1 < len(best):
                    nj1 = instance.customer_node(best[j + 1])
                else:
                    nj1 = depot_node

                if i == 0:
                    prev_i = depot_node
                else:
                    prev_i = instance.customer_node(best[i - 1])

                # Current cost of edges
                old_cost = dist[ni][ni1]
                if j + 1 < len(best):
                    old_cost += dist[nj][nj1]
                else:
                    old_cost += dist[nj][depot_node]

                # New cost after reversal
                new_cost = dist[ni][nj]
                if j + 1 < len(best):
                    new_cost += dist[ni1][nj1]
                else:
                    new_cost += dist[ni1][depot_node]

                if new_cost < old_cost - 1e-10:
                    best[i + 1 : j + 1] = best[i + 1 : j + 1][::-1]
                    improved = True

    return best


def simulated_annealing(
    instance: LRPInstance,
    max_iterations: int = 20_000,
    initial_temp: float | None = None,
    cooling_rate: float = 0.9995,
    seed: int | None = None,
) -> LRPSolution:
    """Solve LRP using simulated annealing.

    Args:
        instance: An LRPInstance.
        max_iterations: Maximum number of SA iterations.
        initial_temp: Initial temperature. Auto-calibrated if None.
        cooling_rate: Geometric cooling factor per iteration.
        seed: Random seed for reproducibility.

    Returns:
        LRPSolution with open depots, routes, and total cost.
    """
    rng = np.random.default_rng(seed)
    m = instance.m
    n = instance.n

    # Warm-start with greedy heuristic
    _gr_mod = _load_mod(
        "lrp_gr_sa",
        os.path.join(_parent_dir, "heuristics", "greedy_lrp.py"),
    )
    init_sol = _gr_mod.greedy_lrp(instance)

    open_depots = list(init_sol.open_depots)
    routes = copy.deepcopy(init_sol.routes)
    current_cost = _total_cost(instance, open_depots, routes)

    best_depots = list(open_depots)
    best_routes = copy.deepcopy(routes)
    best_cost = current_cost

    if initial_temp is None:
        initial_temp = max(best_cost * 0.05, 1.0)

    temp = initial_temp

    for iteration in range(max_iterations):
        move_type = rng.integers(0, 4)
        new_depots = list(open_depots)
        new_routes = copy.deepcopy(routes)
        feasible = True

        if move_type == 0:
            # Toggle depot open/close
            depot = int(rng.integers(0, m))
            if depot in new_depots:
                # Try to close: reassign customers to other depots
                if len(new_depots) <= 1:
                    continue
                custs = _get_all_customers(new_routes, depot)
                other_depots = [d for d in new_depots if d != depot]
                if not other_depots:
                    continue

                # Try to redistribute customers
                reassignment: dict[int, list[int]] = {d: [] for d in other_depots}
                remaining_caps = {
                    d: instance.capacities[d] - _depot_demand(instance, new_routes, d)
                    for d in other_depots
                }
                for c in custs:
                    # Assign to nearest depot with capacity
                    cnode = instance.customer_node(c)
                    best_d = None
                    best_dist = float("inf")
                    for d in other_depots:
                        if remaining_caps[d] >= instance.demands[c] - 1e-10:
                            dd = instance.distance_matrix[
                                instance.depot_node(d)
                            ][cnode]
                            if dd < best_dist:
                                best_dist = dd
                                best_d = d
                    if best_d is None:
                        feasible = False
                        break
                    reassignment[best_d].append(c)
                    remaining_caps[best_d] -= instance.demands[c]

                if not feasible:
                    continue

                new_depots.remove(depot)
                del new_routes[depot]
                for d in other_depots:
                    if reassignment[d]:
                        existing = _get_all_customers(new_routes, d)
                        existing.extend(reassignment[d])
                        new_routes[d] = _rebuild_routes_nn(
                            instance, d, existing
                        )
            else:
                # Open the depot (no customers assigned yet)
                new_depots.append(depot)
                new_routes[depot] = []

        elif move_type == 1:
            # Reassign a random customer to a different depot
            if len(new_depots) < 2:
                continue
            cust = int(rng.integers(0, n))
            # Find current depot
            current_d = None
            for d in new_depots:
                if cust in _get_all_customers(new_routes, d):
                    current_d = d
                    break
            if current_d is None:
                continue

            # Pick a different depot
            other = [d for d in new_depots if d != current_d]
            if not other:
                continue
            target_d = int(rng.choice(other))

            # Check target depot capacity
            target_demand = _depot_demand(instance, new_routes, target_d)
            if (
                target_demand + instance.demands[cust]
                > instance.capacities[target_d] + 1e-10
            ):
                continue

            # Remove from current depot and rebuild
            old_custs = _get_all_customers(new_routes, current_d)
            old_custs.remove(cust)
            new_routes[current_d] = _rebuild_routes_nn(
                instance, current_d, old_custs
            )

            # Add to target depot and rebuild
            target_custs = _get_all_customers(new_routes, target_d)
            target_custs.append(cust)
            new_routes[target_d] = _rebuild_routes_nn(
                instance, target_d, target_custs
            )

        elif move_type == 2:
            # Swap two customers between different depots
            if len(new_depots) < 2:
                continue
            d1_idx = int(rng.integers(0, len(new_depots)))
            d2_idx = int(rng.integers(0, len(new_depots)))
            if d1_idx == d2_idx:
                continue
            d1 = new_depots[d1_idx]
            d2 = new_depots[d2_idx]

            custs1 = _get_all_customers(new_routes, d1)
            custs2 = _get_all_customers(new_routes, d2)
            if not custs1 or not custs2:
                continue

            c1 = int(rng.choice(custs1))
            c2 = int(rng.choice(custs2))

            # Check capacity feasibility after swap
            demand1 = _depot_demand(instance, new_routes, d1)
            demand2 = _depot_demand(instance, new_routes, d2)
            new_demand1 = demand1 - instance.demands[c1] + instance.demands[c2]
            new_demand2 = demand2 - instance.demands[c2] + instance.demands[c1]

            if (
                new_demand1 > instance.capacities[d1] + 1e-10
                or new_demand2 > instance.capacities[d2] + 1e-10
            ):
                continue

            # Perform swap
            custs1.remove(c1)
            custs1.append(c2)
            custs2.remove(c2)
            custs2.append(c1)
            new_routes[d1] = _rebuild_routes_nn(instance, d1, custs1)
            new_routes[d2] = _rebuild_routes_nn(instance, d2, custs2)

        else:
            # 2-opt within a random route
            if not new_depots:
                continue
            d = int(rng.choice(new_depots))
            depot_routes = new_routes.get(d, [])
            if not depot_routes:
                continue
            r_idx = int(rng.integers(0, len(depot_routes)))
            if len(depot_routes[r_idx]) < 3:
                continue
            new_routes[d][r_idx] = _two_opt(
                depot_routes[r_idx], instance, d
            )

        # Remove empty depots from route dict
        active_depots = [
            d for d in new_depots
            if _get_all_customers(new_routes, d)
        ]
        # Keep depots that have no customers only if just opened
        for d in new_depots:
            if d not in active_depots and d not in new_routes:
                pass  # Already not tracked

        new_cost = _total_cost(instance, new_depots, new_routes)
        delta = new_cost - current_cost

        if delta < 0 or (
            temp > 0 and rng.random() < math.exp(-delta / max(temp, 1e-15))
        ):
            open_depots = new_depots
            routes = new_routes
            current_cost = new_cost

            if current_cost < best_cost:
                best_cost = current_cost
                best_depots = list(open_depots)
                best_routes = copy.deepcopy(routes)

        temp *= cooling_rate

    # Clean up: remove depots with no customers
    final_depots = [
        d for d in best_depots if _get_all_customers(best_routes, d)
    ]
    final_routes = {d: best_routes[d] for d in final_depots if d in best_routes}

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

    print("=== Simulated Annealing for LRP ===\n")

    for name, inst_fn in [
        ("small_3_8", small_lrp_3_8),
        ("medium_5_15", medium_lrp_5_15),
    ]:
        inst = inst_fn()
        sol = simulated_annealing(inst, seed=42)
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
