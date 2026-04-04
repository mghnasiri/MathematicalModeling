"""
Simulated Annealing for the Inventory-Routing Problem

Uses neighborhood moves on the period-route-delivery structure:
  1. Reassign: move a customer delivery from one period to another.
  2. Quantity change: adjust delivery quantity for a customer-period pair.
  3. Swap: exchange two customers between routes in the same period.
  4. Intra-route 2-opt: reverse a segment within a route.

The objective is total routing cost + total holding cost, with a penalty
for inventory violations (stockouts or capacity overflows).

Complexity: O(max_iter * T * n) per iteration for evaluation.

References:
    - Coelho, L.C., Cordeau, J.-F. & Laporte, G. (2012). Consistency
      in multi-vehicle inventory-routing. Transp. Res. Part C, 24, 270-287.
      https://doi.org/10.1016/j.trc.2012.03.007
    - Osman, I.H. & Potts, C.N. (1989). Simulated annealing for
      permutation flow-shop scheduling. Omega, 17(6), 551-557.
      https://doi.org/10.1016/0305-0483(89)90059-5
"""
from __future__ import annotations

import sys
import os
import importlib.util
import math
import copy

import numpy as np


def _load_parent(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_base = os.path.dirname(os.path.dirname(__file__))
_inst = _load_parent("irp_instance_sa", os.path.join(_base, "instance.py"))
_heur = _load_parent("irp_greedy_sa", os.path.join(
    _base, "heuristics", "greedy_irp.py"))

IRPInstance = _inst.IRPInstance
IRPSolution = _inst.IRPSolution
compute_cost = _inst.compute_cost
greedy_irp = _heur.greedy_irp


def _evaluate(
    instance: IRPInstance,
    routes_per_period: list[list[list[int]]],
    deliveries_per_period: list[dict[int, float]],
    penalty_weight: float = 500.0,
) -> tuple[float, float, float, bool]:
    """Evaluate a candidate solution with penalty for violations.

    Simulates inventory dynamics. Returns (total_penalized_cost,
    routing_cost, holding_cost, is_feasible).

    Args:
        instance: IRPInstance.
        routes_per_period: Routes per period.
        deliveries_per_period: Deliveries per period.
        penalty_weight: Weight for constraint violations.

    Returns:
        Tuple of (penalized_cost, routing_cost, holding_cost, feasible).
    """
    n = instance.n_customers
    T = instance.T

    inventory = instance.initial_inventory.copy()
    total_routing = 0.0
    total_holding = 0.0
    total_penalty = 0.0
    feasible = True

    for t in range(T):
        # Deliveries
        deliveries = deliveries_per_period[t] if t < len(deliveries_per_period) else {}
        for cust, qty in deliveries.items():
            inventory[cust - 1] += qty

        # Check capacity overflow
        over_cap = inventory - instance.storage_capacities
        mask_over = over_cap > 1e-9
        if np.any(mask_over):
            total_penalty += penalty_weight * float(over_cap[mask_over].sum())
            inventory = np.minimum(inventory, instance.storage_capacities)
            feasible = False

        # Subtract demand
        inventory -= instance.demands

        # Check stockout
        mask_stock = inventory < -1e-9
        if np.any(mask_stock):
            total_penalty += penalty_weight * float((-inventory[mask_stock]).sum())
            inventory = np.maximum(inventory, 0.0)
            feasible = False

        # Holding cost
        total_holding += float(np.dot(inventory, instance.holding_costs))

        # Routing cost
        period_routes = routes_per_period[t] if t < len(routes_per_period) else []
        for route in period_routes:
            total_routing += instance.route_distance(route)

    penalized = total_routing + total_holding + total_penalty
    return penalized, total_routing, total_holding, feasible


def _rebuild_routes_nn(
    instance: IRPInstance,
    customers: list[int],
    deliveries: dict[int, float],
) -> list[list[int]]:
    """Build routes for a set of customers using nearest-neighbor.

    Args:
        instance: IRPInstance.
        customers: List of customer indices (1-indexed) to route.
        deliveries: Delivery quantities for each customer.

    Returns:
        List of routes, each respecting vehicle capacity.
    """
    if not customers:
        return []

    dm = instance.distance_matrix()
    routes: list[list[int]] = []
    remaining = list(customers)

    for _ in range(instance.n_vehicles):
        if not remaining:
            break

        route: list[int] = []
        route_load = 0.0
        current = 0

        while remaining:
            best_c = -1
            best_d = float("inf")

            for c in remaining:
                qty = deliveries.get(c, 0.0)
                if route_load + qty > instance.vehicle_capacity + 1e-9:
                    continue
                if dm[current, c] < best_d:
                    best_d = dm[current, c]
                    best_c = c

            if best_c == -1:
                break

            route.append(best_c)
            route_load += deliveries.get(best_c, 0.0)
            remaining.remove(best_c)
            current = best_c

        if route:
            routes.append(route)

    # If remaining customers cannot fit, add single-customer routes
    for c in remaining:
        routes.append([c])

    return routes


def simulated_annealing(
    instance: IRPInstance,
    max_iterations: int = 5000,
    initial_temp: float = 100.0,
    cooling_rate: float = 0.995,
    penalty_weight: float = 500.0,
    seed: int = 42,
) -> IRPSolution:
    """Simulated Annealing for the Inventory-Routing Problem.

    Starts from a greedy solution and applies neighborhood moves:
    reassign customer-period, adjust delivery quantity, swap customers
    in routes, and intra-route 2-opt.

    Args:
        instance: IRPInstance to solve.
        max_iterations: Number of SA iterations.
        initial_temp: Starting temperature.
        cooling_rate: Geometric cooling factor (0 < cooling_rate < 1).
        penalty_weight: Penalty multiplier for constraint violations.
        seed: Random seed for determinism.

    Returns:
        Best feasible IRPSolution found.
    """
    rng = np.random.default_rng(seed)
    n = instance.n_customers
    T = instance.T

    # Initialize with greedy heuristic
    init_sol = greedy_irp(instance, seed=seed)
    current_routes = [
        [list(r) for r in period_routes]
        for period_routes in init_sol.routes_per_period
    ]
    current_deliveries = [dict(d) for d in init_sol.deliveries_per_period]

    current_obj, cur_routing, cur_holding, cur_feas = _evaluate(
        instance, current_routes, current_deliveries, penalty_weight
    )

    best_routes = copy.deepcopy(current_routes)
    best_deliveries = copy.deepcopy(current_deliveries)
    best_obj = current_obj
    best_routing = cur_routing
    best_holding = cur_holding
    best_feasible = cur_feas

    temp = initial_temp

    for iteration in range(max_iterations):
        # Deep copy current solution
        new_routes = copy.deepcopy(current_routes)
        new_deliveries = [dict(d) for d in current_deliveries]

        move = rng.random()

        if move < 0.35:
            # Move 1: Reassign — move a customer delivery between periods
            # Pick a period with deliveries
            periods_with_del = [
                t for t in range(T) if new_deliveries[t]
            ]
            if periods_with_del:
                t_from = rng.choice(periods_with_del)
                custs = list(new_deliveries[t_from].keys())
                if custs:
                    cust = rng.choice(custs)
                    qty = new_deliveries[t_from][cust]

                    # Pick target period (could be any other)
                    t_to = rng.integers(T)
                    if t_to != t_from:
                        # Remove from source period
                        del new_deliveries[t_from][cust]
                        # Remove from source routes
                        for route in new_routes[t_from]:
                            if cust in route:
                                route.remove(cust)
                                break
                        # Clean empty routes
                        new_routes[t_from] = [
                            r for r in new_routes[t_from] if r
                        ]

                        # Add to target period
                        new_deliveries[t_to][cust] = qty
                        # Rebuild routes for target period
                        customers_t = list(new_deliveries[t_to].keys())
                        new_routes[t_to] = _rebuild_routes_nn(
                            instance, customers_t, new_deliveries[t_to]
                        )

        elif move < 0.60:
            # Move 2: Adjust delivery quantity
            periods_with_del = [
                t for t in range(T) if new_deliveries[t]
            ]
            if periods_with_del:
                t = rng.choice(periods_with_del)
                custs = list(new_deliveries[t].keys())
                if custs:
                    cust = rng.choice(custs)
                    old_qty = new_deliveries[t][cust]
                    # Perturb by +/- up to 30%
                    delta = rng.uniform(-0.3, 0.3) * old_qty
                    new_qty = max(
                        instance.demands[cust - 1],
                        min(
                            old_qty + delta,
                            instance.storage_capacities[cust - 1],
                        ),
                    )
                    new_deliveries[t][cust] = new_qty

        elif move < 0.80:
            # Move 3: Swap two customers in routes within a period
            t = rng.integers(T)
            if len(new_routes[t]) >= 1:
                routes_with_custs = [
                    (ri, r) for ri, r in enumerate(new_routes[t]) if len(r) >= 1
                ]
                if len(routes_with_custs) >= 2:
                    idxs = rng.choice(len(routes_with_custs), 2, replace=False)
                    ri1, r1 = routes_with_custs[idxs[0]]
                    ri2, r2 = routes_with_custs[idxs[1]]
                    p1 = rng.integers(len(r1))
                    p2 = rng.integers(len(r2))
                    new_routes[t][ri1][p1], new_routes[t][ri2][p2] = (
                        new_routes[t][ri2][p2],
                        new_routes[t][ri1][p1],
                    )
                elif len(routes_with_custs) == 1:
                    ri, r = routes_with_custs[0]
                    if len(r) >= 2:
                        p1, p2 = rng.choice(len(r), 2, replace=False)
                        r[p1], r[p2] = r[p2], r[p1]

        else:
            # Move 4: Intra-route 2-opt
            t = rng.integers(T)
            long_routes = [
                (ri, r) for ri, r in enumerate(new_routes[t]) if len(r) >= 3
            ]
            if long_routes:
                ri, route = long_routes[rng.integers(len(long_routes))]
                i = rng.integers(len(route) - 1)
                j = rng.integers(i + 1, len(route))
                route[i:j + 1] = route[i:j + 1][::-1]

        # Evaluate neighbor
        new_obj, new_routing, new_holding, new_feas = _evaluate(
            instance, new_routes, new_deliveries, penalty_weight
        )

        delta = new_obj - current_obj

        if delta < 0 or rng.random() < math.exp(-delta / max(temp, 1e-10)):
            current_routes = new_routes
            current_deliveries = new_deliveries
            current_obj = new_obj

            # Update best if improved and feasible (or first feasible)
            if new_feas and (not best_feasible or new_obj < best_obj):
                best_routes = copy.deepcopy(new_routes)
                best_deliveries = copy.deepcopy(new_deliveries)
                best_obj = new_obj
                best_routing = new_routing
                best_holding = new_holding
                best_feasible = True
            elif not best_feasible and new_obj < best_obj:
                best_routes = copy.deepcopy(new_routes)
                best_deliveries = copy.deepcopy(new_deliveries)
                best_obj = new_obj
                best_routing = new_routing
                best_holding = new_holding

        temp *= cooling_rate

    # Build final solution using compute_cost (uses soft penalty for stockouts)
    return compute_cost(instance, best_routes, best_deliveries)


if __name__ == "__main__":
    inst = IRPInstance.small_instance()
    print(f"Instance: {inst.name}")
    print(f"  Customers: {inst.n_customers}, Periods: {inst.T}")
    print()

    sol_greedy = greedy_irp(inst)
    print(f"Greedy: {sol_greedy}")

    sol_sa = simulated_annealing(inst, max_iterations=3000, seed=42)
    print(f"SA:     {sol_sa}")

    if sol_sa.total_cost < sol_greedy.total_cost:
        improvement = (
            (sol_greedy.total_cost - sol_sa.total_cost)
            / sol_greedy.total_cost * 100
        )
        print(f"SA improved by {improvement:.1f}%")
