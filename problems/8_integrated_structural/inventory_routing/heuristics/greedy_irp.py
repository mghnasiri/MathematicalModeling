"""
Greedy Constructive Heuristic for the Inventory-Routing Problem

At each period, identifies customers at risk of stockout before the
next replenishment opportunity, builds delivery routes using nearest-
neighbor insertion, and delivers enough to prevent stockout (up to
storage capacity if vehicle capacity allows).

Complexity: O(T * n^2) -- for each of T periods, nearest-neighbor
route construction over up to n customers.

References:
    - Campbell, A., Clarke, L., Kleywegt, A. & Savelsbergh, M. (1998).
      The inventory routing problem. In: Fleet Management and Logistics,
      pp. 95-113. Springer.
      https://doi.org/10.1007/978-1-4615-5755-5_4
    - Bertazzi, L., Savelsbergh, M. & Speranza, M.G. (2008).
      Inventory routing. In: The Vehicle Routing Problem: Latest
      Advances and New Challenges, pp. 49-72. Springer.
      https://doi.org/10.1007/978-0-387-77778-8_3
"""
from __future__ import annotations

import sys
import os
import importlib.util

import numpy as np


def _load_parent(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_base = os.path.dirname(os.path.dirname(__file__))
_inst = _load_parent("irp_instance", os.path.join(_base, "instance.py"))

IRPInstance = _inst.IRPInstance
IRPSolution = _inst.IRPSolution
compute_cost = _inst.compute_cost


def _compute_delivery_qty(
    inventory_c: float,
    demand_c: float,
    capacity_c: float,
    remaining_vehicle_cap: float,
) -> float:
    """Compute delivery quantity for a customer.

    Delivers the minimum of: what fills to storage capacity, and what
    the vehicle can carry. Ensures at least enough to cover one period
    of demand (to prevent imminent stockout).

    Args:
        inventory_c: Current inventory (before this period's demand).
        demand_c: Per-period demand.
        capacity_c: Storage capacity.
        remaining_vehicle_cap: Remaining capacity on the vehicle.

    Returns:
        Delivery quantity (>= 0).
    """
    # After demand, inventory would be:
    projected = inventory_c - demand_c

    # Gap to fill to capacity
    gap = capacity_c - projected
    gap = max(gap, 0.0)

    # Minimum needed: enough so post-delivery post-demand inventory >= 0
    min_needed = max(-projected, 0.0)

    # Try to fill to capacity, but at least deliver min_needed
    delivery = min(gap, remaining_vehicle_cap)

    # If can't even deliver minimum needed, deliver what we can
    if delivery < min_needed:
        delivery = min(min_needed, remaining_vehicle_cap)

    return max(delivery, 0.0)


def greedy_irp(instance: IRPInstance, seed: int = 42) -> IRPSolution:
    """Greedy constructive heuristic for IRP.

    Strategy:
        1. At each period t, project inventory levels forward.
        2. Identify customers that would stockout (projected inventory < 0)
           or become critically low (projected < demand).
        3. Build routes prioritizing critical customers, using nearest-
           neighbor insertion and respecting vehicle capacity.
        4. Deliver as much as possible (up to storage capacity), with a
           minimum delivery to prevent stockout.

    Args:
        instance: IRPInstance to solve.
        seed: Random seed (used for tie-breaking).

    Returns:
        IRPSolution with routes, deliveries, and costs.
    """
    rng = np.random.default_rng(seed)
    n = instance.n_customers
    T = instance.T
    dm = instance.distance_matrix()

    # Track inventory over time
    inventory = instance.initial_inventory.copy()

    routes_per_period: list[list[list[int]]] = []
    deliveries_per_period: list[dict[int, float]] = []

    for t in range(T):
        projected = inventory - instance.demands

        # Classify customers by urgency
        # Critical: would stockout this period (projected < 0)
        # Urgent: low inventory, within 1 demand period of stockout
        # Optional: could benefit from delivery but not urgent
        critical: list[int] = []
        urgent: list[int] = []
        optional: list[int] = []

        for c in range(n):
            if projected[c] < -1e-9:
                critical.append(c + 1)
            elif projected[c] < instance.demands[c]:
                urgent.append(c + 1)
            elif projected[c] < 2 * instance.demands[c]:
                optional.append(c + 1)

        # Sort each group by urgency (lowest projected first)
        critical.sort(key=lambda c: projected[c - 1])
        urgent.sort(key=lambda c: projected[c - 1])
        optional.sort(key=lambda c: projected[c - 1])

        # Customers to serve: critical first, then urgent, then optional
        to_serve = critical + urgent

        if not to_serve and not optional:
            routes_per_period.append([])
            deliveries_per_period.append({})
            inventory = np.maximum(projected, 0.0)
            continue

        # Build routes using nearest-neighbor with vehicle capacity
        period_routes: list[list[int]] = []
        period_deliveries: dict[int, float] = {}
        assigned: set[int] = set()

        # Phase 1: Serve critical and urgent customers
        remaining = list(to_serve)

        for _v in range(instance.n_vehicles):
            if not remaining:
                break

            route: list[int] = []
            route_load = 0.0
            current_node = 0

            # Keep trying to add customers to this route
            stalled = False
            while remaining and not stalled:
                best_cust = -1
                best_dist = float("inf")
                best_qty = 0.0

                for cust in remaining:
                    if cust in assigned:
                        continue

                    qty = _compute_delivery_qty(
                        inventory[cust - 1],
                        instance.demands[cust - 1],
                        instance.storage_capacities[cust - 1],
                        instance.vehicle_capacity - route_load,
                    )

                    # Must deliver at least enough to avoid stockout
                    min_needed = max(-(projected[cust - 1]), 0.0)
                    if qty < min_needed - 1e-9:
                        # Can't fit minimum delivery, skip for this route
                        continue

                    if qty < 1e-9:
                        continue

                    d = dm[current_node, cust]
                    if d < best_dist:
                        best_dist = d
                        best_cust = cust
                        best_qty = qty

                if best_cust == -1:
                    stalled = True
                    break

                route.append(best_cust)
                assigned.add(best_cust)
                remaining = [c for c in remaining if c not in assigned]
                period_deliveries[best_cust] = best_qty
                route_load += best_qty
                current_node = best_cust

            if route:
                period_routes.append(route)

        # Phase 2: Try to add optional customers to existing routes
        for route in period_routes:
            route_load = sum(period_deliveries.get(c, 0.0) for c in route)
            current_node = route[-1] if route else 0

            for c in optional:
                if c in assigned:
                    continue

                remaining_cap = instance.vehicle_capacity - route_load
                if remaining_cap < instance.demands[c - 1]:
                    continue

                qty = _compute_delivery_qty(
                    inventory[c - 1],
                    instance.demands[c - 1],
                    instance.storage_capacities[c - 1],
                    remaining_cap,
                )
                if qty < instance.demands[c - 1]:
                    continue

                # Accept if detour is reasonable
                detour = (
                    dm[current_node, c] + dm[c, 0] - dm[current_node, 0]
                )
                holding_savings = qty * instance.holding_costs[c - 1]
                if detour < holding_savings * 2:
                    route.append(c)
                    assigned.add(c)
                    period_deliveries[c] = qty
                    route_load += qty
                    current_node = c

        # Phase 3: Any critical customer still unserved gets an extra route
        # This ensures feasibility even if K vehicles were not enough
        unserved_critical = [c for c in critical if c not in assigned]
        if unserved_critical:
            for c in unserved_critical:
                min_needed = max(-(projected[c - 1]), 0.0)
                qty = min(
                    min_needed,
                    instance.storage_capacities[c - 1] - projected[c - 1],
                )
                qty = max(qty, min_needed)
                period_routes.append([c])
                period_deliveries[c] = qty
                assigned.add(c)

        routes_per_period.append(period_routes)
        deliveries_per_period.append(period_deliveries)

        # Update inventory: add deliveries, subtract demand
        for cust, qty in period_deliveries.items():
            inventory[cust - 1] += qty
        inventory -= instance.demands
        inventory = np.maximum(inventory, 0.0)

    # Compute final cost
    return compute_cost(instance, routes_per_period, deliveries_per_period)


def greedy_fill_up(instance: IRPInstance, seed: int = 42) -> IRPSolution:
    """Simple fill-up policy: visit customers every period, fill to cap.

    A baseline policy that delivers to every customer that needs product
    each period, filling them to storage capacity. Routes are built using
    nearest-neighbor from the depot. If a customer cannot fit on any
    vehicle, a dedicated route is created to prevent stockout.

    Args:
        instance: IRPInstance to solve.
        seed: Random seed (unused, for interface compatibility).

    Returns:
        IRPSolution with routes, deliveries, and costs.
    """
    n = instance.n_customers
    T = instance.T
    dm = instance.distance_matrix()

    inventory = instance.initial_inventory.copy()

    routes_per_period: list[list[list[int]]] = []
    deliveries_per_period: list[dict[int, float]] = []

    for t in range(T):
        projected = inventory - instance.demands
        period_deliveries: dict[int, float] = {}
        customers_to_visit: list[int] = []

        for c in range(n):
            gap = instance.storage_capacities[c] - projected[c]
            if gap > 1e-9 and projected[c] < instance.demands[c]:
                period_deliveries[c + 1] = gap
                customers_to_visit.append(c + 1)

        # Build routes with nearest-neighbor
        period_routes: list[list[int]] = []
        visited: set[int] = set()

        for _ in range(instance.n_vehicles):
            remaining = [c for c in customers_to_visit if c not in visited]
            if not remaining:
                break

            route: list[int] = []
            route_load = 0.0
            current_node = 0

            while remaining:
                best_cust = -1
                best_dist = float("inf")

                for cust in remaining:
                    qty = period_deliveries[cust]
                    if route_load + qty > instance.vehicle_capacity + 1e-9:
                        continue
                    d = dm[current_node, cust]
                    if d < best_dist:
                        best_dist = d
                        best_cust = cust

                if best_cust == -1:
                    break

                route.append(best_cust)
                visited.add(best_cust)
                remaining = [c for c in remaining if c not in visited]
                route_load += period_deliveries[best_cust]
                current_node = best_cust

            if route:
                period_routes.append(route)

        # Ensure all customers that would stockout are served
        for c in customers_to_visit:
            if c not in visited:
                if projected[c - 1] < -1e-9:
                    # Critical: must serve to prevent stockout
                    min_needed = max(-(projected[c - 1]), 0.0)
                    period_deliveries[c] = min_needed
                    period_routes.append([c])
                    visited.add(c)
                else:
                    # Not critical, skip and reduce delivery
                    del period_deliveries[c]

        routes_per_period.append(period_routes)
        deliveries_per_period.append(period_deliveries)

        # Update inventory
        for cust, qty in period_deliveries.items():
            inventory[cust - 1] += qty
        inventory -= instance.demands
        inventory = np.maximum(inventory, 0.0)

    return compute_cost(instance, routes_per_period, deliveries_per_period)


if __name__ == "__main__":
    inst = IRPInstance.small_instance()
    print(f"Instance: {inst.name}")
    print(f"  Customers: {inst.n_customers}, Periods: {inst.T}")
    print()

    sol_greedy = greedy_irp(inst)
    print(f"Greedy IRP: {sol_greedy}")
    for t in range(inst.T):
        print(f"  Period {t}: routes={sol_greedy.routes_per_period[t]}, "
              f"deliveries={sol_greedy.deliveries_per_period[t]}")
    print()

    sol_fillup = greedy_fill_up(inst)
    print(f"Fill-up:    {sol_fillup}")
