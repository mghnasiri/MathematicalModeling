"""
Silver-Meal and Part-Period Balancing heuristics for Dynamic Lot Sizing.

Problem: Dynamic Lot Sizing (uncapacitated)

Silver-Meal: Greedily extends the order horizon until the average cost per
period starts increasing. Complexity: O(T^2) worst case, O(T) typical.

Part-Period Balancing (PPB): Orders such that cumulative holding cost
approximately equals the ordering cost. Complexity: O(T^2) worst case.

References:
    Silver, E.A. & Meal, H.C. (1973). A heuristic for selecting lot size
    quantities for the case of a deterministic time-varying demand rate
    and discrete opportunities for replenishment. Production and Inventory
    Management, 14(2), 64-74.

    DeMatteis, J.J. (1968). An economic lot-sizing technique I: The
    part-period algorithm. IBM Systems Journal, 7(1), 30-38.
    https://doi.org/10.1147/sj.71.0030
"""

from __future__ import annotations

import os
import sys
import importlib.util

import numpy as np

_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_mod(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_inst = _load_mod("ls_instance_sm", os.path.join(_parent_dir, "instance.py"))
LotSizingInstance = _inst.LotSizingInstance
LotSizingSolution = _inst.LotSizingSolution


def silver_meal(instance: LotSizingInstance) -> LotSizingSolution:
    """Silver-Meal heuristic for dynamic lot sizing.

    Starting from the first uncovered period, extend the order to cover
    additional periods as long as the average cost per period decreases.
    When it increases, place the order and move to the next uncovered period.

    Args:
        instance: LotSizingInstance to solve.

    Returns:
        LotSizingSolution with heuristic order quantities.
    """
    T = instance.T
    d = instance.demands
    K = instance.ordering_costs
    h = instance.holding_costs

    order_quantities = np.zeros(T, dtype=float)
    order_periods = []

    t = 0
    while t < T:
        # Start a new order at period t
        best_avg = float("inf")
        best_end = t

        total_cost = K[t]  # ordering cost
        holding_cost = 0.0

        for j in range(t, T):
            # Include demand d[j] in the current order
            # Holding cost for d[j]: carried from period t to period j
            if j > t:
                for p in range(t, j):
                    holding_cost += h[p] * d[j]

            total_cost_j = K[t] + holding_cost
            num_periods = j - t + 1
            avg_cost = total_cost_j / num_periods

            if avg_cost <= best_avg:
                best_avg = avg_cost
                best_end = j
            else:
                break

        # Place order at period t covering periods t..best_end
        order_qty = float(np.sum(d[t:best_end + 1]))
        order_quantities[t] = order_qty
        order_periods.append(t)

        t = best_end + 1

    total_cost = instance.compute_cost(order_quantities)

    return LotSizingSolution(
        order_quantities=order_quantities,
        total_cost=total_cost,
        order_periods=order_periods,
    )


def part_period_balancing(instance: LotSizingInstance) -> LotSizingSolution:
    """Part-Period Balancing (PPB) heuristic for dynamic lot sizing.

    Extends the order horizon until the cumulative holding cost equals
    or exceeds the ordering cost. The order covers the periods up to
    that point.

    Args:
        instance: LotSizingInstance to solve.

    Returns:
        LotSizingSolution with heuristic order quantities.
    """
    T = instance.T
    d = instance.demands
    K = instance.ordering_costs
    h = instance.holding_costs

    order_quantities = np.zeros(T, dtype=float)
    order_periods = []

    t = 0
    while t < T:
        holding_cost = 0.0
        best_end = t

        for j in range(t + 1, T):
            # Holding cost for d[j] carried from period t to period j
            for p in range(t, j):
                holding_cost += h[p] * d[j]

            if holding_cost >= K[t]:
                # Check which is closer: j-1 or j
                # Recompute holding for j-1
                h_prev = 0.0
                for k in range(t + 1, j):
                    for p in range(t, k):
                        h_prev += h[p] * d[k]

                if abs(h_prev - K[t]) <= abs(holding_cost - K[t]):
                    best_end = j - 1
                else:
                    best_end = j
                break
        else:
            best_end = T - 1

        order_qty = float(np.sum(d[t:best_end + 1]))
        order_quantities[t] = order_qty
        order_periods.append(t)

        t = best_end + 1

    total_cost = instance.compute_cost(order_quantities)

    return LotSizingSolution(
        order_quantities=order_quantities,
        total_cost=total_cost,
        order_periods=order_periods,
    )


def lot_for_lot(instance: LotSizingInstance) -> LotSizingSolution:
    """Lot-for-lot baseline: order exactly d[t] each period.

    This is the simplest policy with zero holding cost but maximum
    ordering cost. Useful as a baseline.

    Args:
        instance: LotSizingInstance to solve.

    Returns:
        LotSizingSolution with lot-for-lot order quantities.
    """
    T = instance.T
    d = instance.demands

    order_quantities = d.copy()
    order_periods = [t for t in range(T) if d[t] > 0]
    total_cost = instance.compute_cost(order_quantities)

    return LotSizingSolution(
        order_quantities=order_quantities,
        total_cost=total_cost,
        order_periods=order_periods,
    )


if __name__ == "__main__":
    inst = _inst.textbook_4period()
    sol_sm = silver_meal(inst)
    sol_ppb = part_period_balancing(inst)
    sol_lfl = lot_for_lot(inst)
    print(f"Silver-Meal: {sol_sm}")
    print(f"PPB: {sol_ppb}")
    print(f"Lot-for-lot: {sol_lfl}")
