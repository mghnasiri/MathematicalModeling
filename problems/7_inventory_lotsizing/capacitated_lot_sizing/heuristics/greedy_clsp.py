"""
Greedy Heuristic for Capacitated Lot Sizing Problem (CLSP).

Problem: CLSP (Capacitated Lot Sizing)
Complexity: O(T^2) — forward pass with greedy lookahead

Strategy: Process periods left to right. In each period, produce as much
as possible (up to capacity) to cover future demand, consolidating orders
when the holding cost is less than the fixed setup cost.

References:
    Pochet, Y. & Wolsey, L.A. (2006). Production Planning by Mixed
    Integer Programming. Springer, New York.
    https://doi.org/10.1007/0-387-33477-7
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


_inst = _load_mod("cls_instance_greedy", os.path.join(_parent_dir, "instance.py"))
CapLotSizingInstance = _inst.CapLotSizingInstance
CapLotSizingSolution = _inst.CapLotSizingSolution


def greedy_lot_sizing(instance: CapLotSizingInstance) -> CapLotSizingSolution:
    """Greedy heuristic for capacitated lot sizing.

    Processes periods left to right. In each period with unmet demand,
    produce up to capacity, covering as many future periods as profitable
    (holding cost vs. future setup cost savings).

    Args:
        instance: A CapLotSizingInstance.

    Returns:
        CapLotSizingSolution with greedy production plan.
    """
    T = instance.T
    d = instance.demands.copy()
    C = instance.capacities
    K = instance.fixed_costs
    h = instance.holding_costs
    v = instance.variable_costs

    production = np.zeros(T, dtype=float)
    remaining_demand = d.copy()

    for t in range(T):
        if remaining_demand[t] < 1e-10:
            continue

        # Produce at least enough for period t
        produce = min(remaining_demand[t], C[t])
        remaining_demand[t] -= produce

        # Try to cover future periods if savings exceed holding costs
        capacity_left = C[t] - produce
        for future in range(t + 1, T):
            if capacity_left < 1e-10:
                break
            if remaining_demand[future] < 1e-10:
                continue

            # Holding cost to carry from t to future
            hold_cost = 0.0
            for p in range(t, future):
                hold_cost += h[p]
            extra_hold = hold_cost * remaining_demand[future]

            # Variable cost difference
            var_diff = (v[t] - v[future]) * remaining_demand[future]

            # Benefit: save future setup cost (if this would be the only order)
            # Heuristic: cover future if holding + var_diff < fixed cost
            if extra_hold + var_diff < K[future]:
                cover = min(remaining_demand[future], capacity_left)
                produce += cover
                remaining_demand[future] -= cover
                capacity_left -= cover

        production[t] = produce

    # Handle any remaining unmet demand (produce in earliest feasible period)
    for t in range(T):
        if remaining_demand[t] > 1e-10:
            # Find a period with remaining capacity
            for s in range(t + 1):
                spare = C[s] - production[s]
                if spare > 1e-10:
                    cover = min(remaining_demand[t], spare)
                    production[s] += cover
                    remaining_demand[t] -= cover
                    if remaining_demand[t] < 1e-10:
                        break
            # Also try future periods
            if remaining_demand[t] > 1e-10:
                for s in range(t + 1, T):
                    spare = C[s] - production[s]
                    if spare > 1e-10:
                        cover = min(remaining_demand[t], spare)
                        production[s] += cover
                        remaining_demand[t] -= cover
                        if remaining_demand[t] < 1e-10:
                            break

    production_periods = [t for t in range(T) if production[t] > 1e-10]
    total_cost = instance.compute_cost(production)

    return CapLotSizingSolution(
        production=production,
        total_cost=total_cost,
        production_periods=production_periods,
    )


def lot_for_lot(instance: CapLotSizingInstance) -> CapLotSizingSolution:
    """Lot-for-lot: produce exactly what is needed each period.

    Simple baseline — produces demand[t] in period t if capacity allows.

    Args:
        instance: A CapLotSizingInstance.

    Returns:
        CapLotSizingSolution with lot-for-lot plan.
    """
    T = instance.T
    production = np.minimum(instance.demands, instance.capacities)

    production_periods = [t for t in range(T) if production[t] > 1e-10]
    total_cost = instance.compute_cost(production)

    return CapLotSizingSolution(
        production=production,
        total_cost=total_cost,
        production_periods=production_periods,
    )


if __name__ == "__main__":
    inst = _inst.tight_capacity_6()
    sol = greedy_lot_sizing(inst)
    print(f"Greedy on {inst.name}: {sol}")

    sol2 = lot_for_lot(inst)
    print(f"Lot-for-lot on {inst.name}: {sol2}")
