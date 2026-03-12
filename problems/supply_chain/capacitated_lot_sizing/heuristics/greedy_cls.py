"""
Greedy heuristics for the Capacitated Lot Sizing Problem.

Problem: Capacitated Lot Sizing (CLSP)
Complexity: O(T^2) for forward pass greedy, O(T) for lot-for-lot.

Implements:
1. Lot-for-lot: produce exactly the demand each period (baseline).
2. Forward pass greedy: greedily pull future demand into cheaper periods
   while respecting capacity.

References:
    Maes, J. & Van Wassenhove, L.N. (1988). Multi-item single-level
    capacitated dynamic lot-sizing heuristics: a general review.
    Journal of the Operational Research Society, 39(11), 991-1004.
    https://doi.org/10.1057/jors.1988.170

    Trigeiro, W.W., Thomas, L.J. & McClain, J.O. (1989). Capacitated
    lot sizing with setup times. Management Science, 35(3), 353-366.
    https://doi.org/10.1287/mnsc.35.3.353
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


def lot_for_lot(instance: CapLotSizingInstance) -> CapLotSizingSolution:
    """Lot-for-lot baseline: produce exactly demand each period.

    Requires capacity >= demand in each period.

    Args:
        instance: CapLotSizingInstance to solve.

    Returns:
        CapLotSizingSolution with lot-for-lot production plan.

    Raises:
        ValueError: If demand exceeds capacity in any period.
    """
    T = instance.T
    production = instance.demands.copy()

    for t in range(T):
        if production[t] > instance.capacities[t] + 1e-10:
            raise ValueError(
                f"Demand {production[t]} exceeds capacity "
                f"{instance.capacities[t]} in period {t}"
            )

    production_periods = [t for t in range(T) if production[t] > 1e-10]
    total_cost = instance.compute_cost(production)

    return CapLotSizingSolution(
        production=production,
        total_cost=total_cost,
        production_periods=production_periods,
    )


def forward_greedy(instance: CapLotSizingInstance) -> CapLotSizingSolution:
    """Forward pass greedy heuristic for CLSP.

    Starting from period 0, decides how much to produce in each period.
    Tries to produce ahead (pull future demand) when it saves cost
    (fixed cost saving exceeds additional holding and variable cost),
    subject to capacity constraints.

    Args:
        instance: CapLotSizingInstance to solve.

    Returns:
        CapLotSizingSolution with greedy production plan.
    """
    T = instance.T
    d = instance.demands
    cap = instance.capacities
    K = instance.fixed_costs
    c = instance.variable_costs
    h = instance.holding_costs

    production = np.zeros(T, dtype=float)
    remaining_demand = d.copy()

    for t in range(T):
        if remaining_demand[t] <= 1e-10:
            continue

        # Must produce at least the remaining demand for period t
        production[t] = remaining_demand[t]
        remaining_demand[t] = 0.0
        spare_capacity = cap[t] - production[t]

        # Try to pull future demand into period t
        for j in range(t + 1, T):
            if remaining_demand[j] <= 1e-10:
                continue
            if spare_capacity <= 1e-10:
                break

            # Cost of producing d[j] in period t instead of j:
            # Savings: K[j] (if we eliminate the order in period j)
            # Extra cost: variable cost diff + holding from t to j
            holding_cost_extra = 0.0
            for p in range(t, j):
                holding_cost_extra += h[p]
            holding_cost_extra *= remaining_demand[j]
            variable_cost_extra = (c[t] - c[j]) * remaining_demand[j]

            # Only pull if we'd save the fixed cost (and no other demand
            # in period j has already triggered that setup)
            # Simple check: is it cheaper to produce here?
            pull_amount = min(remaining_demand[j], spare_capacity)
            pull_holding = 0.0
            for p in range(t, j):
                pull_holding += h[p]
            pull_holding *= pull_amount
            pull_variable = (c[t] - c[j]) * pull_amount
            extra_cost = pull_holding + pull_variable

            # Pull if extra cost is less than the fixed cost we'd save
            # (conservative: only if we can pull ALL of period j's demand)
            if pull_amount >= remaining_demand[j] and extra_cost < K[j]:
                production[t] += pull_amount
                remaining_demand[j] -= pull_amount
                spare_capacity -= pull_amount

    # Any remaining demand that couldn't be pulled forward
    for t in range(T):
        if remaining_demand[t] > 1e-10:
            production[t] += remaining_demand[t]
            remaining_demand[t] = 0.0

    production_periods = [t for t in range(T) if production[t] > 1e-10]
    total_cost = instance.compute_cost(production)

    return CapLotSizingSolution(
        production=production,
        total_cost=total_cost,
        production_periods=production_periods,
    )


if __name__ == "__main__":
    inst = _inst.tight_capacity_6()
    sol_lfl = lot_for_lot(inst)
    sol_greedy = forward_greedy(inst)
    print(f"Lot-for-lot: {sol_lfl}")
    print(f"  production: {sol_lfl.production}")
    print(f"Forward greedy: {sol_greedy}")
    print(f"  production: {sol_greedy.production}")
