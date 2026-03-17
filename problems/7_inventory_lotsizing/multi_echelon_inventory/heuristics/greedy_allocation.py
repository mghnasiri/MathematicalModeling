"""
Greedy Allocation Heuristic for Multi-Echelon Inventory.

Problem: Serial multi-echelon inventory with base-stock policies.
Complexity: O(E) where E is the number of echelons.

For each echelon, compute the safety stock needed to cover demand
variability over its lead time, using the normal distribution
approximation. The base-stock level is then mean demand over lead
time plus safety stock.

References:
    Clark, A.J. & Scarf, H. (1960). Optimal policies for a multi-echelon
    inventory problem. Management Science, 6(4), 475-490.
    https://doi.org/10.1287/mnsc.6.4.475

    Simchi-Levi, D. & Zhao, Y. (2012). Performance evaluation of
    stochastic multi-echelon inventory systems: A survey. Advances in
    Operations Research, 2012, 126254.
"""

from __future__ import annotations

import os
import sys
import importlib.util
import math

import numpy as np

_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_mod(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_inst = _load_mod("mei_instance_greedy", os.path.join(_parent_dir, "instance.py"))
MultiEchelonInstance = _inst.MultiEchelonInstance
MultiEchelonSolution = _inst.MultiEchelonSolution


def _z_score(service_level: float) -> float:
    """Approximate z-score for a given service level using inverse normal.

    Args:
        service_level: Target probability (0 < p < 1).

    Returns:
        z such that P(Z <= z) = service_level for standard normal Z.
    """
    from scipy.stats import norm
    return float(norm.ppf(service_level))


def greedy_base_stock(instance: MultiEchelonInstance) -> MultiEchelonSolution:
    """Compute base-stock levels using independent safety stock allocation.

    For each echelon e, safety stock = z * sigma * sqrt(L_e), where L_e is
    the lead time at echelon e. Base-stock = mu * L_e + safety_stock.

    This is a simple heuristic that treats echelons independently.

    Args:
        instance: A MultiEchelonInstance.

    Returns:
        MultiEchelonSolution with base-stock levels and costs.
    """
    z = _z_score(instance.service_level)
    mu = instance.mean_demand
    sigma = instance.std_demand

    base_stocks = np.zeros(instance.n_echelons)
    safety_stocks = np.zeros(instance.n_echelons)

    for e in range(instance.n_echelons):
        L = instance.lead_times[e]
        ss = z * sigma * math.sqrt(L)
        safety_stocks[e] = ss
        base_stocks[e] = mu * L + ss

    # Expected holding cost = sum of h_e * E[inventory_e]
    # E[inventory_e] ~ safety_stock_e (for base-stock policy)
    total_cost = float(np.sum(instance.holding_costs * safety_stocks))

    return MultiEchelonSolution(
        base_stock_levels=base_stocks,
        safety_stocks=safety_stocks,
        total_holding_cost=total_cost,
    )


def echelon_stock(instance: MultiEchelonInstance) -> MultiEchelonSolution:
    """Compute echelon base-stock levels using cumulative lead times.

    Uses echelon stock concept: echelon e covers cumulative lead time
    from echelon e up to (and including) downstream echelons.

    Args:
        instance: A MultiEchelonInstance.

    Returns:
        MultiEchelonSolution with echelon-based stock levels.
    """
    z = _z_score(instance.service_level)
    mu = instance.mean_demand
    sigma = instance.std_demand
    E = instance.n_echelons

    # Cumulative lead time from echelon e to end customer (echelon 0)
    cum_lead = np.zeros(E)
    cum_lead[0] = instance.lead_times[0]
    for e in range(1, E):
        cum_lead[e] = cum_lead[e - 1] + instance.lead_times[e]

    base_stocks = np.zeros(E)
    safety_stocks = np.zeros(E)

    for e in range(E):
        L_cum = cum_lead[e]
        ss = z * sigma * math.sqrt(L_cum)
        safety_stocks[e] = ss
        base_stocks[e] = mu * L_cum + ss

    # Echelon holding cost uses echelon holding cost rate
    # h_e^echelon = h_e - h_{e+1} for e < E-1, h_{E-1}^echelon = h_{E-1}
    echelon_h = np.zeros(E)
    for e in range(E - 1):
        echelon_h[e] = max(0.0, instance.holding_costs[e] - instance.holding_costs[e + 1])
    echelon_h[E - 1] = instance.holding_costs[E - 1]

    total_cost = float(np.sum(echelon_h * safety_stocks))

    return MultiEchelonSolution(
        base_stock_levels=base_stocks,
        safety_stocks=safety_stocks,
        total_holding_cost=total_cost,
    )


if __name__ == "__main__":
    inst = _inst.serial_3()
    sol1 = greedy_base_stock(inst)
    print(f"Greedy base-stock on {inst.name}: {sol1}")
    sol2 = echelon_stock(inst)
    print(f"Echelon stock on {inst.name}: {sol2}")
