"""
Base-stock and Powers-of-Two policies for Multi-Echelon Inventory.

Problem: Serial multi-echelon inventory optimization.
Complexity: O(L) for echelon base-stock, O(L * log(T_max/T_min)) for
powers-of-two.

Implements:
1. Echelon base-stock policy: set base-stock at each echelon to cover
   demand during cumulative lead time plus safety stock for target service.
2. Powers-of-two policy: set reorder intervals as powers of two, achieving
   at most 2% above optimal cost (Roundy, 1985).

References:
    Clark, A.J. & Scarf, H. (1960). Optimal policies for a multi-echelon
    inventory problem. Management Science, 6(4), 475-490.
    https://doi.org/10.1287/mnsc.6.4.475

    Roundy, R. (1985). 98%-effective integer-ratio lot-sizing for
    one-warehouse multi-retailer systems. Management Science, 31(11),
    1416-1430. https://doi.org/10.1287/mnsc.31.11.1416
"""

from __future__ import annotations

import os
import sys
import importlib.util

import numpy as np
from scipy import stats

_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_mod(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_inst = _load_mod("mei_instance_bs", os.path.join(_parent_dir, "instance.py"))
MultiEchelonInstance = _inst.MultiEchelonInstance
MultiEchelonSolution = _inst.MultiEchelonSolution


def echelon_base_stock(instance: MultiEchelonInstance) -> MultiEchelonSolution:
    """Echelon base-stock policy for serial multi-echelon system.

    Sets the base-stock level at each echelon to cover expected demand
    during cumulative lead time plus a safety stock determined by the
    target service level (assuming normal demand).

    Base-stock for echelon i:
        S_i = mu_i + z * sigma_i

    where mu_i, sigma_i are mean and std of demand during cumulative
    lead time from echelon i to the customer, and z = Phi^{-1}(SL).

    Args:
        instance: MultiEchelonInstance to solve.

    Returns:
        MultiEchelonSolution with base-stock levels and costs.
    """
    L = instance.L
    z = stats.norm.ppf(instance.service_level)

    base_stocks = np.zeros(L)
    safety_stocks = np.zeros(L)

    for i in range(L):
        mu, sigma = instance.demand_during_lead_time(i)
        ss = z * sigma
        base_stocks[i] = mu + ss
        safety_stocks[i] = ss

    # Expected holding cost: approximate as h_i * E[on-hand inventory at echelon i]
    # For echelon i, expected on-hand ~= safety_stock_i (in steady state)
    # Echelon holding cost uses incremental holding cost
    incremental_h = np.zeros(L)
    incremental_h[0] = instance.holding_costs[0]
    for i in range(1, L):
        incremental_h[i] = max(0.0, instance.holding_costs[i] - instance.holding_costs[i - 1]) \
            if instance.holding_costs[i] > instance.holding_costs[i - 1] \
            else instance.holding_costs[i]

    # In a serial system, echelon holding cost = sum of incremental h_i * safety_stock_i
    total_holding = float(np.sum(instance.holding_costs * safety_stocks))

    # Ordering cost approximation: D / Q * K, using EOQ-like Q
    total_ordering = 0.0
    for i in range(L):
        D = instance.mean_demand
        K = instance.ordering_costs[i]
        h = instance.holding_costs[i]
        Q_eoq = np.sqrt(2.0 * D * K / h) if h > 0 else D
        total_ordering += (D / Q_eoq) * K

    total_cost = total_holding + total_ordering

    return MultiEchelonSolution(
        base_stock_levels=base_stocks,
        safety_stocks=safety_stocks,
        total_holding_cost=total_holding,
        total_cost=total_cost,
    )


def powers_of_two(instance: MultiEchelonInstance) -> MultiEchelonSolution:
    """Powers-of-two policy for serial multi-echelon system.

    Computes EOQ-optimal reorder intervals, then rounds to the nearest
    power of two. Guarantees at most 6% above optimal for serial systems
    (Roundy, 1985). Nested policy: upstream intervals must be integer
    multiples of downstream intervals.

    Args:
        instance: MultiEchelonInstance to solve.

    Returns:
        MultiEchelonSolution with base-stock levels and costs.
    """
    L = instance.L
    z = stats.norm.ppf(instance.service_level)
    D = instance.mean_demand

    # Step 1: Compute EOQ-optimal reorder intervals per echelon
    T_eoq = np.zeros(L)
    for i in range(L):
        K = instance.ordering_costs[i]
        h = instance.holding_costs[i]
        T_eoq[i] = np.sqrt(2.0 * K / (D * h)) if h > 0 and D > 0 else 1.0

    # Step 2: Round to nearest power of two
    T_pot = np.zeros(L)
    for i in range(L):
        log2_t = np.log2(T_eoq[i])
        # Choose between floor and ceil power of two
        low = 2.0 ** np.floor(log2_t)
        high = 2.0 ** np.ceil(log2_t)
        # Select the one giving lower cost (K/T + D*h*T/2)
        cost_low = instance.ordering_costs[i] / low + D * instance.holding_costs[i] * low / 2.0
        cost_high = instance.ordering_costs[i] / high + D * instance.holding_costs[i] * high / 2.0
        T_pot[i] = low if cost_low <= cost_high else high

    # Step 3: Enforce nesting (upstream interval >= downstream)
    for i in range(1, L):
        if T_pot[i] < T_pot[i - 1]:
            T_pot[i] = T_pot[i - 1]

    # Step 4: Compute base-stock levels
    base_stocks = np.zeros(L)
    safety_stocks = np.zeros(L)

    for i in range(L):
        mu, sigma = instance.demand_during_lead_time(i)
        # Add review interval to lead time demand
        review_demand = D * T_pot[i]
        review_std = instance.std_demand * np.sqrt(T_pot[i])
        total_mu = mu + review_demand
        total_sigma = np.sqrt(sigma ** 2 + review_std ** 2)
        ss = z * total_sigma
        base_stocks[i] = total_mu + ss
        safety_stocks[i] = ss

    # Total cost
    total_holding = float(np.sum(instance.holding_costs * safety_stocks))
    total_ordering = 0.0
    for i in range(L):
        total_ordering += instance.ordering_costs[i] / T_pot[i]

    total_cost = total_holding + total_ordering

    return MultiEchelonSolution(
        base_stock_levels=base_stocks,
        safety_stocks=safety_stocks,
        total_holding_cost=total_holding,
        total_cost=total_cost,
    )


if __name__ == "__main__":
    inst = _inst.serial_3echelon()

    sol_bs = echelon_base_stock(inst)
    print(f"Echelon base-stock: {sol_bs}")
    print(f"  Base-stocks: {sol_bs.base_stock_levels}")
    print(f"  Safety stocks: {sol_bs.safety_stocks}")

    sol_pot = powers_of_two(inst)
    print(f"Powers-of-two: {sol_pot}")
    print(f"  Base-stocks: {sol_pot.base_stock_levels}")
    print(f"  Safety stocks: {sol_pot.safety_stocks}")
