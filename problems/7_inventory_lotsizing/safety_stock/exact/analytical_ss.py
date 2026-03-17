"""
Analytical Safety Stock Formulas — Exact computation for safety stock.

Problem: Safety stock optimization under normal demand and lead time.
Complexity: O(n) for n items.

Computes:
- sigma_DDLT = sqrt(L * sigma_D^2 + D^2 * sigma_L^2)
  (std of demand during lead time)
- Safety stock = z * sigma_DDLT where z = Phi^{-1}(SL)
- Reorder point = D * L + safety_stock

References:
    Silver, E.A., Pyke, D.F. & Thomas, D.J. (2016). Inventory and
    Production Management in Supply Chains. 4th edition, CRC Press.
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


_inst = _load_mod("ss_instance_analytical", os.path.join(_parent_dir, "instance.py"))
SafetyStockInstance = _inst.SafetyStockInstance
SafetyStockSolution = _inst.SafetyStockSolution


def compute_sigma_ddlt(
    mean_demand: float,
    std_demand: float,
    mean_lead_time: float,
    std_lead_time: float,
) -> float:
    """Compute standard deviation of demand during lead time.

    sigma_DDLT = sqrt(L * sigma_D^2 + D^2 * sigma_L^2)

    Args:
        mean_demand: Average demand per period.
        std_demand: Std dev of demand per period.
        mean_lead_time: Average lead time in periods.
        std_lead_time: Std dev of lead time.

    Returns:
        Standard deviation of demand during lead time.
    """
    variance = (mean_lead_time * std_demand ** 2
                + mean_demand ** 2 * std_lead_time ** 2)
    return float(np.sqrt(max(0.0, variance)))


def analytical_safety_stock(
    instance: SafetyStockInstance,
) -> SafetyStockSolution:
    """Compute safety stocks using analytical normal approximation.

    For each item i:
    - sigma_DDLT_i = sqrt(L_i * sigma_D_i^2 + D_i^2 * sigma_L_i^2)
    - SS_i = z * sigma_DDLT_i
    - ROP_i = D_i * L_i + SS_i

    Args:
        instance: A SafetyStockInstance.

    Returns:
        SafetyStockSolution with safety stocks and reorder points.
    """
    z = float(stats.norm.ppf(instance.service_level))
    n = instance.n

    safety_stocks = np.zeros(n)
    reorder_points = np.zeros(n)

    for i in range(n):
        sigma_ddlt = compute_sigma_ddlt(
            instance.mean_demands[i],
            instance.std_demands[i],
            instance.mean_lead_times[i],
            instance.std_lead_times[i],
        )
        ss = z * sigma_ddlt
        safety_stocks[i] = ss
        reorder_points[i] = (instance.mean_demands[i]
                             * instance.mean_lead_times[i] + ss)

    total_cost = float(np.sum(instance.holding_costs * safety_stocks))

    return SafetyStockSolution(
        safety_stocks=safety_stocks,
        reorder_points=reorder_points,
        total_holding_cost=total_cost,
    )


def safety_stock_fill_rate(
    instance: SafetyStockInstance,
    target_fill_rate: float = 0.95,
) -> SafetyStockSolution:
    """Compute safety stocks targeting a fill rate (Type II service).

    Uses the loss function approach:
    E[shortage] = sigma_DDLT * L(z) where L(z) = phi(z) - z*(1-Phi(z))
    Fill rate = 1 - E[shortage] / Q

    Approximation: uses cycle service level z for simplicity.

    Args:
        instance: A SafetyStockInstance.
        target_fill_rate: Target fill rate (0 < p < 1).

    Returns:
        SafetyStockSolution with fill-rate-based safety stocks.
    """
    z = float(stats.norm.ppf(target_fill_rate))
    n = instance.n

    safety_stocks = np.zeros(n)
    reorder_points = np.zeros(n)

    for i in range(n):
        sigma_ddlt = compute_sigma_ddlt(
            instance.mean_demands[i],
            instance.std_demands[i],
            instance.mean_lead_times[i],
            instance.std_lead_times[i],
        )
        ss = z * sigma_ddlt
        safety_stocks[i] = ss
        reorder_points[i] = (instance.mean_demands[i]
                             * instance.mean_lead_times[i] + ss)

    total_cost = float(np.sum(instance.holding_costs * safety_stocks))

    return SafetyStockSolution(
        safety_stocks=safety_stocks,
        reorder_points=reorder_points,
        total_holding_cost=total_cost,
    )


if __name__ == "__main__":
    inst = _inst.basic_3items()
    sol = analytical_safety_stock(inst)
    print(f"Analytical safety stock on {inst.name}:")
    print(f"  {sol}")
    for i in range(inst.n):
        print(f"  Item {i}: SS={sol.safety_stocks[i]:.1f}, "
              f"ROP={sol.reorder_points[i]:.1f}")
