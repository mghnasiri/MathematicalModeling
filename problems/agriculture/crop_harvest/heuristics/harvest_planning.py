"""
Perishable Crop Harvest Planning Algorithms

Solves the multi-crop newsvendor harvest planning problem using:
1. Critical fractile — optimal per-crop independent solution
2. Marginal allocation — budget-constrained multi-product greedy
3. Independent-then-scale — solve independently, scale to budget

Complexity:
    - Critical fractile: O(S log S) per crop
    - Marginal allocation: O(n * S * B/step)
    - Independent-then-scale: O(n * S log S)

References:
    Arrow, K.J., Harris, T. & Marschak, J. (1951). Optimal inventory policy.
    Econometrica, 19(3), 250-272. https://doi.org/10.2307/1906813

    Lau, H.-S. & Lau, A.H.-L. (1996). The newsstand problem: A capacitated
    multiple-product single-period inventory problem. EJOR, 94(1), 29-42.
    https://doi.org/10.1016/0377-2217(95)00192-1
"""
from __future__ import annotations

import sys
import os
import importlib.util

import numpy as np


def _load_mod(name: str, filepath: str):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_inst = _load_mod(
    "crop_harvest_inst",
    os.path.join(os.path.dirname(__file__), "..", "instance.py"),
)
CropHarvestInstance = _inst.CropHarvestInstance
CropHarvestSolution = _inst.CropHarvestSolution
CropProfile = _inst.CropProfile


def _get_newsvendor_modules():
    """Load newsvendor solver modules from the stochastic_robust family."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )))
    nv_dir = os.path.join(base_dir, "problems", "stochastic_robust", "newsvendor")

    nv_inst = _load_mod(
        "nv_inst_ch", os.path.join(nv_dir, "instance.py")
    )
    nv_cf = _load_mod(
        "nv_cf_ch", os.path.join(nv_dir, "exact", "critical_fractile.py")
    )
    nv_mp = _load_mod(
        "nv_mp_ch", os.path.join(nv_dir, "heuristics", "multi_product.py")
    )
    return nv_inst, nv_cf, nv_mp


def _crop_to_newsvendor(crop: CropProfile, demand_scenarios: np.ndarray, nv_inst):
    """Convert a CropProfile + scenarios into a NewsvendorInstance."""
    return nv_inst.NewsvendorInstance(
        unit_cost=crop.unit_cost,
        selling_price=crop.selling_price,
        salvage_value=crop.salvage_value,
        demand_scenarios=demand_scenarios,
    )


def critical_fractile_harvest(
    instance: CropHarvestInstance,
    seed: int = 42,
) -> CropHarvestSolution:
    """Solve each crop independently via the critical fractile.

    Computes the optimal harvest quantity for each crop independently,
    ignoring the shared labor budget constraint.

    Args:
        instance: CropHarvestInstance to solve.
        seed: Random seed for scenario generation.

    Returns:
        CropHarvestSolution with unconstrained per-crop optima.
    """
    nv_inst, nv_cf, _ = _get_newsvendor_modules()
    scenarios = instance.generate_scenarios(seed=seed)

    quantities = []
    profits = []
    service_levels = []
    total_cost = 0.0

    for i, crop in enumerate(instance.crops):
        nv = _crop_to_newsvendor(crop, scenarios[:, i], nv_inst)
        sol = nv_cf.critical_fractile(nv)
        quantities.append(sol.order_quantity)
        profits.append(sol.expected_profit)
        service_levels.append(sol.service_level)
        total_cost += sol.order_quantity * crop.unit_cost

    total_profit = sum(profits)
    feasible = total_cost <= instance.daily_labor_budget

    return CropHarvestSolution(
        harvest_quantities=quantities,
        expected_profits=profits,
        service_levels=service_levels,
        total_expected_profit=total_profit,
        total_harvest_cost=total_cost,
        budget_feasible=feasible,
        method="Critical Fractile (unconstrained)",
    )


def marginal_allocation_harvest(
    instance: CropHarvestInstance,
    step: float = 5.0,
    seed: int = 42,
) -> CropHarvestSolution:
    """Budget-constrained harvest via marginal allocation.

    Iteratively allocates harvest capacity to the crop with the highest
    marginal expected profit per dollar, until the labor budget is exhausted.

    Args:
        instance: CropHarvestInstance to solve.
        step: Allocation step size in kg.
        seed: Random seed for scenario generation.

    Returns:
        CropHarvestSolution with budget-constrained allocation.
    """
    nv_inst, _, nv_mp = _get_newsvendor_modules()
    scenarios = instance.generate_scenarios(seed=seed)

    nv_products = []
    for i, crop in enumerate(instance.crops):
        nv_products.append(
            _crop_to_newsvendor(crop, scenarios[:, i], nv_inst)
        )

    multi_inst = nv_mp.MultiProductInstance(
        products=nv_products,
        budget=instance.daily_labor_budget,
    )
    sol = nv_mp.marginal_allocation(multi_inst, step=step)

    # Compute per-crop metrics
    profits = []
    service_levels = []
    for i, crop in enumerate(instance.crops):
        q = sol.order_quantities[i]
        nv = nv_products[i]
        profits.append(nv.expected_profit(q))
        sl = float(np.mean(scenarios[:, i] <= q))
        service_levels.append(sl)

    return CropHarvestSolution(
        harvest_quantities=sol.order_quantities,
        expected_profits=profits,
        service_levels=service_levels,
        total_expected_profit=sol.total_expected_profit,
        total_harvest_cost=sol.total_cost,
        budget_feasible=sol.total_cost <= instance.daily_labor_budget + 1e-6,
        method="Marginal Allocation (budget-constrained)",
    )


def independent_scale_harvest(
    instance: CropHarvestInstance,
    seed: int = 42,
) -> CropHarvestSolution:
    """Solve independently then scale to fit budget.

    Computes unconstrained optimal per crop, then proportionally scales
    all quantities if total cost exceeds the labor budget.

    Args:
        instance: CropHarvestInstance to solve.
        seed: Random seed for scenario generation.

    Returns:
        CropHarvestSolution with scaled allocation.
    """
    nv_inst, _, nv_mp = _get_newsvendor_modules()
    scenarios = instance.generate_scenarios(seed=seed)

    nv_products = []
    for i, crop in enumerate(instance.crops):
        nv_products.append(
            _crop_to_newsvendor(crop, scenarios[:, i], nv_inst)
        )

    multi_inst = nv_mp.MultiProductInstance(
        products=nv_products,
        budget=instance.daily_labor_budget,
    )
    sol = nv_mp.independent_then_scale(multi_inst)

    profits = []
    service_levels = []
    for i, crop in enumerate(instance.crops):
        q = sol.order_quantities[i]
        nv = nv_products[i]
        profits.append(nv.expected_profit(q))
        sl = float(np.mean(scenarios[:, i] <= q))
        service_levels.append(sl)

    return CropHarvestSolution(
        harvest_quantities=sol.order_quantities,
        expected_profits=profits,
        service_levels=service_levels,
        total_expected_profit=sol.total_expected_profit,
        total_harvest_cost=sol.total_cost,
        budget_feasible=sol.total_cost <= instance.daily_labor_budget + 1e-6,
        method="Independent + Scale (budget-constrained)",
    )


if __name__ == "__main__":
    inst = CropHarvestInstance.quebec_farm()
    print("=== Perishable Crop Harvest Planning ===\n")

    # Unconstrained
    sol_cf = critical_fractile_harvest(inst)
    print(f"1. {sol_cf.method}")
    print(f"   Total profit: ${sol_cf.total_expected_profit:.2f}")
    print(f"   Total cost:   ${sol_cf.total_harvest_cost:.2f}")
    print(f"   Budget OK:    {sol_cf.budget_feasible}")
    for i, crop in enumerate(inst.crops):
        print(f"     {crop.name}: Q={sol_cf.harvest_quantities[i]:.1f} kg, "
              f"SL={sol_cf.service_levels[i]:.3f}")

    # Marginal allocation
    sol_ma = marginal_allocation_harvest(inst)
    print(f"\n2. {sol_ma.method}")
    print(f"   Total profit: ${sol_ma.total_expected_profit:.2f}")
    print(f"   Total cost:   ${sol_ma.total_harvest_cost:.2f}")

    # Independent + scale
    sol_is = independent_scale_harvest(inst)
    print(f"\n3. {sol_is.method}")
    print(f"   Total profit: ${sol_is.total_expected_profit:.2f}")
    print(f"   Total cost:   ${sol_is.total_harvest_cost:.2f}")
