"""
Real-World Application: Perishable Crop Harvest Planning for a Vegetable Farm.

Domain: Agriculture / perishable produce supply chain
Model: Single-product Newsvendor + Multi-product with budget constraint

Scenario:
    A vegetable farm in Quebec produces 8 perishable crops (tomatoes,
    strawberries, lettuce, cucumbers, peppers, blueberries, zucchini,
    herbs). Each day during peak season, the farmer must decide how many
    kilograms of each crop to harvest before demand from farmers markets,
    restaurants, and grocery stores is revealed.

    Unharvested crop perishes in the field (no cost but lost revenue).
    Excess harvested produce that cannot be sold at full price is either
    sold at a discount to food processors, donated, or composted
    (salvage value). The harvest/packing cost per kg varies by crop.

    Two planning modes are considered:
    1. Unconstrained — each crop is optimized independently via the
       critical fractile (single-product newsvendor).
    2. Budget-constrained — a total daily labor/logistics budget limits
       how much can be harvested across all crops, requiring multi-product
       allocation via marginal analysis.

Real-world considerations modeled:
    - Perishability (1-3 day shelf life for most crops)
    - Heterogeneous profit margins across crops
    - Demand variability (weather, weekday vs weekend, seasonal)
    - Salvage channels (food processors, composting, food banks)
    - Labor budget constraints during peak harvest season

Industry context:
    Fresh produce waste in Canada reaches 30-40% of total production.
    Newsvendor-style ordering decisions are made daily by farms supplying
    perishable goods to local markets. Optimal harvest quantities can
    reduce waste by 15-25% while maintaining service levels above 90%
    (Minner & Transchel, 2010).

References:
    Minner, S. & Transchel, S. (2010). Periodic review inventory-control
    for perishable products under service-level constraints. OR Spectrum,
    32(4), 979-996. https://doi.org/10.1007/s00291-010-0196-1

    Ketzenberg, M. & Ferguson, M.E. (2008). Managing slow-moving
    perishables in the grocery industry. Production and Operations
    Management, 17(5), 513-521. https://doi.org/10.3401/poms.1080.0052
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


# ── Domain Data ──────────────────────────────────────────────────────────────

# 8 perishable crops grown on the Quebec vegetable farm
CROPS = [
    {
        "name": "Tomatoes",
        "unit_cost": 1.50,       # $/kg harvest + packing cost
        "selling_price": 4.00,   # $/kg at farmers market / grocery
        "salvage_value": 0.50,   # $/kg food processor / compost
        "demand_mean": 500.0,    # kg/day expected demand
        "demand_std": 100.0,     # kg/day demand variability
    },
    {
        "name": "Strawberries",
        "unit_cost": 3.00,
        "selling_price": 8.00,
        "salvage_value": 1.00,
        "demand_mean": 200.0,
        "demand_std": 60.0,
    },
    {
        "name": "Lettuce",
        "unit_cost": 0.80,
        "selling_price": 2.50,
        "salvage_value": 0.20,
        "demand_mean": 300.0,
        "demand_std": 80.0,
    },
    {
        "name": "Cucumbers",
        "unit_cost": 0.60,
        "selling_price": 2.00,
        "salvage_value": 0.15,
        "demand_mean": 350.0,
        "demand_std": 90.0,
    },
    {
        "name": "Peppers",
        "unit_cost": 1.80,
        "selling_price": 5.00,
        "salvage_value": 0.60,
        "demand_mean": 250.0,
        "demand_std": 70.0,
    },
    {
        "name": "Blueberries",
        "unit_cost": 4.00,
        "selling_price": 10.00,
        "salvage_value": 1.50,
        "demand_mean": 150.0,
        "demand_std": 50.0,
    },
    {
        "name": "Zucchini",
        "unit_cost": 0.50,
        "selling_price": 1.80,
        "salvage_value": 0.10,
        "demand_mean": 400.0,
        "demand_std": 120.0,
    },
    {
        "name": "Herbs",
        "unit_cost": 2.50,
        "selling_price": 12.00,
        "salvage_value": 0.80,
        "demand_mean": 80.0,
        "demand_std": 25.0,
    },
]

# Total daily labor/logistics budget (in dollars) constraining total harvest
DAILY_LABOR_BUDGET = 2500.0

N_SCENARIOS = 50


def create_crop_instances(seed: int = 42) -> list[dict]:
    """Create a NewsvendorInstance for each crop with stochastic demand.

    Generates demand scenarios from Normal distributions clipped at zero
    (demand cannot be negative).

    Args:
        seed: Random seed for reproducibility.

    Returns:
        List of dicts, each containing the crop metadata and its
        NewsvendorInstance.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    nv_dir = os.path.join(base_dir, "problems", "stochastic_robust", "newsvendor")

    nv_inst_mod = _load_mod(
        "nv_inst_agr", os.path.join(nv_dir, "instance.py")
    )

    rng = np.random.default_rng(seed)
    crop_instances = []

    for crop in CROPS:
        demand_scenarios = np.maximum(
            0.0,
            rng.normal(crop["demand_mean"], crop["demand_std"], N_SCENARIOS),
        )

        instance = nv_inst_mod.NewsvendorInstance(
            unit_cost=crop["unit_cost"],
            selling_price=crop["selling_price"],
            salvage_value=crop["salvage_value"],
            demand_scenarios=demand_scenarios,
        )

        crop_instances.append({
            "name": crop["name"],
            "demand_mean": crop["demand_mean"],
            "demand_std": crop["demand_std"],
            "instance": instance,
        })

    return crop_instances


def solve_crop_harvest(verbose: bool = True, seed: int = 42) -> dict:
    """Solve the perishable crop harvest planning problem.

    1. Solves each crop independently via the critical fractile
       (unconstrained newsvendor).
    2. Solves the multi-product problem under a daily labor budget
       using marginal allocation and independent-then-scale heuristics.

    Args:
        verbose: If True, print detailed results.
        seed: Random seed for scenario generation.

    Returns:
        Dictionary with unconstrained and constrained results.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    nv_dir = os.path.join(base_dir, "problems", "stochastic_robust", "newsvendor")

    # Load solver modules
    cf_mod = _load_mod(
        "nv_cf_agr", os.path.join(nv_dir, "exact", "critical_fractile.py")
    )
    mp_mod = _load_mod(
        "nv_mp_agr", os.path.join(nv_dir, "heuristics", "multi_product.py")
    )

    crop_data = create_crop_instances(seed=seed)

    # ── 1. Unconstrained single-product newsvendor per crop ──────────────
    unconstrained_results = []
    total_unconstrained_cost = 0.0
    total_unconstrained_profit = 0.0

    for entry in crop_data:
        sol = cf_mod.critical_fractile(entry["instance"])
        harvest_cost = sol.order_quantity * entry["instance"].unit_cost
        total_unconstrained_cost += harvest_cost
        total_unconstrained_profit += sol.expected_profit

        unconstrained_results.append({
            "name": entry["name"],
            "order_quantity": sol.order_quantity,
            "expected_profit": sol.expected_profit,
            "expected_cost": sol.expected_cost,
            "service_level": sol.service_level,
            "critical_fractile": entry["instance"].critical_fractile,
            "harvest_cost": harvest_cost,
        })

    # ── 2. Budget-constrained multi-product newsvendor ───────────────────
    multi_instance = mp_mod.MultiProductInstance(
        products=[entry["instance"] for entry in crop_data],
        budget=DAILY_LABOR_BUDGET,
    )

    ma_sol = mp_mod.marginal_allocation(multi_instance, step=5.0)
    is_sol = mp_mod.independent_then_scale(multi_instance)

    constrained_results = {
        "marginal_allocation": ma_sol,
        "independent_scale": is_sol,
    }

    results = {
        "unconstrained": unconstrained_results,
        "total_unconstrained_cost": total_unconstrained_cost,
        "total_unconstrained_profit": total_unconstrained_profit,
        "budget": DAILY_LABOR_BUDGET,
        "constrained": constrained_results,
    }

    # ── Verbose output ───────────────────────────────────────────────────
    if verbose:
        print("=" * 70)
        print("PERISHABLE CROP HARVEST PLANNING — QUEBEC VEGETABLE FARM")
        print(f"  {len(CROPS)} crops, {N_SCENARIOS} demand scenarios, "
              f"budget=${DAILY_LABOR_BUDGET:.0f}/day")
        print("=" * 70)

        # Unconstrained results
        print("\n--- Unconstrained Newsvendor (per crop) ---\n")
        print(f"  {'Crop':<15s} {'Q* (kg)':>8s} {'CF':>6s} {'SL':>6s} "
              f"{'E[profit]':>10s} {'Harv. cost':>10s}")
        print("  " + "-" * 60)

        for r in unconstrained_results:
            print(f"  {r['name']:<15s} {r['order_quantity']:>8.1f} "
                  f"{r['critical_fractile']:>6.3f} "
                  f"{r['service_level']:>6.3f} "
                  f"${r['expected_profit']:>9.2f} "
                  f"${r['harvest_cost']:>9.2f}")

        print("  " + "-" * 60)
        print(f"  {'TOTAL':<15s} {'':>8s} {'':>6s} {'':>6s} "
              f"${total_unconstrained_profit:>9.2f} "
              f"${total_unconstrained_cost:>9.2f}")

        budget_feasible = total_unconstrained_cost <= DAILY_LABOR_BUDGET
        print(f"\n  Total harvest cost: ${total_unconstrained_cost:.2f} "
              f"(budget: ${DAILY_LABOR_BUDGET:.2f}) "
              f"{'<= FEASIBLE' if budget_feasible else '> INFEASIBLE'}")

        # Constrained results
        print("\n--- Budget-Constrained Multi-Product Solutions ---\n")

        for method_name, sol in [("Marginal Allocation", ma_sol),
                                  ("Independent + Scale", is_sol)]:
            print(f"  {method_name}:")
            print(f"    Total E[profit]: ${sol.total_expected_profit:.2f}")
            print(f"    Total cost:      ${sol.total_cost:.2f}")
            print(f"    Budget used:     {sol.budget_used:.1%}")
            print(f"    Quantities:")
            for i, entry in enumerate(crop_data):
                q = sol.order_quantities[i]
                if q > 0:
                    print(f"      {entry['name']:<15s} {q:>8.1f} kg")
            print()

        # Comparison
        print("--- Comparison: Unconstrained vs Constrained ---\n")
        print(f"  Unconstrained total profit:   ${total_unconstrained_profit:.2f}")
        print(f"  Marginal Alloc. profit:       ${ma_sol.total_expected_profit:.2f}")
        print(f"  Independent+Scale profit:     ${is_sol.total_expected_profit:.2f}")

        if total_unconstrained_profit > 0:
            ma_gap = (1 - ma_sol.total_expected_profit / total_unconstrained_profit) * 100
            is_gap = (1 - is_sol.total_expected_profit / total_unconstrained_profit) * 100
            print(f"\n  Profit loss from budget constraint:")
            print(f"    Marginal Allocation:    {ma_gap:>6.1f}%")
            print(f"    Independent + Scale:    {is_gap:>6.1f}%")

    return results


if __name__ == "__main__":
    solve_crop_harvest()
