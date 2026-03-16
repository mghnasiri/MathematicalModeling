"""
Real-World Application: Seed Procurement for Planting Season.

Domain: Agricultural seed supply / Cooperative seed distribution
Model: Newsvendor (single-period stochastic inventory)

Scenario:
    A regional seed cooperative serves 50 farms across a temperate
    agricultural zone. Before each planting season, the cooperative
    must order seeds for 5 major crop varieties: corn, wheat, soybean,
    canola, and barley. Demand for each variety is uncertain, driven
    by fluctuating weather forecasts, commodity futures prices, crop
    rotation requirements, and government subsidy programs.

    Over-ordering wastes the cooperative's limited procurement budget
    (excess seed has reduced viability next season and must be sold at
    a discount). Under-ordering means member farms cannot plant their
    intended acreage, losing an entire growing season's revenue on
    those acres.

    Each crop variety is modeled as an independent newsvendor problem
    with scenario-based demand derived from historical planting data,
    weather variability, and market price elasticity.

Real-world considerations modeled:
    - Demand uncertainty from weather forecasts and commodity prices
    - Perishability of seed (reduced germination rate if stored)
    - High underage cost (lost planting opportunity = lost season)
    - Salvage value for excess seed (discounted resale or carry-over)
    - Crop-specific cost structures and margin profiles

Industry context:
    Seed costs represent 15-20% of total farm input costs in North
    American row crop agriculture (USDA ERS, 2023). Optimal seed
    ordering through cooperative procurement can reduce waste by
    10-15% while ensuring 95%+ fill rates for member farms. The
    US seed market exceeds $25 billion annually, with corn and
    soybean seed accounting for over 60% of expenditure.

References:
    Nahmias, S. & Olsen, T.L. (2015). Production and Operations
    Analysis. 7th ed. Waveland Press.

    USDA Economic Research Service (2023). Commodity Costs and
    Returns. https://www.ers.usda.gov/data-products/commodity-costs-and-returns/

    Ahumada, O. & Villalobos, J.R. (2009). Application of planning
    models in the agri-food supply chain: A review. European Journal
    of Operational Research, 196(1), 1-20.
    https://doi.org/10.1016/j.ejor.2008.02.014
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

N_FARMS = 50  # member farms in the cooperative

# 5 crop varieties with realistic seed economics
# cost = cooperative purchase price per unit (bag/bushel)
# price = value to farms (avoided lost-season revenue per unit demand)
# salvage = discounted resale of excess seed (reduced germination)
SEED_VARIETIES = [
    {
        "name": "Corn (80K kernel bag)",
        "cost": 280.00,
        "price": 520.00,
        "salvage": 85.00,
        "mean_demand": 1200,
        "std_demand": 250,
        "unit": "bags",
        "acres_per_unit": 2.5,
    },
    {
        "name": "Wheat (50 lb bag)",
        "cost": 18.50,
        "price": 42.00,
        "salvage": 5.50,
        "mean_demand": 3500,
        "std_demand": 700,
        "unit": "bags",
        "acres_per_unit": 1.0,
    },
    {
        "name": "Soybean (140K seed bag)",
        "cost": 62.00,
        "price": 115.00,
        "salvage": 18.00,
        "mean_demand": 2200,
        "std_demand": 480,
        "unit": "bags",
        "acres_per_unit": 1.0,
    },
    {
        "name": "Canola (10 lb bag)",
        "cost": 45.00,
        "price": 95.00,
        "salvage": 12.00,
        "mean_demand": 800,
        "std_demand": 200,
        "unit": "bags",
        "acres_per_unit": 2.0,
    },
    {
        "name": "Barley (48 lb bushel)",
        "cost": 14.00,
        "price": 32.00,
        "salvage": 4.00,
        "mean_demand": 1500,
        "std_demand": 350,
        "unit": "bushels",
        "acres_per_unit": 0.5,
    },
]

# Weather scenario multipliers (affects demand across all crops)
WEATHER_SCENARIOS = {
    "ideal_spring": 1.15,    # warm, dry planting = high demand
    "normal": 1.00,          # average conditions
    "wet_spring": 0.85,      # delayed planting, reduced acreage
    "drought_forecast": 0.70, # farmers cut plantings
    "early_warm": 1.10,      # early season = more planting
}


def create_seed_instances(seed: int = 42) -> list[dict]:
    """Create newsvendor instances for each seed variety.

    Demand scenarios are generated from a truncated normal distribution,
    modulated by weather scenario probabilities.

    Args:
        seed: Random seed for demand scenario generation.

    Returns:
        List of dictionaries with variety info and demand scenarios.
    """
    rng = np.random.default_rng(seed)
    instances = []

    for variety in SEED_VARIETIES:
        # Generate 200 demand scenarios incorporating weather variability
        scenarios = []
        weather_multipliers = list(WEATHER_SCENARIOS.values())

        for _ in range(200):
            # Random weather effect
            wx = rng.choice(weather_multipliers)
            # Base demand from normal distribution
            base = rng.normal(variety["mean_demand"], variety["std_demand"])
            demand = max(0, base * wx)
            scenarios.append(demand)

        scenarios = np.array(scenarios)

        instances.append({
            "name": variety["name"],
            "cost": variety["cost"],
            "price": variety["price"],
            "salvage": variety["salvage"],
            "scenarios": scenarios,
            "mean_demand": variety["mean_demand"],
            "std_demand": variety["std_demand"],
            "unit": variety["unit"],
            "acres_per_unit": variety["acres_per_unit"],
        })

    return instances


def solve_seed_inventory(seed: int = 42, verbose: bool = True) -> dict:
    """Solve newsvendor problems for seed procurement.

    Args:
        seed: Random seed for scenario generation.
        verbose: Whether to print detailed results.

    Returns:
        Dictionary with results per crop variety.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    nv_dir = os.path.join(
        base_dir, "problems", "stochastic_robust", "newsvendor"
    )

    nv_inst_mod = _load_mod(
        "nv_inst_seed", os.path.join(nv_dir, "instance.py")
    )
    nv_exact_mod = _load_mod(
        "nv_exact_seed", os.path.join(nv_dir, "exact", "critical_fractile.py")
    )

    varieties = create_seed_instances(seed=seed)
    results = []

    for variety in varieties:
        instance = nv_inst_mod.NewsvendorInstance(
            unit_cost=variety["cost"],
            selling_price=variety["price"],
            salvage_value=variety["salvage"],
            demand_scenarios=variety["scenarios"],
        )

        sol = nv_exact_mod.critical_fractile(instance)

        total_order_cost = sol.order_quantity * variety["cost"]
        acres_covered = sol.order_quantity * variety["acres_per_unit"]

        results.append({
            "name": variety["name"],
            "unit": variety["unit"],
            "order_quantity": sol.order_quantity,
            "expected_profit": sol.expected_profit,
            "service_level": sol.service_level,
            "critical_fractile": instance.critical_fractile,
            "cost": variety["cost"],
            "price": variety["price"],
            "salvage": variety["salvage"],
            "total_order_cost": total_order_cost,
            "acres_covered": acres_covered,
        })

    if verbose:
        print("=" * 70)
        print("SEED COOPERATIVE — PLANTING SEASON PROCUREMENT (Newsvendor)")
        print(f"  Serving {N_FARMS} member farms")
        print("=" * 70)

        total_budget = 0
        total_profit = 0
        total_acres = 0

        for r in results:
            print(f"\n  {r['name']}:")
            print(f"    Purchase cost: ${r['cost']:.2f}/{r['unit'][:-1]}, "
                  f"Value: ${r['price']:.2f}, "
                  f"Salvage: ${r['salvage']:.2f}")
            print(f"    Critical fractile: {r['critical_fractile']:.3f}")
            print(f"    Optimal order: {r['order_quantity']:.0f} {r['unit']}")
            print(f"    Total procurement cost: ${r['total_order_cost']:,.0f}")
            print(f"    Acres covered: {r['acres_covered']:,.0f}")
            print(f"    Expected profit: ${r['expected_profit']:,.0f}")
            print(f"    Service level: {r['service_level']:.1%}")
            total_budget += r["total_order_cost"]
            total_profit += r["expected_profit"]
            total_acres += r["acres_covered"]

        print(f"\n  {'─' * 50}")
        print(f"  Total procurement budget: ${total_budget:,.0f}")
        print(f"  Total expected profit: ${total_profit:,.0f}")
        print(f"  Total acres covered: {total_acres:,.0f}")
        print(f"  Budget per farm: ${total_budget / N_FARMS:,.0f}")

    return {"seed_inventory": results}


if __name__ == "__main__":
    solve_seed_inventory()
