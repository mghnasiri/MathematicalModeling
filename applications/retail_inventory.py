"""
Real-World Application: Retail Inventory Management Under Demand Uncertainty.

Domain: Grocery retail / Fashion retail / Seasonal merchandise
Model: Newsvendor (single-period) + EOQ (continuous review) + Lot Sizing (multi-period)

Scenario:
    A regional grocery chain manages inventory for perishable and
    seasonal products across multiple categories:

    1. **Fresh bakery** (newsvendor): Daily bread orders with uncertain
       demand — overstock is donated (salvage), understock loses sales.
    2. **Staple goods** (EOQ): Canned goods, pasta, rice — continuous
       demand with ordering costs and warehouse holding costs.
    3. **Seasonal promotions** (lot sizing): Monthly promotional items
       with time-varying demand — decide order timing and quantities.

    The chain operates 25 stores with a central distribution warehouse.

Real-world considerations modeled:
    - Demand uncertainty from historical POS data (newsvendor)
    - Trade-off between ordering frequency and holding costs (EOQ)
    - Time-varying demand with setup costs (lot sizing)
    - Perishability and shelf-life constraints

Industry context:
    US grocery stores waste ~30% of perishable inventory (USDA, 2020).
    Optimal newsvendor ordering can reduce waste by 15-20% while
    maintaining service levels above 95% (Silver et al., 2017).

References:
    Silver, E.A., Pyke, D.F. & Thomas, D.J. (2017). Inventory and
    Production Management in Supply Chains. 4th ed. CRC Press.

    Nahmias, S. & Olsen, T.L. (2015). Production and Operations
    Analysis. 7th ed. Waveland Press.
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

# Fresh bakery products for newsvendor model
BAKERY_PRODUCTS = [
    {"name": "Sourdough Loaf",       "cost": 2.50, "price": 5.99, "salvage": 0.50,
     "mean_demand": 120, "std_demand": 30},
    {"name": "Whole Wheat Bread",    "cost": 2.00, "price": 4.49, "salvage": 0.40,
     "mean_demand": 90,  "std_demand": 25},
    {"name": "Croissants (6-pack)",  "cost": 3.00, "price": 7.99, "salvage": 0.75,
     "mean_demand": 60,  "std_demand": 20},
    {"name": "Bagels (dozen)",       "cost": 2.80, "price": 6.49, "salvage": 0.60,
     "mean_demand": 80,  "std_demand": 22},
    {"name": "Cinnamon Rolls (4-pk)","cost": 3.50, "price": 8.99, "salvage": 1.00,
     "mean_demand": 45,  "std_demand": 18},
]

# Staple goods for EOQ model
STAPLE_GOODS = [
    {"name": "Canned Tomatoes (case)", "annual_demand": 15000,
     "ordering_cost": 75.0, "holding_cost": 1.20},
    {"name": "Pasta (case)",           "annual_demand": 12000,
     "ordering_cost": 60.0, "holding_cost": 0.80},
    {"name": "Rice (25kg bag)",        "annual_demand": 8000,
     "ordering_cost": 50.0, "holding_cost": 1.50},
]

# Seasonal promotion for lot sizing (12-month horizon)
SEASONAL_PROMO = {
    "name": "Holiday Gift Baskets",
    "demands": np.array([10, 15, 20, 25, 30, 20, 15, 25, 40, 80, 150, 200]),
    "months": ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    "ordering_cost": 500.0,  # per order fixed cost
    "holding_cost": 2.0,     # per unit per month
}


def create_bakery_instances(seed: int = 42) -> list[dict]:
    """Create newsvendor instances for fresh bakery products.

    Args:
        seed: Random seed for demand scenario generation.

    Returns:
        List of dictionaries with product info and demand scenarios.
    """
    rng = np.random.default_rng(seed)
    instances = []

    for product in BAKERY_PRODUCTS:
        # Generate demand scenarios from truncated normal
        raw = rng.normal(product["mean_demand"], product["std_demand"], size=200)
        scenarios = np.maximum(raw, 0).astype(float)

        instances.append({
            "name": product["name"],
            "cost": product["cost"],
            "price": product["price"],
            "salvage": product["salvage"],
            "scenarios": scenarios,
            "mean_demand": product["mean_demand"],
            "std_demand": product["std_demand"],
        })

    return instances


def create_eoq_instances() -> list[dict]:
    """Create EOQ instances for staple goods.

    Returns:
        List of dictionaries with product info.
    """
    return [
        {
            "name": good["name"],
            "annual_demand": good["annual_demand"],
            "ordering_cost": good["ordering_cost"],
            "holding_cost": good["holding_cost"],
        }
        for good in STAPLE_GOODS
    ]


def create_lot_sizing_instance() -> dict:
    """Create a lot sizing instance for seasonal promotions.

    Returns:
        Dictionary with demand pattern and cost parameters.
    """
    promo = SEASONAL_PROMO
    return {
        "name": promo["name"],
        "T": 12,
        "demands": promo["demands"].copy(),
        "ordering_cost": promo["ordering_cost"],
        "holding_cost": promo["holding_cost"],
        "months": promo["months"],
    }


def solve_bakery_newsvendor(seed: int = 42, verbose: bool = True) -> dict:
    """Solve newsvendor problems for fresh bakery products.

    Returns:
        Dictionary with results per product.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    nv_dir = os.path.join(
        base_dir, "problems", "stochastic_robust", "newsvendor"
    )

    nv_inst_mod = _load_mod(
        "nv_inst_app", os.path.join(nv_dir, "instance.py")
    )
    nv_exact_mod = _load_mod(
        "nv_exact_app", os.path.join(nv_dir, "exact", "critical_fractile.py")
    )

    bakery = create_bakery_instances(seed=seed)
    results = []

    for product in bakery:
        instance = nv_inst_mod.NewsvendorInstance(
            unit_cost=product["cost"],
            selling_price=product["price"],
            salvage_value=product["salvage"],
            demand_scenarios=product["scenarios"],
        )

        sol = nv_exact_mod.critical_fractile(instance)

        results.append({
            "name": product["name"],
            "order_quantity": sol.order_quantity,
            "expected_profit": sol.expected_profit,
            "service_level": sol.service_level,
            "critical_fractile": instance.critical_fractile,
            "cost": product["cost"],
            "price": product["price"],
        })

    if verbose:
        print("=" * 70)
        print("FRESH BAKERY — DAILY ORDER OPTIMIZATION (Newsvendor)")
        print("=" * 70)
        total_profit = 0
        for r in results:
            print(f"\n  {r['name']}:")
            print(f"    Cost=${r['cost']:.2f}, Price=${r['price']:.2f}")
            print(f"    Critical fractile: {r['critical_fractile']:.3f}")
            print(f"    Optimal order: {r['order_quantity']:.0f} units")
            print(f"    Expected daily profit: ${r['expected_profit']:.2f}")
            print(f"    Service level: {r['service_level']:.1%}")
            total_profit += r["expected_profit"]
        print(f"\n  Total expected daily profit: ${total_profit:.2f}")

    return {"bakery": results}


def solve_staple_eoq(verbose: bool = True) -> dict:
    """Solve EOQ problems for staple goods.

    Returns:
        Dictionary with EOQ results per product.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    eoq_dir = os.path.join(base_dir, "problems", "supply_chain", "eoq")

    eoq_inst_mod = _load_mod(
        "eoq_inst_app", os.path.join(eoq_dir, "instance.py")
    )
    eoq_exact_mod = _load_mod(
        "eoq_exact_app", os.path.join(eoq_dir, "exact", "eoq_formula.py")
    )

    goods = create_eoq_instances()
    results = []

    for good in goods:
        instance = eoq_inst_mod.EOQInstance(
            demand_rate=good["annual_demand"],
            ordering_cost=good["ordering_cost"],
            holding_cost=good["holding_cost"],
        )

        sol = eoq_exact_mod.classic_eoq(instance)

        results.append({
            "name": good["name"],
            "eoq": sol.order_quantity,
            "total_cost": sol.total_cost,
            "num_orders": sol.num_orders,
            "cycle_time": sol.cycle_time,
            "annual_demand": good["annual_demand"],
        })

    if verbose:
        print("\n" + "=" * 70)
        print("STAPLE GOODS — ECONOMIC ORDER QUANTITY")
        print("=" * 70)
        for r in results:
            print(f"\n  {r['name']}:")
            print(f"    Annual demand: {r['annual_demand']:,.0f} units")
            print(f"    EOQ: {r['eoq']:.0f} units per order")
            print(f"    Orders/year: {r['num_orders']:.1f}")
            cycle_days = r["cycle_time"] * 365 if r["cycle_time"] > 0 else 0
            print(f"    Cycle time: {cycle_days:.0f} days")
            print(f"    Total annual cost: ${r['total_cost']:,.2f}")

    return {"staples": results}


def solve_seasonal_lot_sizing(verbose: bool = True) -> dict:
    """Solve lot sizing for seasonal promotional items.

    Returns:
        Dictionary with lot sizing results.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ls_dir = os.path.join(base_dir, "problems", "supply_chain", "lot_sizing")

    ls_inst_mod = _load_mod(
        "ls_inst_app", os.path.join(ls_dir, "instance.py")
    )
    ls_exact_mod = _load_mod(
        "ls_exact_app", os.path.join(ls_dir, "exact", "wagner_whitin.py")
    )
    ls_heur_mod = _load_mod(
        "ls_heur_app", os.path.join(ls_dir, "heuristics", "silver_meal.py")
    )

    data = create_lot_sizing_instance()

    instance = ls_inst_mod.LotSizingInstance(
        T=data["T"],
        demands=data["demands"].astype(float),
        ordering_costs=np.full(data["T"], data["ordering_cost"]),
        holding_costs=np.full(data["T"], data["holding_cost"]),
    )

    results = {}

    # Wagner-Whitin (optimal)
    ww_sol = ls_exact_mod.wagner_whitin(instance)
    results["Wagner-Whitin"] = {
        "total_cost": ww_sol.total_cost,
        "order_periods": ww_sol.order_periods,
        "order_quantities": ww_sol.order_quantities.tolist(),
    }

    # Silver-Meal heuristic
    sm_sol = ls_heur_mod.silver_meal(instance)
    results["Silver-Meal"] = {
        "total_cost": sm_sol.total_cost,
        "order_periods": sm_sol.order_periods,
        "order_quantities": sm_sol.order_quantities.tolist(),
    }

    # Lot-for-lot
    lfl_sol = ls_heur_mod.lot_for_lot(instance)
    results["Lot-for-Lot"] = {
        "total_cost": lfl_sol.total_cost,
        "order_periods": lfl_sol.order_periods,
        "order_quantities": lfl_sol.order_quantities.tolist(),
    }

    if verbose:
        print("\n" + "=" * 70)
        print(f"SEASONAL PROMOTION — LOT SIZING: {data['name']}")
        print(f"  12-month horizon, setup=${data['ordering_cost']}, "
              f"hold=${data['holding_cost']}/unit/month")
        print("=" * 70)

        print("\n  Monthly demands:")
        for i, (month, d) in enumerate(zip(data["months"], data["demands"])):
            print(f"    {month}: {d:>4d}", end="")
            if (i + 1) % 6 == 0:
                print()
        print()

        for method, res in results.items():
            print(f"\n  {method}:")
            print(f"    Total cost: ${res['total_cost']:,.2f}")
            n_orders = len(res["order_periods"])
            print(f"    Number of orders: {n_orders}")
            order_months = [data["months"][p] for p in res["order_periods"]]
            print(f"    Order months: {', '.join(order_months)}")

    return {"lot_sizing": results}


def solve_retail_inventory(verbose: bool = True, seed: int = 42) -> dict:
    """Solve all retail inventory problems.

    Returns:
        Combined results from newsvendor, EOQ, and lot sizing.
    """
    results = {}
    results.update(solve_bakery_newsvendor(seed=seed, verbose=verbose))
    results.update(solve_staple_eoq(verbose=verbose))
    results.update(solve_seasonal_lot_sizing(verbose=verbose))
    return results


if __name__ == "__main__":
    solve_retail_inventory()
