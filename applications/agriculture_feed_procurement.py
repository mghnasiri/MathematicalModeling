"""
Real-World Application: Agriculture Feed & Fertilizer Procurement Planning.

Domain: Dairy farming / Agricultural input procurement
Models: EOQ + Silver-Meal Lot Sizing + Wagner-Whitin DP

Scenario:
    A 500-head dairy farm in Quebec needs to plan procurement of three
    key agricultural inputs over a 12-month horizon (Jan-Dec):

    1. Cattle feed (grain mix) — steady demand with seasonal variation;
       cows consume more feed in winter when pasture is unavailable.
    2. Fertilizer (NPK 15-15-15) — highly seasonal demand concentrated
       in spring and summer for hay field and corn silage production.
    3. Seeds (hay/silage mix) — demand concentrated in the spring
       planting window (March-May) with zero demand otherwise.

    For each input, the farm evaluates three procurement strategies:
    - EOQ: steady-state baseline assuming constant average demand
    - Silver-Meal: dynamic lot sizing heuristic for time-varying demand
    - Wagner-Whitin: optimal dynamic programming solution

    The comparison reveals cost savings from dynamic lot sizing over
    the naive EOQ approach when demand is highly seasonal.

Real-world considerations modeled:
    - Seasonal demand variation (winter feeding, spring planting)
    - Storage/holding costs (grain spoilage, fertilizer caking)
    - Fixed ordering/delivery costs (truck dispatch, paperwork)
    - Zero-demand periods (fertilizer in winter, seeds outside spring)

Industry context:
    Feed costs represent 50-70% of dairy farm operating expenses.
    Optimizing procurement timing and quantities can reduce total
    inventory costs by 10-25%, especially for inputs with seasonal
    demand patterns (Boyabatli et al., 2019). The Wagner-Whitin model
    is particularly effective when demand variability is high, as with
    fertilizer tied to the growing season.

References:
    Wagner, H.M. & Whitin, T.M. (1958). Dynamic version of the economic
    lot size model. Management Science, 5(1), 89-96.
    https://doi.org/10.1287/mnsc.5.1.89

    Silver, E.A. & Meal, H.C. (1973). A heuristic for selecting lot size
    quantities for the case of a deterministic time-varying demand rate
    and discrete opportunities for replenishment. Production and Inventory
    Management, 14(2), 64-74.

    Boyabatli, O., Nasiry, J. & Zhou, Y.H. (2019). Crop planning in
    sustainable agriculture: Dynamic farmland allocation in the presence
    of crop rotation benefits. Management Science, 65(5), 2060-2076.
    https://doi.org/10.1287/mnsc.2018.3044
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

MONTHS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]

# Three agricultural inputs with realistic cost and demand parameters.

INPUTS = {
    "cattle_feed": {
        "name": "Cattle Feed (grain mix)",
        "unit": "tons",
        "ordering_cost": 150.0,    # $/order (truck dispatch + paperwork)
        "holding_cost": 2.0,       # $/ton/month (spoilage, storage)
        # Monthly demand (tons): higher in winter (Nov-Mar), lower in summer
        "demands": np.array([
            75.0, 72.0, 65.0, 55.0, 45.0, 40.0,
            42.0, 45.0, 50.0, 58.0, 70.0, 80.0,
        ]),
    },
    "fertilizer": {
        "name": "Fertilizer (NPK 15-15-15)",
        "unit": "tons",
        "ordering_cost": 200.0,    # $/order
        "holding_cost": 5.0,       # $/ton/month (caking, moisture)
        # Monthly demand (tons): spring/summer peak, zero in winter
        "demands": np.array([
            0.0, 0.0, 15.0, 45.0, 60.0, 40.0,
            25.0, 10.0, 0.0, 0.0, 0.0, 0.0,
        ]),
    },
    "seeds": {
        "name": "Seeds (hay/silage mix)",
        "unit": "tons",
        "ordering_cost": 100.0,    # $/order
        "holding_cost": 3.0,       # $/ton/month (germination loss)
        # Monthly demand (tons): spring planting window only
        "demands": np.array([
            0.0, 0.0, 8.0, 15.0, 12.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ]),
    },
}


def create_eoq_instance(input_key: str):
    """Create an EOQ instance for steady-state baseline analysis.

    Uses average monthly demand annualized (x12) as the demand rate.

    Args:
        input_key: Key into INPUTS dictionary.

    Returns:
        EOQInstance for the specified agricultural input.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    eoq_dir = os.path.join(base_dir, "problems", "supply_chain", "eoq")

    eoq_inst_mod = _load_mod(
        "eoq_inst_agr", os.path.join(eoq_dir, "instance.py")
    )

    data = INPUTS[input_key]
    demands = data["demands"]
    # Only consider non-zero demand months for average
    nonzero = demands[demands > 0]
    avg_monthly = float(np.mean(nonzero)) if len(nonzero) > 0 else 0.0
    annual_demand = avg_monthly * 12.0

    return eoq_inst_mod.EOQInstance(
        demand_rate=annual_demand,
        ordering_cost=data["ordering_cost"],
        holding_cost=data["holding_cost"] * 12.0,  # annualize holding cost
        name=f"eoq_{input_key}",
    )


def create_lot_sizing_instance(input_key: str):
    """Create a LotSizingInstance for Silver-Meal heuristic.

    Args:
        input_key: Key into INPUTS dictionary.

    Returns:
        LotSizingInstance for the specified agricultural input.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ls_dir = os.path.join(base_dir, "problems", "supply_chain", "lot_sizing")

    ls_inst_mod = _load_mod(
        "ls_inst_agr", os.path.join(ls_dir, "instance.py")
    )

    data = INPUTS[input_key]
    T = 12

    return ls_inst_mod.LotSizingInstance(
        T=T,
        demands=data["demands"].copy(),
        ordering_costs=np.full(T, data["ordering_cost"]),
        holding_costs=np.full(T, data["holding_cost"]),
        name=f"ls_{input_key}",
    )


def create_wagner_whitin_instance(input_key: str):
    """Create a WagnerWhitinInstance for optimal DP solution.

    Args:
        input_key: Key into INPUTS dictionary.

    Returns:
        WagnerWhitinInstance for the specified agricultural input.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ww_dir = os.path.join(
        base_dir, "problems", "supply_chain", "wagner_whitin"
    )

    ww_inst_mod = _load_mod(
        "ww_inst_agr", os.path.join(ww_dir, "instance.py")
    )

    data = INPUTS[input_key]
    T = 12

    return ww_inst_mod.WagnerWhitinInstance(
        T=T,
        demands=data["demands"].copy(),
        ordering_costs=np.full(T, data["ordering_cost"]),
        holding_costs=np.full(T, data["holding_cost"]),
        name=f"ww_{input_key}",
    )


def solve_feed_procurement(verbose: bool = True) -> dict:
    """Solve the agriculture feed procurement problem.

    For each agricultural input, computes:
    1. EOQ baseline (classic formula on average demand)
    2. Silver-Meal dynamic lot sizing plan
    3. Wagner-Whitin optimal lot sizing plan

    Args:
        verbose: If True, print detailed results and comparisons.

    Returns:
        Dictionary mapping input names to method results.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Load solver modules
    eoq_dir = os.path.join(base_dir, "problems", "supply_chain", "eoq")
    ls_dir = os.path.join(base_dir, "problems", "supply_chain", "lot_sizing")
    ww_dir = os.path.join(
        base_dir, "problems", "supply_chain", "wagner_whitin"
    )

    eoq_formula_mod = _load_mod(
        "eoq_formula_agr", os.path.join(eoq_dir, "exact", "eoq_formula.py")
    )
    silver_meal_mod = _load_mod(
        "silver_meal_agr",
        os.path.join(ls_dir, "heuristics", "silver_meal.py"),
    )
    ww_dp_mod = _load_mod(
        "ww_dp_agr", os.path.join(ww_dir, "exact", "wagner_whitin_dp.py")
    )

    all_results = {}

    for input_key, data in INPUTS.items():
        results = {}
        demands = data["demands"]
        total_demand = float(np.sum(demands))

        # ── 1. EOQ baseline ───────────────────────────────────────────────
        eoq_instance = create_eoq_instance(input_key)
        eoq_sol = eoq_formula_mod.classic_eoq(eoq_instance)

        # Convert EOQ to a monthly order schedule for cost comparison
        # Order EOQ quantity whenever cumulative demand reaches Q*
        Q = eoq_sol.order_quantity
        eoq_orders = np.zeros(12, dtype=float)
        inventory = 0.0
        eoq_total = 0.0
        for t in range(12):
            if inventory < demands[t]:
                # Place an order of size Q (or enough to cover demand)
                order_qty = max(Q, demands[t] - inventory)
                eoq_orders[t] = order_qty
                inventory += order_qty
                eoq_total += data["ordering_cost"]
            inventory -= demands[t]
            eoq_total += data["holding_cost"] * inventory

        results["EOQ"] = {
            "eoq_quantity": eoq_sol.order_quantity,
            "annual_cost": eoq_sol.total_cost,
            "monthly_orders": eoq_orders,
            "order_periods": [t for t in range(12) if eoq_orders[t] > 0],
            "simulated_cost": eoq_total,
        }

        # ── 2. Silver-Meal ────────────────────────────────────────────────
        ls_instance = create_lot_sizing_instance(input_key)
        sm_sol = silver_meal_mod.silver_meal(ls_instance)

        results["Silver-Meal"] = {
            "total_cost": sm_sol.total_cost,
            "order_quantities": sm_sol.order_quantities,
            "order_periods": sm_sol.order_periods,
            "num_orders": len(sm_sol.order_periods),
        }

        # ── 3. Wagner-Whitin ─────────────────────────────────────────────
        ww_instance = create_wagner_whitin_instance(input_key)
        ww_sol = ww_dp_mod.wagner_whitin_dp(ww_instance)

        results["Wagner-Whitin"] = {
            "total_cost": ww_sol.total_cost,
            "order_quantities": ww_sol.order_quantities,
            "order_periods": ww_sol.order_periods,
            "num_orders": len(ww_sol.order_periods),
        }

        all_results[input_key] = {
            "name": data["name"],
            "unit": data["unit"],
            "total_demand": total_demand,
            "methods": results,
        }

    if verbose:
        print("=" * 70)
        print("AGRICULTURE FEED & FERTILIZER PROCUREMENT PLANNING")
        print("  500-head dairy farm, Quebec — 12-month horizon")
        print("=" * 70)

        for input_key, res in all_results.items():
            data = INPUTS[input_key]
            methods = res["methods"]

            print(f"\n{'─' * 70}")
            print(f"  {res['name']}")
            print(f"  Total annual demand: {res['total_demand']:.0f} {res['unit']}")
            print(f"  Ordering cost: ${data['ordering_cost']:.0f}/order")
            print(f"  Holding cost: ${data['holding_cost']:.2f}/{res['unit']}/month")
            print(f"{'─' * 70}")

            # Demand profile
            print("\n  Monthly demand profile:")
            print("    " + "  ".join(f"{m:>5s}" for m in MONTHS))
            print("    " + "  ".join(
                f"{d:5.0f}" for d in data["demands"]
            ))

            # EOQ baseline
            eoq = methods["EOQ"]
            print(f"\n  1. EOQ (steady-state baseline):")
            print(f"     EOQ quantity: {eoq['eoq_quantity']:.1f} {res['unit']}")
            print(f"     EOQ annual cost (formula): ${eoq['annual_cost']:.2f}")
            print(f"     Simulated 12-month cost:   ${eoq['simulated_cost']:.2f}")
            print(f"     Orders placed in: {[MONTHS[t] for t in eoq['order_periods']]}")

            # Silver-Meal
            sm = methods["Silver-Meal"]
            print(f"\n  2. Silver-Meal (dynamic heuristic):")
            print(f"     Total cost: ${sm['total_cost']:.2f}")
            print(f"     Number of orders: {sm['num_orders']}")
            print(f"     Orders placed in: {[MONTHS[t] for t in sm['order_periods']]}")
            print("     Monthly order schedule:")
            print("    " + "  ".join(f"{m:>5s}" for m in MONTHS))
            print("    " + "  ".join(
                f"{q:5.0f}" for q in sm["order_quantities"]
            ))

            # Wagner-Whitin
            ww = methods["Wagner-Whitin"]
            print(f"\n  3. Wagner-Whitin (optimal DP):")
            print(f"     Total cost: ${ww['total_cost']:.2f}")
            print(f"     Number of orders: {ww['num_orders']}")
            print(f"     Orders placed in: {[MONTHS[t] for t in ww['order_periods']]}")
            print("     Monthly order schedule:")
            print("    " + "  ".join(f"{m:>5s}" for m in MONTHS))
            print("    " + "  ".join(
                f"{q:5.0f}" for q in ww["order_quantities"]
            ))

            # Cost comparison
            eoq_cost = eoq["simulated_cost"]
            sm_cost = sm["total_cost"]
            ww_cost = ww["total_cost"]

            print(f"\n  Cost comparison:")
            print(f"    EOQ (simulated):  ${eoq_cost:>10,.2f}")
            print(f"    Silver-Meal:      ${sm_cost:>10,.2f}")
            print(f"    Wagner-Whitin:    ${ww_cost:>10,.2f}")

            if eoq_cost > 0:
                sm_saving = (eoq_cost - sm_cost) / eoq_cost * 100
                ww_saving = (eoq_cost - ww_cost) / eoq_cost * 100
                print(f"    Silver-Meal vs EOQ savings:      {sm_saving:+.1f}%")
                print(f"    Wagner-Whitin vs EOQ savings:    {ww_saving:+.1f}%")

        # Overall summary
        print(f"\n{'=' * 70}")
        print("OVERALL SUMMARY")
        print(f"{'=' * 70}")
        total_eoq = sum(
            r["methods"]["EOQ"]["simulated_cost"] for r in all_results.values()
        )
        total_sm = sum(
            r["methods"]["Silver-Meal"]["total_cost"]
            for r in all_results.values()
        )
        total_ww = sum(
            r["methods"]["Wagner-Whitin"]["total_cost"]
            for r in all_results.values()
        )
        print(f"  Total EOQ cost:          ${total_eoq:>10,.2f}")
        print(f"  Total Silver-Meal cost:  ${total_sm:>10,.2f}")
        print(f"  Total Wagner-Whitin cost:${total_ww:>10,.2f}")
        if total_eoq > 0:
            print(
                f"  Overall WW vs EOQ savings: "
                f"${total_eoq - total_ww:,.2f} "
                f"({(total_eoq - total_ww) / total_eoq * 100:.1f}%)"
            )

    return all_results


if __name__ == "__main__":
    solve_feed_procurement()
