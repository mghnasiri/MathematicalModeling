"""
Agricultural Feed & Fertilizer Procurement Planning Algorithms

Solves the multi-input procurement planning problem using:
1. EOQ baseline — classic formula on average demand
2. Silver-Meal — dynamic lot sizing heuristic
3. Wagner-Whitin — optimal dynamic programming

Complexity:
    - EOQ: O(1) per input
    - Silver-Meal: O(T^2) per input
    - Wagner-Whitin: O(T^2) per input

References:
    Wagner, H.M. & Whitin, T.M. (1958). Dynamic version of the economic
    lot size model. Management Science, 5(1), 89-96.
    https://doi.org/10.1287/mnsc.5.1.89

    Silver, E.A. & Meal, H.C. (1973). A heuristic for selecting lot size
    quantities for the case of a deterministic time-varying demand rate.
    Production and Inventory Management, 14(2), 64-74.
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
    "feed_proc_inst",
    os.path.join(os.path.dirname(__file__), "..", "instance.py"),
)
FeedProcurementInstance = _inst.FeedProcurementInstance
FeedProcurementSolution = _inst.FeedProcurementSolution
InputProcurementSolution = _inst.InputProcurementSolution
FarmInputProfile = _inst.FarmInputProfile


def _get_supply_chain_modules():
    """Load supply chain solver modules."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )))
    sc_dir = os.path.join(base_dir, "problems", "supply_chain")

    eoq_inst = _load_mod("eoq_inst_fp", os.path.join(sc_dir, "eoq", "instance.py"))
    eoq_formula = _load_mod(
        "eoq_formula_fp", os.path.join(sc_dir, "eoq", "exact", "eoq_formula.py")
    )
    ls_inst = _load_mod("ls_inst_fp", os.path.join(sc_dir, "lot_sizing", "instance.py"))
    silver_meal = _load_mod(
        "silver_meal_fp",
        os.path.join(sc_dir, "lot_sizing", "heuristics", "silver_meal.py"),
    )
    ww_inst = _load_mod(
        "ww_inst_fp", os.path.join(sc_dir, "wagner_whitin", "instance.py")
    )
    ww_dp = _load_mod(
        "ww_dp_fp",
        os.path.join(sc_dir, "wagner_whitin", "exact", "wagner_whitin_dp.py"),
    )
    return eoq_inst, eoq_formula, ls_inst, silver_meal, ww_inst, ww_dp


def _simulate_eoq_monthly(
    input_profile: FarmInputProfile, eoq_quantity: float
) -> tuple[np.ndarray, float]:
    """Simulate EOQ ordering on monthly demands.

    Args:
        input_profile: Agricultural input profile.
        eoq_quantity: EOQ order quantity.

    Returns:
        Tuple of (monthly_orders, total_cost).
    """
    demands = input_profile.monthly_demands
    T = len(demands)
    orders = np.zeros(T, dtype=float)
    inventory = 0.0
    total_cost = 0.0

    for t in range(T):
        if inventory < demands[t]:
            order_qty = max(eoq_quantity, demands[t] - inventory)
            orders[t] = order_qty
            inventory += order_qty
            total_cost += input_profile.ordering_cost
        inventory -= demands[t]
        total_cost += input_profile.holding_cost_per_unit_per_month * inventory

    return orders, total_cost


def eoq_procurement(
    instance: FeedProcurementInstance,
) -> FeedProcurementSolution:
    """Solve procurement using EOQ baseline for each input.

    Computes the classic EOQ on average demand, then simulates monthly
    ordering to get actual costs under seasonal demand patterns.

    Args:
        instance: FeedProcurementInstance to solve.

    Returns:
        FeedProcurementSolution with EOQ results per input.
    """
    eoq_inst, eoq_formula, _, _, _, _ = _get_supply_chain_modules()

    solutions = {}
    total_cost = 0.0

    for i, inp in enumerate(instance.inputs):
        # Create EOQ instance with annualized parameters
        eoq_instance = eoq_inst.EOQInstance(
            demand_rate=inp.avg_monthly_demand * 12.0,
            ordering_cost=inp.ordering_cost,
            holding_cost=inp.holding_cost_per_unit_per_month * 12.0,
            name=f"eoq_{inp.name}",
        )
        eoq_sol = eoq_formula.classic_eoq(eoq_instance)

        # Simulate monthly ordering
        orders, sim_cost = _simulate_eoq_monthly(inp, eoq_sol.order_quantity)
        order_periods = [t for t in range(len(orders)) if orders[t] > 0]

        sol = InputProcurementSolution(
            input_name=inp.name,
            method="EOQ",
            total_cost=sim_cost,
            order_quantities=orders,
            order_periods=order_periods,
            num_orders=len(order_periods),
        )
        solutions[inp.name] = sol
        total_cost += sim_cost

    return FeedProcurementSolution(
        input_solutions=solutions,
        total_cost=total_cost,
        method="EOQ",
    )


def silver_meal_procurement(
    instance: FeedProcurementInstance,
) -> FeedProcurementSolution:
    """Solve procurement using Silver-Meal dynamic lot sizing.

    Args:
        instance: FeedProcurementInstance to solve.

    Returns:
        FeedProcurementSolution with Silver-Meal results per input.
    """
    _, _, ls_inst, silver_meal_mod, _, _ = _get_supply_chain_modules()

    solutions = {}
    total_cost = 0.0

    for inp in instance.inputs:
        T = instance.horizon
        ls_instance = ls_inst.LotSizingInstance(
            T=T,
            demands=inp.monthly_demands.copy(),
            ordering_costs=np.full(T, inp.ordering_cost),
            holding_costs=np.full(T, inp.holding_cost_per_unit_per_month),
            name=f"sm_{inp.name}",
        )
        sm_sol = silver_meal_mod.silver_meal(ls_instance)

        sol = InputProcurementSolution(
            input_name=inp.name,
            method="Silver-Meal",
            total_cost=sm_sol.total_cost,
            order_quantities=sm_sol.order_quantities,
            order_periods=sm_sol.order_periods,
            num_orders=len(sm_sol.order_periods),
        )
        solutions[inp.name] = sol
        total_cost += sm_sol.total_cost

    return FeedProcurementSolution(
        input_solutions=solutions,
        total_cost=total_cost,
        method="Silver-Meal",
    )


def wagner_whitin_procurement(
    instance: FeedProcurementInstance,
) -> FeedProcurementSolution:
    """Solve procurement using Wagner-Whitin optimal DP.

    Args:
        instance: FeedProcurementInstance to solve.

    Returns:
        FeedProcurementSolution with Wagner-Whitin results per input.
    """
    _, _, _, _, ww_inst, ww_dp = _get_supply_chain_modules()

    solutions = {}
    total_cost = 0.0

    for inp in instance.inputs:
        T = instance.horizon
        ww_instance = ww_inst.WagnerWhitinInstance(
            T=T,
            demands=inp.monthly_demands.copy(),
            ordering_costs=np.full(T, inp.ordering_cost),
            holding_costs=np.full(T, inp.holding_cost_per_unit_per_month),
            name=f"ww_{inp.name}",
        )
        ww_sol = ww_dp.wagner_whitin_dp(ww_instance)

        sol = InputProcurementSolution(
            input_name=inp.name,
            method="Wagner-Whitin",
            total_cost=ww_sol.total_cost,
            order_quantities=ww_sol.order_quantities,
            order_periods=ww_sol.order_periods,
            num_orders=len(ww_sol.order_periods),
        )
        solutions[inp.name] = sol
        total_cost += ww_sol.total_cost

    return FeedProcurementSolution(
        input_solutions=solutions,
        total_cost=total_cost,
        method="Wagner-Whitin",
    )


if __name__ == "__main__":
    inst = FeedProcurementInstance.quebec_dairy_farm()
    print("=== Agricultural Feed & Fertilizer Procurement ===\n")

    for method_name, solver in [
        ("EOQ", eoq_procurement),
        ("Silver-Meal", silver_meal_procurement),
        ("Wagner-Whitin", wagner_whitin_procurement),
    ]:
        sol = solver(inst)
        print(f"{method_name}: total cost = ${sol.total_cost:,.2f}")
        for name, isol in sol.input_solutions.items():
            print(f"  {name}: ${isol.total_cost:,.2f}, {isol.num_orders} orders")
        print()
