"""
Real-World Application: Crop Rotation and Land Allocation Optimization.

Domain: Agricultural planning / Precision farming
Models: Linear Programming (revenue maximization) + Multi-Objective
        (revenue vs soil health via epsilon-constraint)

Scenario:
    A farm has 6 fields (varying acreage: 10-40 hectares, ~150 ha total)
    and 5 candidate crops (corn, wheat, soybeans, vegetables, hay). Each
    crop-field combination has specific yield, revenue, water requirement,
    labor requirement, and nitrogen fixation effect determined by soil
    quality and crop characteristics.

    Questions answered:
    1. LP: What crop allocation maximizes total revenue subject to water,
       labor, diversity, and rotation constraints? What are the shadow
       prices on water and labor?
    2. Multi-Objective: What are the Pareto-efficient allocations trading
       off revenue against soil health (nitrogen balance)?

Real-world considerations modeled:
    - Field-specific yields reflecting heterogeneous soil quality
    - Water budget constraint (irrigation infrastructure limit)
    - Seasonal labor constraint (available worker-hours)
    - Crop diversity requirement (no single crop > 40% of total area)
    - Rotation constraint (certain fields cannot repeat last year's crop)
    - Nitrogen fixation: legumes (soybeans) add nitrogen, heavy feeders
      (corn, vegetables) deplete it

Industry context:
    Crop rotation planning is a critical annual decision for commercial
    farms. Optimal allocation can increase revenue by 10-20% while
    maintaining long-term soil fertility. The USDA estimates that proper
    crop rotation reduces nitrogen fertilizer needs by 25-50% for
    subsequent crops (Bullock, 1992). Multi-objective models help farmers
    balance short-term profit against long-term sustainability.

References:
    Bullock, D. G. (1992). Crop rotation. Critical Reviews in Plant
    Sciences, 11(4), 309-326.
    https://doi.org/10.1080/07352689209382349

    Dury, J., Schaller, N., Garcia, F., Reynaud, A. & Bergez, J. E.
    (2012). Models to support cropping plan and crop rotation decisions.
    A review. Agronomy for Sustainable Development, 32(2), 567-580.
    https://doi.org/10.1007/s13593-011-0037-x

    Detlefsen, N. K. & Jensen, A. L. (2007). Modelling optimal crop
    sequences using network flows. Agricultural Systems, 94(2), 566-572.
    https://doi.org/10.1016/j.agsy.2007.02.002
"""

from __future__ import annotations

import sys
import os
import importlib.util

import numpy as np
from scipy.optimize import linprog


def _load_mod(name: str, filepath: str):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# -- Domain Data ---------------------------------------------------------------

CROPS = ["corn", "wheat", "soybeans", "vegetables", "hay"]

FIELDS = [
    {"name": "North Ridge",    "hectares": 35, "soil_quality": "high"},
    {"name": "East Valley",    "hectares": 25, "soil_quality": "medium"},
    {"name": "South Flat",     "hectares": 40, "soil_quality": "high"},
    {"name": "West Hill",      "hectares": 15, "soil_quality": "low"},
    {"name": "Central Plot",   "hectares": 20, "soil_quality": "medium"},
    {"name": "Creek Bottom",   "hectares": 10, "soil_quality": "high"},
]

TOTAL_HECTARES = sum(f["hectares"] for f in FIELDS)  # 145 ha

# Revenue per ton ($/ton)
REVENUE_PER_TON = {
    "corn": 180,
    "wheat": 220,
    "soybeans": 400,
    "vegetables": 600,
    "hay": 120,
}

# Yield (tons/ha) by crop and soil quality
YIELD_BY_SOIL = {
    "corn":       {"high": 10.5, "medium": 8.5, "low": 6.0},
    "wheat":      {"high": 5.0,  "medium": 4.0, "low": 3.0},
    "soybeans":   {"high": 3.5,  "medium": 3.0, "low": 2.2},
    "vegetables": {"high": 18.0, "medium": 14.0, "low": 9.0},
    "hay":        {"high": 8.0,  "medium": 7.0, "low": 5.5},
}

# Water requirement (m^3/ha)
WATER_PER_HA = {
    "corn": 5500,
    "wheat": 3000,
    "soybeans": 4000,
    "vegetables": 7000,
    "hay": 2500,
}

# Labor requirement (hours/ha)
LABOR_PER_HA = {
    "corn": 35,
    "wheat": 25,
    "soybeans": 20,
    "vegetables": 120,
    "hay": 15,
}

# Nitrogen fixation effect (kg N/ha): positive = adds to soil, negative = depletes
NITROGEN_EFFECT = {
    "corn": -80,
    "wheat": -30,
    "soybeans": 60,
    "vegetables": -50,
    "hay": 10,
}

# Resource budgets
WATER_BUDGET = 500_000   # m^3 total
LABOR_BUDGET = 8_000     # hours total

# Crop diversity: no single crop on more than 40% of total area
MAX_CROP_FRACTION = 0.40

# Rotation constraints: (field_index, crop_name) pairs that are forbidden
# because that field grew that crop last year
ROTATION_FORBIDDEN = [
    (0, "corn"),       # North Ridge grew corn last year
    (2, "wheat"),      # South Flat grew wheat last year
    (4, "soybeans"),   # Central Plot grew soybeans last year
]


def _get_yield(crop: str, field_idx: int) -> float:
    """Get expected yield (tons/ha) for a crop on a specific field."""
    soil = FIELDS[field_idx]["soil_quality"]
    return YIELD_BY_SOIL[crop][soil]


def _get_revenue(crop: str, field_idx: int) -> float:
    """Get expected revenue ($/ha) for a crop on a specific field."""
    return _get_yield(crop, field_idx) * REVENUE_PER_TON[crop]


def create_crop_allocation_instance() -> dict:
    """Create the crop allocation data structures.

    Decision variables: x[f, c] in {0, 1} for field f, crop c.
    Relaxed to continuous [0, 1] for LP; assignment constraints
    ensure each field gets exactly one crop.

    Returns:
        Dictionary with all problem data and LP formulation components.
    """
    n_fields = len(FIELDS)
    n_crops = len(CROPS)
    n_vars = n_fields * n_crops  # x[f*n_crops + c]

    def var_idx(f: int, c: int) -> int:
        return f * n_crops + c

    # Objective: maximize revenue => minimize negative revenue
    c_obj = np.zeros(n_vars)
    for f in range(n_fields):
        ha = FIELDS[f]["hectares"]
        for ci, crop in enumerate(CROPS):
            c_obj[var_idx(f, ci)] = -_get_revenue(crop, f) * ha

    # Revenue matrix (for reporting)
    revenue_matrix = np.zeros((n_fields, n_crops))
    for f in range(n_fields):
        ha = FIELDS[f]["hectares"]
        for ci, crop in enumerate(CROPS):
            revenue_matrix[f, ci] = _get_revenue(crop, f) * ha

    # Nitrogen matrix (for multi-objective)
    nitrogen_matrix = np.zeros((n_fields, n_crops))
    for f in range(n_fields):
        ha = FIELDS[f]["hectares"]
        for ci, crop in enumerate(CROPS):
            nitrogen_matrix[f, ci] = NITROGEN_EFFECT[crop] * ha

    # Inequality constraints (A_ub x <= b_ub)
    ub_rows = []
    ub_rhs = []
    constraint_names = []

    # 1. Water budget: sum over all (f, c) of water_per_ha[c] * ha[f] * x[f,c] <= WATER_BUDGET
    water_row = np.zeros(n_vars)
    for f in range(n_fields):
        ha = FIELDS[f]["hectares"]
        for ci, crop in enumerate(CROPS):
            water_row[var_idx(f, ci)] = WATER_PER_HA[crop] * ha
    ub_rows.append(water_row)
    ub_rhs.append(float(WATER_BUDGET))
    constraint_names.append("Water budget")

    # 2. Labor budget: sum over all (f, c) of labor_per_ha[c] * ha[f] * x[f,c] <= LABOR_BUDGET
    labor_row = np.zeros(n_vars)
    for f in range(n_fields):
        ha = FIELDS[f]["hectares"]
        for ci, crop in enumerate(CROPS):
            labor_row[var_idx(f, ci)] = LABOR_PER_HA[crop] * ha
    ub_rows.append(labor_row)
    ub_rhs.append(float(LABOR_BUDGET))
    constraint_names.append("Labor budget")

    # 3. Crop diversity: for each crop, sum of ha[f] * x[f,c] <= MAX_CROP_FRACTION * TOTAL_HECTARES
    max_area = MAX_CROP_FRACTION * TOTAL_HECTARES
    for ci, crop in enumerate(CROPS):
        row = np.zeros(n_vars)
        for f in range(n_fields):
            row[var_idx(f, ci)] = FIELDS[f]["hectares"]
        ub_rows.append(row)
        ub_rhs.append(max_area)
        constraint_names.append(f"Diversity ({crop} <= {max_area:.0f} ha)")

    A_ub = np.array(ub_rows)
    b_ub = np.array(ub_rhs)

    # Equality constraints: each field grows exactly one crop
    # sum_c x[f, c] = 1 for each field f
    eq_rows = []
    eq_rhs = []
    for f in range(n_fields):
        row = np.zeros(n_vars)
        for ci in range(n_crops):
            row[var_idx(f, ci)] = 1.0
        eq_rows.append(row)
        eq_rhs.append(1.0)

    A_eq = np.array(eq_rows)
    b_eq = np.array(eq_rhs)

    # Bounds: 0 <= x[f,c] <= 1 (relaxed binary)
    # Rotation-forbidden pairs get upper bound = 0
    bounds = []
    for f in range(n_fields):
        for ci, crop in enumerate(CROPS):
            forbidden = any(
                ff == f and cc == crop for ff, cc in ROTATION_FORBIDDEN
            )
            bounds.append((0.0, 0.0 if forbidden else 1.0))

    return {
        "n_fields": n_fields,
        "n_crops": n_crops,
        "n_vars": n_vars,
        "c_obj": c_obj,
        "A_ub": A_ub,
        "b_ub": b_ub,
        "A_eq": A_eq,
        "b_eq": b_eq,
        "bounds": bounds,
        "constraint_names": constraint_names,
        "revenue_matrix": revenue_matrix,
        "nitrogen_matrix": nitrogen_matrix,
        "var_idx": var_idx,
    }


def solve_crop_allocation(verbose: bool = True) -> dict:
    """Solve crop allocation using LP and multi-objective optimization.

    Model 1 (LP): Maximize total revenue subject to water, labor,
    diversity, and rotation constraints with sensitivity analysis.

    Model 2 (Multi-Objective): Epsilon-constraint method trading off
    revenue (objective 1) against soil health / nitrogen balance
    (objective 2).

    Args:
        verbose: Whether to print detailed results.

    Returns:
        Dictionary with LP and Pareto front results.
    """
    data = create_crop_allocation_instance()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Load LP module
    lp_dir = os.path.join(
        base_dir, "problems", "continuous", "linear_programming"
    )
    lp_inst_mod = _load_mod(
        "lp_inst_ag", os.path.join(lp_dir, "instance.py")
    )
    lp_solver_mod = _load_mod(
        "lp_solv_ag", os.path.join(lp_dir, "exact", "lp_solver.py")
    )

    results = {}

    # == Model 1: Revenue-Maximizing LP ========================================

    lp_instance = lp_inst_mod.LPInstance(
        n=data["n_vars"],
        c=data["c_obj"],
        A_ub=data["A_ub"],
        b_ub=data["b_ub"],
        A_eq=data["A_eq"],
        b_eq=data["b_eq"],
        bounds=data["bounds"],
    )

    lp_sol = lp_solver_mod.solve_lp(lp_instance)
    sensitivity = lp_solver_mod.sensitivity_report(lp_instance, lp_sol)

    # Decode solution
    var_idx = data["var_idx"]
    n_fields = data["n_fields"]
    n_crops = data["n_crops"]
    allocation = {}
    total_revenue = 0.0
    total_water = 0.0
    total_labor = 0.0
    total_nitrogen = 0.0

    for f in range(n_fields):
        best_crop = -1
        best_val = -1.0
        for ci in range(n_crops):
            val = lp_sol.x[var_idx(f, ci)]
            if val > best_val:
                best_val = val
                best_crop = ci
        allocation[f] = {
            "field": FIELDS[f]["name"],
            "crop": CROPS[best_crop],
            "hectares": FIELDS[f]["hectares"],
            "revenue": data["revenue_matrix"][f, best_crop],
            "fraction": best_val,
        }
        ha = FIELDS[f]["hectares"]
        total_revenue += data["revenue_matrix"][f, best_crop] * best_val
        total_water += WATER_PER_HA[CROPS[best_crop]] * ha * best_val
        total_labor += LABOR_PER_HA[CROPS[best_crop]] * ha * best_val
        total_nitrogen += NITROGEN_EFFECT[CROPS[best_crop]] * ha * best_val

    results["LP"] = {
        "success": lp_sol.success,
        "total_revenue": -lp_sol.objective,
        "allocation": allocation,
        "total_water": total_water,
        "total_labor": total_labor,
        "total_nitrogen": total_nitrogen,
        "sensitivity": sensitivity,
    }

    # == Model 2: Multi-Objective (Revenue vs Soil Health) =====================
    # Epsilon-constraint: maximize revenue subject to nitrogen >= epsilon
    # Vary epsilon from min achievable nitrogen to max achievable nitrogen

    pareto_front = []
    pareto_allocations = []

    # Find nitrogen range by solving for max and min nitrogen
    nitrogen_obj = np.zeros(data["n_vars"])
    for f in range(n_fields):
        ha = FIELDS[f]["hectares"]
        for ci, crop in enumerate(CROPS):
            nitrogen_obj[var_idx(f, ci)] = -NITROGEN_EFFECT[crop] * ha

    # Max nitrogen (min of negative nitrogen objective)
    res_max_n = linprog(
        c=nitrogen_obj,
        A_ub=data["A_ub"],
        b_ub=data["b_ub"],
        A_eq=data["A_eq"],
        b_eq=data["b_eq"],
        bounds=data["bounds"],
        method="highs",
    )
    max_nitrogen = -res_max_n.fun if res_max_n.success else 0.0

    # Min nitrogen (max of negative nitrogen = min nitrogen)
    res_min_n = linprog(
        c=-nitrogen_obj,
        A_ub=data["A_ub"],
        b_ub=data["b_ub"],
        A_eq=data["A_eq"],
        b_eq=data["b_eq"],
        bounds=data["bounds"],
        method="highs",
    )
    min_nitrogen = res_min_n.fun if res_min_n.success else 0.0

    n_points = 15
    epsilons = np.linspace(min_nitrogen, max_nitrogen, n_points)

    for eps in epsilons:
        # Add constraint: sum nitrogen >= eps
        # i.e. -sum nitrogen <= -eps
        nitrogen_row = np.zeros(data["n_vars"])
        for f in range(n_fields):
            ha = FIELDS[f]["hectares"]
            for ci, crop in enumerate(CROPS):
                nitrogen_row[var_idx(f, ci)] = -NITROGEN_EFFECT[crop] * ha

        A_ub_ext = np.vstack([data["A_ub"], nitrogen_row.reshape(1, -1)])
        b_ub_ext = np.append(data["b_ub"], -eps)

        res = linprog(
            c=data["c_obj"],
            A_ub=A_ub_ext,
            b_ub=b_ub_ext,
            A_eq=data["A_eq"],
            b_eq=data["b_eq"],
            bounds=data["bounds"],
            method="highs",
        )

        if res.success:
            revenue = -res.fun
            # Compute actual nitrogen
            nitrogen = 0.0
            alloc = {}
            for f in range(n_fields):
                best_crop = -1
                best_val = -1.0
                for ci in range(n_crops):
                    val = res.x[var_idx(f, ci)]
                    if val > best_val:
                        best_val = val
                        best_crop = ci
                ha = FIELDS[f]["hectares"]
                nitrogen += NITROGEN_EFFECT[CROPS[best_crop]] * ha * best_val
                alloc[f] = {
                    "field": FIELDS[f]["name"],
                    "crop": CROPS[best_crop],
                }

            # Check if dominated by existing points
            dominated = False
            for prev_rev, prev_nit, _ in pareto_front:
                if prev_rev >= revenue - 1e-6 and prev_nit >= nitrogen - 1e-6:
                    dominated = True
                    break

            if not dominated:
                # Remove any points now dominated by this one
                pareto_front = [
                    (r, n, a) for r, n, a in pareto_front
                    if not (revenue >= r - 1e-6 and nitrogen >= n - 1e-6
                            and (revenue > r + 1e-6 or nitrogen > n + 1e-6))
                ]
                pareto_front.append((revenue, nitrogen, alloc))

    # Sort Pareto front by revenue
    pareto_front.sort(key=lambda x: x[0])

    results["Pareto"] = {
        "n_points": len(pareto_front),
        "front": [(rev, nit) for rev, nit, _ in pareto_front],
        "allocations": [alloc for _, _, alloc in pareto_front],
        "nitrogen_range": (min_nitrogen, max_nitrogen),
    }

    # == Print Results =========================================================

    if verbose:
        print("=" * 70)
        print("CROP ROTATION & LAND ALLOCATION OPTIMIZATION")
        print(f"  {n_fields} fields ({TOTAL_HECTARES} ha total), "
              f"{n_crops} candidate crops")
        print("=" * 70)

        print("\n--- MODEL 1: REVENUE-MAXIMIZING LP ---")
        print(f"  Status: {'Optimal' if lp_sol.success else 'Infeasible'}")
        print(f"  Total revenue: ${results['LP']['total_revenue']:,.0f}")
        print(f"  Water used: {total_water:,.0f} / {WATER_BUDGET:,} m^3 "
              f"({total_water / WATER_BUDGET * 100:.1f}%)")
        print(f"  Labor used: {total_labor:,.0f} / {LABOR_BUDGET:,} hours "
              f"({total_labor / LABOR_BUDGET * 100:.1f}%)")
        print(f"  Nitrogen balance: {total_nitrogen:+,.0f} kg N")

        print("\n  Allocation:")
        for f in range(n_fields):
            a = allocation[f]
            print(f"    {a['field']:16s} ({a['hectares']:2d} ha) -> "
                  f"{a['crop']:12s}  revenue=${a['revenue']:>10,.0f}")

        # Sensitivity analysis
        print("\n  Sensitivity Analysis (shadow prices):")
        if sensitivity.get("shadow_prices"):
            sp = sensitivity["shadow_prices"]
            for i, name in enumerate(data["constraint_names"]):
                if abs(sp[i]) > 1e-6:
                    print(f"    {name}: {sp[i]:+.2f} $/unit")

        binding = sensitivity.get("binding_constraints", [])
        if binding:
            print("\n  Binding constraints:")
            for i in binding:
                print(f"    {data['constraint_names'][i]}")

        # Pareto front
        print("\n--- MODEL 2: PARETO FRONT (Revenue vs Soil Health) ---")
        print(f"  Nitrogen range: [{min_nitrogen:+,.0f}, "
              f"{max_nitrogen:+,.0f}] kg N")
        print(f"  {results['Pareto']['n_points']} Pareto-efficient "
              f"allocations found")
        print(f"\n  {'Revenue ($)':>14s}  {'Nitrogen (kg)':>14s}  Allocation")
        print(f"  {'-' * 14}  {'-' * 14}  {'-' * 30}")
        for rev, nit, alloc in pareto_front:
            crops_summary = ", ".join(
                f"{alloc[f]['crop']}" for f in sorted(alloc.keys())
            )
            print(f"  {rev:>14,.0f}  {nit:>+14,.0f}  {crops_summary}")

        # Highlight trade-off
        if len(pareto_front) >= 2:
            low_rev = pareto_front[0]
            high_rev = pareto_front[-1]
            rev_gain = high_rev[0] - low_rev[0]
            nit_loss = low_rev[1] - high_rev[1]
            print(f"\n  Trade-off: gaining ${rev_gain:,.0f} in revenue costs "
                  f"{nit_loss:+,.0f} kg N in soil health")

    return results


if __name__ == "__main__":
    solve_crop_allocation()
