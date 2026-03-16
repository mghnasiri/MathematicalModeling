"""
Real-World Application: Agricultural Water Resource Allocation.

Domain: Water resource allocation to fields during growing season
Model: Linear Programming — allocate limited water from a reservoir to
       fields to minimize total irrigation cost while meeting minimum
       crop water requirements.

Scenario:
    A farm cooperative manages a reservoir with 5000 cubic meters of water
    that must be allocated across 8 fields with different crops during a
    critical irrigation period. Each field has:
    - A minimum water requirement based on crop type and growth stage
    - A maximum useful amount (beyond which water is wasted)
    - A cost per cubic meter that varies by pipe distance and elevation

    The objective is to minimize total pumping and delivery cost while
    ensuring every field receives at least its minimum water requirement.

Real-world considerations modeled:
    - Pipe distance and elevation affect delivery cost per field
    - Minimum crop water requirements based on evapotranspiration (ET)
    - Maximum useful water (soil saturation limits)
    - Total reservoir capacity constraint
    - Different crop sensitivities to water stress

Industry context:
    Agriculture accounts for 70% of global freshwater withdrawals. Optimal
    water allocation can increase water use efficiency by 20-35%, critical
    in water-scarce regions. Deficit irrigation strategies intentionally
    under-irrigate low-value crops to redirect water to high-value ones
    (English et al., 2002). LP-based irrigation scheduling is used in
    large-scale projects like Israel's National Water Carrier.

References:
    English, M.J., Solomon, K.H. & Hoffman, G.J. (2002). A paradigm
    shift in irrigation management. Journal of Irrigation and Drainage
    Engineering, 128(5), 267-277.
    https://doi.org/10.1061/(ASCE)0733-9437(2002)128:5(267)

    Harou, J.J., Pulido-Velazquez, M., Rosenberg, D.E., Medellin-Azuara,
    J., Lund, J.R. & Howitt, R.E. (2009). Hydro-economic models:
    Concepts, design, applications, and future prospects. Journal of
    Hydrology, 375(3-4), 627-643.
    https://doi.org/10.1016/j.jhydrol.2009.06.037
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

RESERVOIR_CAPACITY = 5000  # cubic meters available for this irrigation cycle

# 8 fields with different crops, distances, and water needs
FIELDS = [
    {"name": "Field A — Winter Wheat",   "crop": "wheat",    "area_ha": 12,
     "distance_km": 0.8, "elevation_m": 5,  "growth_stage": "grain_fill"},
    {"name": "Field B — Corn (Maize)",    "crop": "corn",     "area_ha": 10,
     "distance_km": 1.5, "elevation_m": 12, "growth_stage": "tasseling"},
    {"name": "Field C — Soybeans",        "crop": "soybean",  "area_ha": 15,
     "distance_km": 2.0, "elevation_m": 8,  "growth_stage": "pod_fill"},
    {"name": "Field D — Alfalfa",         "crop": "alfalfa",  "area_ha": 20,
     "distance_km": 0.5, "elevation_m": 2,  "growth_stage": "regrowth"},
    {"name": "Field E — Tomatoes",        "crop": "tomato",   "area_ha": 3,
     "distance_km": 0.3, "elevation_m": 0,  "growth_stage": "fruiting"},
    {"name": "Field F — Potatoes",        "crop": "potato",   "area_ha": 5,
     "distance_km": 1.2, "elevation_m": 6,  "growth_stage": "tuber_bulk"},
    {"name": "Field G — Onions",          "crop": "onion",    "area_ha": 4,
     "distance_km": 1.8, "elevation_m": 10, "growth_stage": "bulbing"},
    {"name": "Field H — Canola",          "crop": "canola",   "area_ha": 14,
     "distance_km": 2.5, "elevation_m": 15, "growth_stage": "flowering"},
]

# Crop water requirement (m3/ha) for current growth stage — ET-based
CROP_WATER_NEEDS = {
    ("wheat", "grain_fill"):    40,
    ("corn", "tasseling"):      65,
    ("soybean", "pod_fill"):    50,
    ("alfalfa", "regrowth"):    35,
    ("tomato", "fruiting"):     70,
    ("potato", "tuber_bulk"):   55,
    ("onion", "bulbing"):       60,
    ("canola", "flowering"):    45,
}

# Maximum useful water (m3/ha) — beyond this, runoff/waterlogging
CROP_MAX_WATER = {
    "wheat": 80,
    "corn": 120,
    "soybean": 90,
    "alfalfa": 70,
    "tomato": 100,
    "potato": 95,
    "onion": 85,
    "canola": 75,
}

# Base pumping cost: $/m3 at reservoir outlet
BASE_COST = 0.05

# Cost factors
DISTANCE_COST_FACTOR = 0.03   # $/m3 per km of pipe
ELEVATION_COST_FACTOR = 0.008  # $/m3 per meter of elevation


def create_water_allocation_instance() -> dict:
    """Create a water allocation LP instance.

    Returns:
        Dictionary with LP formulation data and metadata.
    """
    n = len(FIELDS)

    # Cost per m3 for each field (pumping + distance + elevation)
    costs = np.array([
        BASE_COST
        + DISTANCE_COST_FACTOR * field["distance_km"]
        + ELEVATION_COST_FACTOR * field["elevation_m"]
        for field in FIELDS
    ])

    # Minimum water requirement per field (m3)
    min_water = np.array([
        CROP_WATER_NEEDS[(field["crop"], field["growth_stage"])] * field["area_ha"]
        for field in FIELDS
    ])

    # Maximum useful water per field (m3)
    max_water = np.array([
        CROP_MAX_WATER[field["crop"]] * field["area_ha"]
        for field in FIELDS
    ])

    return {
        "n": n,
        "costs": costs,
        "min_water": min_water,
        "max_water": max_water,
        "reservoir_capacity": RESERVOIR_CAPACITY,
        "fields": FIELDS,
    }


def solve_water_allocation(verbose: bool = True) -> dict:
    """Solve the water allocation problem using LP.

    Minimizes total irrigation cost subject to:
    - Each field gets at least its minimum water requirement
    - Each field gets at most its maximum useful water
    - Total water used does not exceed reservoir capacity

    Returns:
        Dictionary with LP solution and analysis.
    """
    data = create_water_allocation_instance()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    lp_dir = os.path.join(base_dir, "problems", "continuous", "linear_programming")

    lp_inst_mod = _load_mod(
        "lp_inst_water", os.path.join(lp_dir, "instance.py")
    )
    lp_solve_mod = _load_mod(
        "lp_solve_water", os.path.join(lp_dir, "exact", "lp_solver.py")
    )

    n = data["n"]
    costs = data["costs"]
    min_water = data["min_water"]
    max_water = data["max_water"]
    capacity = data["reservoir_capacity"]

    # LP: min c^T x
    # s.t. sum(x) <= reservoir capacity (inequality)
    #      x_i >= min_water_i (lower bounds)
    #      x_i <= max_water_i (upper bounds)

    # Total capacity constraint: sum(x_i) <= capacity
    A_ub = np.ones((1, n))
    b_ub = np.array([float(capacity)])

    bounds = [
        (float(min_water[i]), float(max_water[i]))
        for i in range(n)
    ]

    lp_instance = lp_inst_mod.LPInstance(
        n=n,
        c=costs,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds,
    )

    lp_sol = lp_solve_mod.solve_lp(lp_instance)

    results = {
        "success": lp_sol.success,
        "total_cost": lp_sol.objective if lp_sol.success else None,
        "allocations": lp_sol.x.tolist() if lp_sol.success else [],
    }

    # Also solve a scenario with reduced water (drought: 70% capacity)
    drought_capacity = int(capacity * 0.70)
    b_ub_drought = np.array([float(drought_capacity)])

    lp_drought = lp_inst_mod.LPInstance(
        n=n,
        c=costs,
        A_ub=np.ones((1, n)),
        b_ub=b_ub_drought,
        bounds=bounds,
    )

    drought_sol = lp_solve_mod.solve_lp(lp_drought)
    results["drought"] = {
        "success": drought_sol.success,
        "capacity": drought_capacity,
        "total_cost": drought_sol.objective if drought_sol.success else None,
        "allocations": drought_sol.x.tolist() if drought_sol.success else [],
    }

    if verbose:
        print("=" * 70)
        print("AGRICULTURAL WATER RESOURCE ALLOCATION")
        print(f"  {n} fields, reservoir capacity: {capacity} m3")
        print(f"  Total minimum requirement: {min_water.sum():.0f} m3")
        print(f"  Total maximum useful: {max_water.sum():.0f} m3")
        print("=" * 70)

        print("\n  Field details:")
        for i, field in enumerate(data["fields"]):
            print(f"    {i}. {field['name']}")
            print(f"       Need: {min_water[i]:.0f}-{max_water[i]:.0f} m3, "
                  f"Cost: ${costs[i]:.4f}/m3, "
                  f"Distance: {field['distance_km']}km, "
                  f"Elev: {field['elevation_m']}m")

        if lp_sol.success:
            print(f"\n--- Normal Season (capacity = {capacity} m3) ---")
            print(f"  Total irrigation cost: ${lp_sol.objective:,.2f}")
            total_allocated = sum(lp_sol.x)
            print(f"  Total water allocated: {total_allocated:,.0f} m3 "
                  f"({total_allocated/capacity*100:.1f}% of reservoir)")
            print(f"\n  Allocation per field:")
            for i, field in enumerate(data["fields"]):
                alloc = lp_sol.x[i]
                pct_of_max = alloc / max_water[i] * 100
                field_cost = alloc * costs[i]
                surplus = alloc - min_water[i]
                print(f"    {field['name'][:35]:35s} "
                      f"{alloc:7.0f} m3 ({pct_of_max:4.1f}% of max) "
                      f"surplus: {surplus:5.0f} m3  cost: ${field_cost:.2f}")

            # Water use efficiency metric
            total_min = min_water.sum()
            efficiency = total_min / total_allocated * 100 if total_allocated > 0 else 0
            print(f"\n  Water use efficiency: {efficiency:.1f}% "
                  f"(min required / allocated)")

        if drought_sol.success:
            print(f"\n--- Drought Scenario (capacity = {drought_capacity} m3, "
                  f"70% of normal) ---")
            print(f"  Total irrigation cost: ${drought_sol.objective:,.2f}")
            total_drought = sum(drought_sol.x)
            print(f"  Total water allocated: {total_drought:,.0f} m3")
            print(f"\n  Allocation changes vs normal:")
            for i, field in enumerate(data["fields"]):
                normal = lp_sol.x[i]
                drought = drought_sol.x[i]
                change = drought - normal
                pct_change = (change / normal * 100) if normal > 0 else 0
                if abs(change) > 0.5:
                    print(f"    {field['name'][:35]:35s} "
                          f"{normal:7.0f} -> {drought:7.0f} m3 "
                          f"({pct_change:+.1f}%)")
        elif not drought_sol.success:
            print(f"\n--- Drought Scenario: INFEASIBLE ---")
            print(f"  Cannot meet minimum requirements with {drought_capacity} m3")
            print(f"  Shortage: {min_water.sum() - drought_capacity:.0f} m3")

    return results


if __name__ == "__main__":
    solve_water_allocation()
