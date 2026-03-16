"""
Real-World Application: Crop-to-Field Assignment for Yield Optimization.

Domain: Precision agriculture / Farm field management
Model: Linear Assignment Problem (Hungarian method)

Scenario:
    A 2000-acre diversified farm operation has 8 distinct fields with
    varying characteristics: soil type (clay, loam, sandy), organic
    matter content, drainage quality, sunlight exposure, irrigation
    access, and historical pest pressure. The farm grows 8 crop types
    and must assign exactly one crop to each field for the upcoming
    season to maximize total expected yield (equivalently, minimize
    the negative-yield cost matrix).

    Each crop-field combination has a different expected yield based
    on agronomic compatibility. For example, corn thrives in deep
    loam with good drainage, while rice requires heavy clay with
    poor drainage. Sunflower tolerates sandy soil but needs full sun.

    The assignment must be one-to-one: each field gets exactly one
    crop, and each crop is planted in exactly one field (crop rotation
    and diversification requirements).

Real-world considerations modeled:
    - Soil-crop compatibility (texture, pH, organic matter)
    - Water availability and drainage characteristics
    - Sunlight and microclimate effects on yield
    - One-to-one assignment (crop rotation constraints)
    - Yield maximization through optimal matching

Industry context:
    Proper crop-field matching can increase yields by 15-25% compared
    to arbitrary assignment (Basso et al., 2013). Precision agriculture
    and variable-rate technology have made field-level optimization
    increasingly important. The US crop production sector generates
    over $200 billion annually, where even small yield improvements
    translate to significant revenue gains.

References:
    Basso, B., Dumont, B., Cammarano, D., Pezzuolo, A., Marinello, F.
    & Sartori, L. (2013). Environmental and economic benefits of
    variable rate nitrogen fertilization in a nitrate vulnerable zone.
    Science of The Total Environment, 545-546, 227-235.
    https://doi.org/10.1016/j.scitotenv.2015.12.104

    Kuhn, H.W. (1955). The Hungarian method for the assignment problem.
    Naval Research Logistics Quarterly, 2(1-2), 83-97.
    https://doi.org/10.1002/nav.3800020109

    Schirrmann, M., Giebel, A., Gleiniger, F., Dammer, K.H. (2016).
    Monitoring agronomic parameters of winter wheat crops with
    low-cost UAV imagery. Remote Sensing, 8(9), 706.
    https://doi.org/10.3390/rs8090706
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

FIELDS = [
    {"name": "North Ridge",     "acres": 320, "soil": "loam",  "drainage": "good",
     "sun": "full",    "irrigation": True,  "organic_matter": 3.8},
    {"name": "River Bottom",    "acres": 280, "soil": "clay",  "drainage": "poor",
     "sun": "partial", "irrigation": True,  "organic_matter": 4.5},
    {"name": "West Prairie",    "acres": 250, "soil": "sandy", "drainage": "good",
     "sun": "full",    "irrigation": False, "organic_matter": 2.1},
    {"name": "Hilltop East",    "acres": 200, "soil": "loam",  "drainage": "good",
     "sun": "full",    "irrigation": True,  "organic_matter": 3.2},
    {"name": "South Meadow",    "acres": 300, "soil": "clay",  "drainage": "moderate",
     "sun": "full",    "irrigation": True,  "organic_matter": 4.0},
    {"name": "Creek Bend",      "acres": 180, "soil": "loam",  "drainage": "moderate",
     "sun": "partial", "irrigation": True,  "organic_matter": 3.5},
    {"name": "Sandy Flats",     "acres": 220, "soil": "sandy", "drainage": "good",
     "sun": "full",    "irrigation": False, "organic_matter": 1.8},
    {"name": "Central Valley",  "acres": 250, "soil": "loam",  "drainage": "good",
     "sun": "full",    "irrigation": True,  "organic_matter": 3.9},
]

CROPS = [
    {"name": "Corn",       "ideal_soil": "loam",  "needs_irrigation": True,
     "needs_sun": "full",    "base_yield": 180, "unit": "bu/acre"},
    {"name": "Wheat",      "ideal_soil": "loam",  "needs_irrigation": False,
     "needs_sun": "full",    "base_yield": 55,  "unit": "bu/acre"},
    {"name": "Soybean",    "ideal_soil": "clay",  "needs_irrigation": False,
     "needs_sun": "full",    "base_yield": 50,  "unit": "bu/acre"},
    {"name": "Canola",     "ideal_soil": "loam",  "needs_irrigation": False,
     "needs_sun": "full",    "base_yield": 40,  "unit": "bu/acre"},
    {"name": "Barley",     "ideal_soil": "loam",  "needs_irrigation": False,
     "needs_sun": "partial", "base_yield": 70,  "unit": "bu/acre"},
    {"name": "Sunflower",  "ideal_soil": "sandy", "needs_irrigation": False,
     "needs_sun": "full",    "base_yield": 1800, "unit": "lb/acre"},
    {"name": "Oats",       "ideal_soil": "clay",  "needs_irrigation": False,
     "needs_sun": "partial", "base_yield": 75,  "unit": "bu/acre"},
    {"name": "Alfalfa",    "ideal_soil": "loam",  "needs_irrigation": True,
     "needs_sun": "full",    "base_yield": 4.5, "unit": "tons/acre"},
]

# Market prices per unit for revenue calculation
MARKET_PRICES = {
    "Corn": 5.80,        # $/bu
    "Wheat": 7.20,       # $/bu
    "Soybean": 13.50,    # $/bu
    "Canola": 15.00,     # $/bu (per bushel)
    "Barley": 6.50,      # $/bu
    "Sunflower": 0.28,   # $/lb
    "Oats": 4.00,        # $/bu
    "Alfalfa": 220.00,   # $/ton
}


def _compute_yield_factor(crop: dict, field: dict) -> float:
    """Compute the yield multiplier for a crop-field combination.

    Factors: soil match, drainage, sunlight, irrigation, organic matter.

    Args:
        crop: Crop dictionary with agronomic preferences.
        field: Field dictionary with characteristics.

    Returns:
        Yield multiplier (0.3 to 1.3).
    """
    factor = 1.0

    # Soil compatibility
    if field["soil"] == crop["ideal_soil"]:
        factor *= 1.15
    elif (field["soil"] == "loam"):
        factor *= 1.00  # loam is decent for most crops
    elif field["soil"] == "sandy" and crop["ideal_soil"] == "clay":
        factor *= 0.60  # poor match
    elif field["soil"] == "clay" and crop["ideal_soil"] == "sandy":
        factor *= 0.65  # poor match
    else:
        factor *= 0.85

    # Irrigation
    if crop["needs_irrigation"] and not field["irrigation"]:
        factor *= 0.55  # significant yield loss without irrigation
    elif not crop["needs_irrigation"] and field["irrigation"]:
        factor *= 1.05  # slight boost from available water

    # Sunlight
    if crop["needs_sun"] == "full" and field["sun"] == "partial":
        factor *= 0.80
    elif crop["needs_sun"] == "partial" and field["sun"] == "full":
        factor *= 1.00  # full sun is fine for shade-tolerant crops

    # Drainage
    if field["drainage"] == "poor" and crop["ideal_soil"] != "clay":
        factor *= 0.75
    elif field["drainage"] == "good" and crop["ideal_soil"] == "clay":
        factor *= 0.90  # clay crops prefer some moisture retention

    # Organic matter bonus (higher OM = better for most crops)
    om_bonus = (field["organic_matter"] - 2.5) * 0.04
    factor *= (1.0 + max(-0.1, min(0.1, om_bonus)))

    return max(0.3, min(1.3, factor))


def create_assignment_instance() -> dict:
    """Create crop-to-field assignment cost matrix.

    The cost matrix uses negative revenue (yield * price * acres) so that
    minimizing cost maximizes total farm revenue.

    Returns:
        Dictionary with assignment instance data.
    """
    n = len(CROPS)
    cost_matrix = np.zeros((n, n))
    yield_matrix = np.zeros((n, n))
    revenue_matrix = np.zeros((n, n))

    for i, crop in enumerate(CROPS):
        for j, field in enumerate(FIELDS):
            yf = _compute_yield_factor(crop, field)
            expected_yield = crop["base_yield"] * yf
            yield_matrix[i][j] = expected_yield

            price = MARKET_PRICES[crop["name"]]
            revenue = expected_yield * price * field["acres"]
            revenue_matrix[i][j] = revenue

            # Negative revenue for minimization (Hungarian minimizes)
            cost_matrix[i][j] = -revenue

    return {
        "n": n,
        "cost_matrix": cost_matrix,
        "yield_matrix": yield_matrix,
        "revenue_matrix": revenue_matrix,
        "crops": CROPS,
        "fields": FIELDS,
    }


def solve_field_assignment(verbose: bool = True) -> dict:
    """Solve crop-to-field assignment using Hungarian method.

    Args:
        verbose: Whether to print detailed results.

    Returns:
        Dictionary with assignment results.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assign_dir = os.path.join(
        base_dir, "problems", "location_network", "assignment"
    )

    ap_inst_mod = _load_mod(
        "ap_inst_agr", os.path.join(assign_dir, "instance.py")
    )
    ap_hu_mod = _load_mod(
        "ap_hu_agr", os.path.join(assign_dir, "exact", "hungarian.py")
    )
    ap_gr_mod = _load_mod(
        "ap_gr_agr",
        os.path.join(assign_dir, "heuristics", "greedy_assignment.py"),
    )

    data = create_assignment_instance()

    ap_instance = ap_inst_mod.AssignmentInstance(
        n=data["n"], cost_matrix=data["cost_matrix"], name="crop_field",
    )

    hungarian_sol = ap_hu_mod.hungarian(ap_instance)
    greedy_sol = ap_gr_mod.greedy_assignment(ap_instance)

    results = {}
    for method_name, sol in [("Hungarian", hungarian_sol), ("Greedy", greedy_sol)]:
        assignments = []
        total_revenue = 0
        for i, j in enumerate(sol.assignment):
            crop = CROPS[i]
            field = FIELDS[j]
            revenue = data["revenue_matrix"][i][j]
            expected_yield = data["yield_matrix"][i][j]
            total_revenue += revenue
            assignments.append({
                "crop": crop["name"],
                "field": field["name"],
                "acres": field["acres"],
                "expected_yield": expected_yield,
                "yield_unit": crop["unit"],
                "revenue": revenue,
            })
        results[method_name] = {
            "assignment": sol.assignment,
            "total_revenue": total_revenue,
            "details": assignments,
        }

    if verbose:
        print("=" * 70)
        print("CROP-TO-FIELD ASSIGNMENT — YIELD OPTIMIZATION")
        print(f"  {len(CROPS)} crops, {len(FIELDS)} fields, "
              f"{sum(f['acres'] for f in FIELDS)} total acres")
        print("=" * 70)

        # Show field characteristics
        print("\n  Field characteristics:")
        for f in FIELDS:
            irr = "irrigated" if f["irrigation"] else "dryland"
            print(f"    {f['name']:18s}: {f['acres']:3d} ac, {f['soil']:5s}, "
                  f"{f['drainage']:8s} drain, {f['sun']:7s} sun, "
                  f"OM={f['organic_matter']:.1f}%, {irr}")

        for method_name in ["Hungarian", "Greedy"]:
            res = results[method_name]
            print(f"\n--- {method_name} Assignment ---")
            print(f"  Total expected revenue: ${res['total_revenue']:,.0f}")
            print()
            for a in res["details"]:
                print(f"    {a['crop']:12s} -> {a['field']:18s} "
                      f"({a['acres']:3d} ac): "
                      f"{a['expected_yield']:7.1f} {a['yield_unit']:10s} "
                      f"= ${a['revenue']:>10,.0f}")

        # Compare methods
        h_rev = results["Hungarian"]["total_revenue"]
        g_rev = results["Greedy"]["total_revenue"]
        if h_rev > g_rev:
            gain = h_rev - g_rev
            print(f"\n  Hungarian optimal gains ${gain:,.0f} over Greedy "
                  f"({gain / g_rev * 100:.1f}% improvement)")
        elif g_rev > h_rev:
            print(f"\n  Both methods found equivalent assignments")

    return {"field_assignment": results}


if __name__ == "__main__":
    solve_field_assignment()
