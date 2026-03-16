"""
Real-World Application: Agricultural Distribution Center Location.

Domain: Post-harvest — locating agricultural distribution centers
Model: Uncapacitated Facility Location (UFLP) + p-Median — choose which
       distribution centers to open to serve farming regions, minimizing
       total fixed + transport cost.

Scenario:
    A regional agricultural cooperative must establish distribution centers
    to collect, sort, and ship produce from 12 farming regions. There are
    5 potential distribution center locations, each with different facility
    costs reflecting land prices, cold storage infrastructure, and labor
    availability. Each farming region has a production volume (weight)
    that determines shipping frequency.

    Two formulations are solved:
    1. UFLP: Decide which centers to open, balancing fixed facility costs
       against transportation costs from farms to centers.
    2. p-Median (p=2): If budget allows only 2 centers, find the optimal
       pair minimizing total weighted transport distance.

Real-world considerations modeled:
    - Fixed costs vary by location (urban hub vs rural site)
    - Transport costs proportional to distance and production volume
    - Geographic spread of farming regions across the territory
    - Perishability premium — closer centers reduce spoilage losses

Industry context:
    Strategic distribution center placement reduces agricultural
    transport costs by 15-30% and post-harvest losses by 10-20%.
    In developing countries, 30-40% of fresh produce is lost between
    farm and market due to inadequate cold chain logistics. Optimal
    hub-and-spoke networks with cold storage can cut losses to under
    10% (Parfitt et al., 2010).

References:
    Parfitt, J., Barthel, M. & Macnaughton, S. (2010). Food waste
    within food supply chains: Quantification and potential for change
    to 2050. Philosophical Transactions of the Royal Society B,
    365(1554), 3065-3081.
    https://doi.org/10.1098/rstb.2010.0126

    Melo, M.T., Nickel, S. & Saldanha-da-Gama, F. (2009). Facility
    location and supply chain management — A review. European Journal
    of Operational Research, 196(2), 401-412.
    https://doi.org/10.1016/j.ejor.2008.05.007

    Daskin, M.S. (2013). Network and Discrete Location: Models,
    Algorithms, and Applications. 2nd ed. Wiley.
    https://doi.org/10.1002/9781118537015
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

# 5 potential distribution center locations
CENTERS = [
    {"name": "Valley Hub (Central)",      "coords": (50, 45),
     "annual_cost": 320_000, "type": "urban",
     "features": "Full cold storage, rail access"},
    {"name": "Riverside Depot (South)",    "coords": (35, 15),
     "annual_cost": 180_000, "type": "rural",
     "features": "Basic cold storage, road only"},
    {"name": "Highland Station (North)",   "coords": (45, 80),
     "annual_cost": 220_000, "type": "suburban",
     "features": "Cold storage, highway access"},
    {"name": "Eastport Terminal (East)",   "coords": (85, 50),
     "annual_cost": 280_000, "type": "suburban",
     "features": "Full cold storage, port access"},
    {"name": "Prairie Crossing (West)",    "coords": (10, 55),
     "annual_cost": 150_000, "type": "rural",
     "features": "Ambient storage, road only"},
]

# 12 farming regions with production volumes
FARMING_REGIONS = [
    {"name": "Grain Valley Farms",     "coords": (40, 40), "production_tons": 5000,
     "primary_crop": "wheat"},
    {"name": "Sunflower Plains",       "coords": (25, 60), "production_tons": 3200,
     "primary_crop": "sunflower"},
    {"name": "Riverside Orchards",     "coords": (30, 20), "production_tons": 2800,
     "primary_crop": "apples"},
    {"name": "Northern Dairy Farms",   "coords": (50, 85), "production_tons": 4500,
     "primary_crop": "dairy"},
    {"name": "Eastside Vineyards",     "coords": (80, 55), "production_tons": 1500,
     "primary_crop": "grapes"},
    {"name": "Corn Belt South",        "coords": (55, 20), "production_tons": 6000,
     "primary_crop": "corn"},
    {"name": "Hilltop Berry Farms",    "coords": (65, 70), "production_tons": 900,
     "primary_crop": "berries"},
    {"name": "Western Rangelands",     "coords": (5, 45),  "production_tons": 3800,
     "primary_crop": "cattle"},
    {"name": "Delta Rice Paddies",     "coords": (70, 30), "production_tons": 4200,
     "primary_crop": "rice"},
    {"name": "Mountain Herb Gardens",  "coords": (60, 90), "production_tons": 600,
     "primary_crop": "herbs"},
    {"name": "Lakeshore Vegetables",   "coords": (20, 75), "production_tons": 2100,
     "primary_crop": "vegetables"},
    {"name": "Southern Cotton Fields", "coords": (45, 5),  "production_tons": 3500,
     "primary_crop": "cotton"},
]

# Transport cost per ton-km ($/ton/km)
TRANSPORT_COST_PER_TON_KM = 0.12


def create_distribution_instance() -> dict:
    """Create agricultural distribution center location instances.

    Returns:
        Dictionary with UFLP and p-Median instance data.
    """
    m = len(CENTERS)
    n = len(FARMING_REGIONS)

    center_coords = np.array([c["coords"] for c in CENTERS], dtype=float)
    farm_coords = np.array([f["coords"] for f in FARMING_REGIONS], dtype=float)

    # Fixed costs (in $K)
    fixed_costs = np.array(
        [c["annual_cost"] / 1000.0 for c in CENTERS], dtype=float
    )

    # Euclidean distance matrix (m, n)
    dist = np.sqrt(
        np.sum((center_coords[:, None, :] - farm_coords[None, :, :]) ** 2, axis=2)
    )

    # Production weights (normalized)
    production = np.array(
        [f["production_tons"] for f in FARMING_REGIONS], dtype=float
    )
    weights = production / production.mean()

    # Assignment costs: distance * weight * transport cost factor
    assignment_costs = dist * weights[None, :] * TRANSPORT_COST_PER_TON_KM

    return {
        "m": m,
        "n": n,
        "fixed_costs": fixed_costs,
        "assignment_costs": assignment_costs,
        "dist": dist,
        "weights": weights,
        "production": production,
        "center_coords": center_coords,
        "farm_coords": farm_coords,
    }


def solve_distribution_centers(verbose: bool = True) -> dict:
    """Solve the distribution center location problem.

    Solves both UFLP (optimal number of centers) and p-Median (exactly 2).

    Returns:
        Dictionary with results from both formulations.
    """
    data = create_distribution_instance()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    loc_dir = os.path.join(base_dir, "problems", "location_network")

    # Load facility location modules
    fl_inst_mod = _load_mod(
        "fl_inst_agri", os.path.join(loc_dir, "facility_location", "instance.py")
    )
    fl_gr_mod = _load_mod(
        "fl_gr_agri",
        os.path.join(loc_dir, "facility_location", "heuristics", "greedy_facility.py"),
    )

    # Load p-median modules
    pm_inst_mod = _load_mod(
        "pm_inst_agri", os.path.join(loc_dir, "p_median", "instance.py")
    )
    pm_gr_mod = _load_mod(
        "pm_gr_agri",
        os.path.join(loc_dir, "p_median", "heuristics", "greedy_pmedian.py"),
    )

    results = {}

    # ── UFLP ─────────────────────────────────────────────────────────────
    fl_instance = fl_inst_mod.FacilityLocationInstance(
        m=data["m"],
        n=data["n"],
        fixed_costs=data["fixed_costs"],
        assignment_costs=data["assignment_costs"],
        name="agri_distribution_uflp",
    )

    ga_sol = fl_gr_mod.greedy_add(fl_instance)
    gd_sol = fl_gr_mod.greedy_drop(fl_instance)

    results["UFLP"] = {}
    for name, sol in [("Greedy-Add", ga_sol), ("Greedy-Drop", gd_sol)]:
        results["UFLP"][name] = {
            "cost": sol.cost,
            "open": sol.open_facilities,
            "assignments": sol.assignments,
        }

    # ── p-Median (p=2) ───────────────────────────────────────────────────
    pm_instance = pm_inst_mod.PMedianInstance(
        n=data["n"],
        m=data["m"],
        p=2,
        weights=data["weights"],
        distance_matrix=data["dist"],
        name="agri_distribution_pmedian",
    )

    pm_gr = pm_gr_mod.greedy_pmedian(pm_instance)
    pm_tb = pm_gr_mod.interchange(pm_instance)

    results["p-Median"] = {}
    for name, sol in [("Greedy", pm_gr), ("Interchange", pm_tb)]:
        results["p-Median"][name] = {
            "cost": sol.cost,
            "open": sol.open_facilities,
            "assignments": sol.assignments,
        }

    if verbose:
        print("=" * 70)
        print("AGRICULTURAL DISTRIBUTION CENTER LOCATION")
        print(f"  {data['m']} candidate centers, {data['n']} farming regions")
        print(f"  Total annual production: "
              f"{data['production'].sum():,.0f} tons")
        print("=" * 70)

        print("\n  Candidate centers:")
        for i, center in enumerate(CENTERS):
            print(f"    {i}. {center['name']}")
            print(f"       Cost: ${center['annual_cost']:,}/yr, "
                  f"Type: {center['type']}, {center['features']}")

        print("\n  Farming regions:")
        for j, farm in enumerate(FARMING_REGIONS):
            print(f"    {j:2d}. {farm['name']:30s} "
                  f"{farm['production_tons']:5,} tons ({farm['primary_crop']})")

        # UFLP results
        print("\n--- UFLP (minimize fixed + transport cost) ---")
        for method, res in results["UFLP"].items():
            center_names = [CENTERS[i]["name"] for i in res["open"]]
            fixed = sum(data["fixed_costs"][i] for i in res["open"])
            transport = res["cost"] - fixed
            print(f"\n  {method}: total = ${res['cost']:,.1f}K/yr "
                  f"(fixed=${fixed:,.0f}K + transport=${transport:,.1f}K)")
            print(f"    Open {len(res['open'])} centers: "
                  f"{', '.join(center_names)}")

            # Show assignments
            for j in range(data["n"]):
                fac = res["assignments"][j]
                d = data["dist"][fac][j]
                print(f"      {FARMING_REGIONS[j]['name']:30s} -> "
                      f"{CENTERS[fac]['name']:30s} ({d:.1f} km)")

        # p-Median results
        print(f"\n--- p-Median (exactly 2 centers, minimize distance) ---")
        for method, res in results["p-Median"].items():
            center_names = [CENTERS[i]["name"] for i in res["open"]]
            print(f"\n  {method}: weighted distance = {res['cost']:,.1f}")
            print(f"    Open: {', '.join(center_names)}")

            # Coverage analysis
            distances = []
            for j in range(data["n"]):
                fac = res["assignments"][j]
                d = data["dist"][fac][j]
                distances.append(d)
            avg_d = np.mean(distances)
            max_d = np.max(distances)
            farthest = FARMING_REGIONS[np.argmax(distances)]["name"]
            print(f"    Avg transport distance: {avg_d:.1f} km")
            print(f"    Max transport distance: {max_d:.1f} km ({farthest})")

    return results


if __name__ == "__main__":
    solve_distribution_centers()
