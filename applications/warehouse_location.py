"""
Real-World Application: Warehouse Location for E-Commerce Distribution.

Domain: Regional distribution network / E-commerce fulfillment
Model: Uncapacitated Facility Location (UFLP) + p-Median

Scenario:
    An e-commerce company is expanding into a region with 30 demand zones
    (cities/towns). They need to decide:
    1. UFLP: Which warehouses to open (balancing fixed lease costs vs
       delivery distances)?
    2. p-Median: If budget constrains them to exactly p=3 warehouses,
       which locations minimize total customer-weighted distance?

    Warehouse candidates are at 8 logistics hubs with different lease
    costs (urban hubs are expensive but central; suburban are cheaper
    but farther from demand).

Real-world considerations modeled:
    - Fixed costs vary by location (urban vs suburban vs rural)
    - Demand-weighted distances (populous cities get more weight)
    - Geographic clustering of demand
    - Trade-off between number of warehouses and total delivery cost

Industry context:
    Warehouse location is a strategic decision with 10-20 year horizons.
    Opening one additional warehouse typically reduces average delivery
    time by 15-25% but adds $1-5M in annual fixed costs (Melo et al., 2009).

References:
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

# 8 candidate warehouse locations (logistics hubs)
WAREHOUSES = [
    {"name": "Metro Hub Central",  "coords": (50, 50), "annual_cost": 4_500_000, "type": "urban"},
    {"name": "Airport Logistics",  "coords": (70, 65), "annual_cost": 3_800_000, "type": "suburban"},
    {"name": "Port District",      "coords": (85, 30), "annual_cost": 3_200_000, "type": "industrial"},
    {"name": "Northgate Park",     "coords": (40, 85), "annual_cost": 2_500_000, "type": "suburban"},
    {"name": "Southfield Center",  "coords": (55, 15), "annual_cost": 2_200_000, "type": "suburban"},
    {"name": "Westlake Industrial","coords": (10, 45), "annual_cost": 1_800_000, "type": "rural"},
    {"name": "Eastview Depot",     "coords": (90, 55), "annual_cost": 2_000_000, "type": "rural"},
    {"name": "Valley Crossroads",  "coords": (30, 30), "annual_cost": 1_500_000, "type": "rural"},
]

# 30 demand zones (cities/towns)
DEMAND_ZONES = [
    # Major cities (high demand)
    {"name": "Capital City",    "coords": (48, 52), "population": 800_000},
    {"name": "Harbor Town",     "coords": (82, 28), "population": 350_000},
    {"name": "North Bay",       "coords": (45, 88), "population": 250_000},
    {"name": "East Ridge",      "coords": (88, 58), "population": 200_000},
    {"name": "South Valley",    "coords": (52, 12), "population": 180_000},
    # Medium towns
    {"name": "Riverside",       "coords": (30, 55), "population": 120_000},
    {"name": "Hilltop",         "coords": (65, 72), "population": 95_000},
    {"name": "Lakewood",        "coords": (22, 38), "population": 85_000},
    {"name": "Oakdale",         "coords": (75, 42), "population": 78_000},
    {"name": "Pinecrest",       "coords": (38, 68), "population": 70_000},
    {"name": "Cedarville",      "coords": (60, 35), "population": 65_000},
    {"name": "Maplewood",       "coords": (15, 60), "population": 55_000},
    {"name": "Elm Grove",       "coords": (55, 78), "population": 50_000},
    {"name": "Birchwood",       "coords": (72, 20), "population": 48_000},
    {"name": "Willowbrook",     "coords": (25, 75), "population": 42_000},
    # Small towns
    {"name": "Ashland",         "coords": (90, 80), "population": 30_000},
    {"name": "Brookville",      "coords": (10, 20), "population": 28_000},
    {"name": "Clayton",         "coords": (68, 55), "population": 25_000},
    {"name": "Dover",           "coords": (35, 42), "population": 22_000},
    {"name": "Fairview",        "coords": (80, 72), "population": 20_000},
    {"name": "Georgetown",      "coords": (45, 25), "population": 18_000},
    {"name": "Hampton",         "coords": (62, 88), "population": 16_000},
    {"name": "Irving",          "coords": (18, 52), "population": 15_000},
    {"name": "Jefferson",       "coords": (55, 60), "population": 14_000},
    {"name": "Kingston",        "coords": (92, 40), "population": 12_000},
    {"name": "Lancaster",       "coords": (5, 70),  "population": 11_000},
    {"name": "Morrison",        "coords": (42, 10), "population": 10_000},
    {"name": "Newport",         "coords": (78, 85), "population": 9_000},
    {"name": "Oxford",          "coords": (33, 90), "population": 8_000},
    {"name": "Plymouth",        "coords": (60, 5),  "population": 7_000},
]


def create_uflp_instance() -> dict:
    """Create a UFLP warehouse location instance.

    Returns:
        Dictionary with instance data and metadata.
    """
    m = len(WAREHOUSES)
    n = len(DEMAND_ZONES)

    wh_coords = np.array([w["coords"] for w in WAREHOUSES], dtype=float)
    cust_coords = np.array([d["coords"] for d in DEMAND_ZONES], dtype=float)

    fixed_costs = np.array(
        [w["annual_cost"] / 1000 for w in WAREHOUSES], dtype=float  # in $K
    )

    # Assignment costs: distance × demand weight (delivery cost per year)
    dist = np.sqrt(
        np.sum((wh_coords[:, None, :] - cust_coords[None, :, :]) ** 2, axis=2)
    )

    # Weight by population (annual delivery volume proxy)
    populations = np.array([d["population"] for d in DEMAND_ZONES], dtype=float)
    weights = populations / populations.mean()

    # Cost = distance × weight × cost_per_km_per_unit
    cost_per_unit_km = 2.0  # $/km/year per unit demand
    assignment_costs = dist * weights[None, :] * cost_per_unit_km

    return {
        "m": m, "n": n,
        "fixed_costs": fixed_costs,
        "assignment_costs": assignment_costs,
        "wh_coords": wh_coords,
        "cust_coords": cust_coords,
        "weights": weights,
        "dist": dist,
        "warehouses": WAREHOUSES,
        "demand_zones": DEMAND_ZONES,
    }


def solve_warehouse_location(verbose: bool = True) -> dict:
    """Solve warehouse location as both UFLP and p-Median.

    Returns:
        Dictionary with results.
    """
    data = create_uflp_instance()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    loc_dir = os.path.join(base_dir, "problems", "location_network")

    # ── UFLP ─────────────────────────────────────────────────────────────
    fl_inst_mod = _load_mod(
        "fl_inst_wh", os.path.join(loc_dir, "facility_location", "instance.py")
    )
    fl_gr_mod = _load_mod(
        "fl_gr_wh",
        os.path.join(loc_dir, "facility_location", "heuristics", "greedy_facility.py"),
    )
    fl_sa_mod = _load_mod(
        "fl_sa_wh",
        os.path.join(loc_dir, "facility_location", "metaheuristics", "simulated_annealing.py"),
    )

    fl_instance = fl_inst_mod.FacilityLocationInstance(
        m=data["m"], n=data["n"],
        fixed_costs=data["fixed_costs"],
        assignment_costs=data["assignment_costs"],
        name="warehouse_uflp",
    )

    ga_sol = fl_gr_mod.greedy_add(fl_instance)
    gd_sol = fl_gr_mod.greedy_drop(fl_instance)
    sa_sol = fl_sa_mod.simulated_annealing(
        fl_instance, max_iterations=30_000, seed=42
    )

    results = {"UFLP": {}}
    for name, sol in [("Greedy-Add", ga_sol), ("Greedy-Drop", gd_sol), ("SA", sa_sol)]:
        results["UFLP"][name] = {
            "cost": sol.cost,
            "open": sol.open_facilities,
            "assignments": sol.assignments,
        }

    # ── p-Median ─────────────────────────────────────────────────────────
    pm_inst_mod = _load_mod(
        "pm_inst_wh", os.path.join(loc_dir, "p_median", "instance.py")
    )
    pm_gr_mod = _load_mod(
        "pm_gr_wh",
        os.path.join(loc_dir, "p_median", "heuristics", "greedy_pmedian.py"),
    )

    pm_instance = pm_inst_mod.PMedianInstance(
        n=data["n"], m=data["m"], p=3,
        weights=data["weights"],
        distance_matrix=data["dist"],
        name="warehouse_pmedian",
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
        print("WAREHOUSE LOCATION FOR E-COMMERCE DISTRIBUTION")
        print(f"  {data['m']} candidate sites, {data['n']} demand zones")
        print("=" * 70)

        print("\n--- UFLP (minimize fixed + delivery cost) ---")
        for method, res in results["UFLP"].items():
            wh_names = [WAREHOUSES[i]["name"] for i in res["open"]]
            fixed = sum(data["fixed_costs"][i] for i in res["open"])
            delivery = res["cost"] - fixed
            print(f"\n  {method}: total = ${res['cost']:.0f}K/year "
                  f"(fixed=${fixed:.0f}K + delivery=${delivery:.0f}K)")
            print(f"    Open {len(res['open'])} warehouses: {', '.join(wh_names)}")

        print("\n--- p-Median (exactly 3 warehouses, minimize distance) ---")
        for method, res in results["p-Median"].items():
            wh_names = [WAREHOUSES[i]["name"] for i in res["open"]]
            print(f"\n  {method}: weighted distance = {res['cost']:.1f}")
            print(f"    Open: {', '.join(wh_names)}")

            # Coverage analysis
            zone_distances = []
            for j in range(data["n"]):
                fac = res["assignments"][j]
                d = data["dist"][fac][j]
                zone_distances.append(d)
            avg_d = np.mean(zone_distances)
            max_d = np.max(zone_distances)
            farthest = DEMAND_ZONES[np.argmax(zone_distances)]["name"]
            print(f"    Avg delivery distance: {avg_d:.1f} km")
            print(f"    Max delivery distance: {max_d:.1f} km ({farthest})")

    return results


if __name__ == "__main__":
    solve_warehouse_location()
