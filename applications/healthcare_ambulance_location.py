"""
Real-World Application: Emergency Ambulance Base Location.

Domain: Emergency Medical Services (EMS) network planning
Model: p-Median (coverage) + Facility Location (cost-aware)

Scenario:
    A county EMS authority serves 20 population zones and must locate
    p=4 ambulance bases from 10 candidate sites. Two perspectives:

    1. p-Median: Minimize population-weighted average response time
       (proxy: distance). Ensures every zone is as close as possible
       to its nearest ambulance base.

    2. Facility Location: Consider that some sites have higher
       construction/lease costs. Find the cost-optimal set of bases
       that balances fixed costs vs coverage quality.

    Response time is critical: survival rates for cardiac arrest drop
    7-10% per minute without CPR. The NFPA 1710 standard targets
    4-minute EMS response in urban areas, 8 minutes in suburban.

Real-world considerations modeled:
    - Population density weighting (more ambulances near dense areas)
    - Road-network proxy distances (Euclidean × 1.4 factor)
    - Site cost variation (urban vs suburban infrastructure costs)
    - Coverage analysis (% population within 8-minute response)

Industry context:
    EMS location is a life-or-death optimization. Studies show that
    optimal relocation of ambulance bases can reduce average response
    times by 20-40% with the same number of units (Li et al., 2011).
    Every minute of reduced response time for cardiac arrest increases
    survival by 5-10%.

References:
    Li, X., Zhao, Z., Zhu, X. & Wyatt, T. (2011). Covering models
    and optimization techniques for emergency response facility
    location and planning: A review. Mathematical Methods of
    Operations Research, 74(3), 281-310.
    https://doi.org/10.1007/s00186-011-0363-4

    Brotcorne, L., Laporte, G. & Semet, F. (2003). Ambulance location
    and relocation models. European Journal of Operational Research,
    147(3), 451-463.
    https://doi.org/10.1016/S0377-2217(02)00364-8

    Aringhieri, R., Bruni, M.E., Khodaparasti, S. & van Essen, J.T.
    (2017). Emergency medical services location problems: A review.
    Annals of Operations Research, 253(1), 129-154.
    https://doi.org/10.1007/s10479-016-2268-8
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

# 10 candidate ambulance base sites
CANDIDATE_SITES = [
    {"name": "Central Fire Station",     "coords": (50, 50), "annual_cost": 850_000, "type": "urban"},
    {"name": "North Hospital Campus",    "coords": (48, 82), "annual_cost": 720_000, "type": "urban"},
    {"name": "East Industrial Park",     "coords": (85, 55), "annual_cost": 450_000, "type": "suburban"},
    {"name": "South Community Center",   "coords": (52, 18), "annual_cost": 380_000, "type": "suburban"},
    {"name": "West Shopping District",   "coords": (15, 48), "annual_cost": 520_000, "type": "suburban"},
    {"name": "Airport Zone",             "coords": (75, 78), "annual_cost": 400_000, "type": "suburban"},
    {"name": "University Campus",        "coords": (35, 65), "annual_cost": 600_000, "type": "urban"},
    {"name": "Rural Highway Junction",   "coords": (90, 15), "annual_cost": 280_000, "type": "rural"},
    {"name": "Lakeside Residential",     "coords": (20, 75), "annual_cost": 350_000, "type": "suburban"},
    {"name": "Downtown Transit Hub",     "coords": (55, 55), "annual_cost": 780_000, "type": "urban"},
]

# 20 population demand zones
POPULATION_ZONES = [
    {"name": "Downtown Core",         "coords": (50, 52), "population": 45_000},
    {"name": "University District",   "coords": (35, 63), "population": 32_000},
    {"name": "North Residential",     "coords": (47, 80), "population": 28_000},
    {"name": "Hospital Area",         "coords": (50, 78), "population": 18_000},
    {"name": "East Business Park",    "coords": (82, 53), "population": 15_000},
    {"name": "South Suburbs",         "coords": (53, 20), "population": 22_000},
    {"name": "West Village",          "coords": (18, 50), "population": 20_000},
    {"name": "Airport District",      "coords": (73, 75), "population": 12_000},
    {"name": "Waterfront",            "coords": (60, 45), "population": 16_000},
    {"name": "Tech Campus",           "coords": (70, 60), "population": 25_000},
    {"name": "Old Town",              "coords": (42, 45), "population": 19_000},
    {"name": "Lakeside",              "coords": (22, 73), "population": 14_000},
    {"name": "Industrial Zone",       "coords": (88, 40), "population": 8_000},
    {"name": "South Gateway",         "coords": (48, 12), "population": 11_000},
    {"name": "Hilltop Estates",       "coords": (30, 85), "population": 9_000},
    {"name": "Highway Corridor",      "coords": (85, 18), "population": 7_000},
    {"name": "River District",        "coords": (40, 35), "population": 13_000},
    {"name": "Northern Heights",      "coords": (55, 90), "population": 10_000},
    {"name": "Meadowlands",           "coords": (25, 30), "population": 6_000},
    {"name": "Eastside Apartments",   "coords": (78, 65), "population": 17_000},
]

# Average ambulance speed: 50 km/h → response time (min) = distance * 1.4 / 50 * 60
SPEED_KMH = 50
ROAD_FACTOR = 1.4  # Manhattan distance correction


def create_ambulance_data() -> dict:
    """Create ambulance location problem data.

    Returns:
        Dictionary with p-median and facility location instance data.
    """
    m = len(CANDIDATE_SITES)
    n = len(POPULATION_ZONES)

    site_coords = np.array([s["coords"] for s in CANDIDATE_SITES], dtype=float)
    zone_coords = np.array([z["coords"] for z in POPULATION_ZONES], dtype=float)

    # Euclidean distance × road factor → km
    euclidean = np.sqrt(
        np.sum((site_coords[:, None, :] - zone_coords[None, :, :]) ** 2, axis=2)
    )
    road_dist = euclidean * ROAD_FACTOR  # km

    # Response time in minutes
    response_time = road_dist / SPEED_KMH * 60

    # Population weights (normalized)
    populations = np.array([z["population"] for z in POPULATION_ZONES], dtype=float)
    weights = populations / populations.mean()

    # Facility location costs (annual cost in $K)
    fixed_costs = np.array([s["annual_cost"] / 1000 for s in CANDIDATE_SITES], dtype=float)

    # Assignment cost for UFLP: response_time × population_weight × cost_factor
    cost_factor = 50  # $/min/year weighted
    assignment_costs = response_time * weights[None, :] * cost_factor

    return {
        "m": m, "n": n, "p": 4,
        "site_coords": site_coords,
        "zone_coords": zone_coords,
        "road_dist": road_dist,
        "response_time": response_time,
        "populations": populations,
        "weights": weights,
        "fixed_costs": fixed_costs,
        "assignment_costs": assignment_costs,
        "sites": CANDIDATE_SITES,
        "zones": POPULATION_ZONES,
    }


def solve_ambulance_location(verbose: bool = True) -> dict:
    """Solve ambulance base location using p-Median and UFLP.

    Returns:
        Dictionary with results.
    """
    data = create_ambulance_data()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    loc_dir = os.path.join(base_dir, "problems", "location_network")

    results = {}

    # ── p-Median: Minimize response time ─────────────────────────────────
    pm_inst_mod = _load_mod("pm_inst_amb", os.path.join(loc_dir, "p_median", "instance.py"))
    pm_gr_mod = _load_mod(
        "pm_gr_amb",
        os.path.join(loc_dir, "p_median", "heuristics", "greedy_pmedian.py"),
    )

    pm_instance = pm_inst_mod.PMedianInstance(
        n=data["n"], m=data["m"], p=data["p"],
        weights=data["weights"],
        distance_matrix=data["response_time"],  # response time as distance
        name="ambulance_pmedian",
    )

    pm_gr = pm_gr_mod.greedy_pmedian(pm_instance)
    pm_tb = pm_gr_mod.interchange(pm_instance)

    results["p_median"] = {
        "Greedy": {"open": pm_gr.open_facilities, "cost": pm_gr.cost, "assignments": pm_gr.assignments},
        "Interchange": {"open": pm_tb.open_facilities, "cost": pm_tb.cost, "assignments": pm_tb.assignments},
    }

    # ── UFLP: Cost-aware location ───────────────────────────────────────
    fl_inst_mod = _load_mod("fl_inst_amb", os.path.join(loc_dir, "facility_location", "instance.py"))
    fl_gr_mod = _load_mod(
        "fl_gr_amb",
        os.path.join(loc_dir, "facility_location", "heuristics", "greedy_facility.py"),
    )
    fl_sa_mod = _load_mod(
        "fl_sa_amb",
        os.path.join(loc_dir, "facility_location", "metaheuristics", "simulated_annealing.py"),
    )

    fl_instance = fl_inst_mod.FacilityLocationInstance(
        m=data["m"], n=data["n"],
        fixed_costs=data["fixed_costs"],
        assignment_costs=data["assignment_costs"],
        name="ambulance_uflp",
    )

    fl_ga = fl_gr_mod.greedy_add(fl_instance)
    fl_sa = fl_sa_mod.simulated_annealing(fl_instance, max_iterations=30_000, seed=42)

    results["uflp"] = {
        "Greedy-Add": {"open": fl_ga.open_facilities, "cost": fl_ga.cost, "assignments": fl_ga.assignments},
        "SA": {"open": fl_sa.open_facilities, "cost": fl_sa.cost, "assignments": fl_sa.assignments},
    }

    if verbose:
        print("=" * 70)
        print("EMERGENCY AMBULANCE BASE LOCATION")
        print(f"  {data['m']} candidate sites, {data['n']} demand zones")
        print(f"  Total served population: {data['populations'].sum():,.0f}")
        print("=" * 70)

        # p-Median results
        print("\n--- p-Median: Locate 4 bases to minimize response time ---")
        best_pm = min(results["p_median"], key=lambda k: results["p_median"][k]["cost"])
        res = results["p_median"][best_pm]
        site_names = [CANDIDATE_SITES[i]["name"] for i in res["open"]]
        print(f"  Best ({best_pm}): {', '.join(site_names)}")

        # Response time analysis
        response_times = []
        pop_within_8 = 0
        for j in range(data["n"]):
            fac = res["assignments"][j]
            rt = data["response_time"][fac][j]
            response_times.append(rt)
            if rt <= 8:
                pop_within_8 += data["populations"][j]

        avg_rt = np.average(response_times, weights=data["populations"])
        max_rt = max(response_times)
        worst_zone = POPULATION_ZONES[np.argmax(response_times)]["name"]
        coverage_pct = pop_within_8 / data["populations"].sum() * 100

        print(f"  Pop-weighted avg response: {avg_rt:.1f} min")
        print(f"  Max response time: {max_rt:.1f} min ({worst_zone})")
        print(f"  Coverage (<=8 min): {coverage_pct:.1f}% of population")
        print(f"  Annual base cost: ${sum(CANDIDATE_SITES[i]['annual_cost'] for i in res['open']):,.0f}")

        # Zone-by-zone breakdown
        print("\n  Zone coverage:")
        for j in range(data["n"]):
            fac = res["assignments"][j]
            rt = data["response_time"][fac][j]
            zone = POPULATION_ZONES[j]
            site = CANDIDATE_SITES[fac]["name"]
            status = "OK" if rt <= 8 else "SLOW"
            print(f"    {zone['name']:22s} (pop {zone['population']:6,}) "
                  f"→ {site:25s} {rt:5.1f} min [{status}]")

        # UFLP results
        print("\n--- Facility Location: Cost-aware base selection ---")
        best_fl = min(results["uflp"], key=lambda k: results["uflp"][k]["cost"])
        fl_res = results["uflp"][best_fl]
        fl_names = [CANDIDATE_SITES[i]["name"] for i in fl_res["open"]]
        fixed = sum(data["fixed_costs"][i] for i in fl_res["open"])
        delivery = fl_res["cost"] - fixed
        print(f"  Best ({best_fl}): {', '.join(fl_names)}")
        print(f"  Total cost: ${fl_res['cost']:.0f}K (fixed ${fixed:.0f}K + coverage ${delivery:.0f}K)")

    return results


if __name__ == "__main__":
    solve_ambulance_location()
