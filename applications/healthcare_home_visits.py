"""
Real-World Application: Home Healthcare Nurse Routing.

Domain: Community nursing / Home health services
Model: TSP (single nurse) + VRPTW (team with time windows)

Scenario:
    A home healthcare agency dispatches nurses to visit patients at home.
    Each patient requires a specific visit duration and has a preferred
    time window (e.g., morning medication, afternoon wound care).

    Problem 1 (TSP): A single community nurse must visit 15 homebound
    patients in a metropolitan area. Find the shortest route to minimize
    total driving time and maximize face-to-face patient care hours.

    Problem 2 (VRPTW): A team of nurses must visit 15 patients, each
    with a 2-hour appointment window. Each nurse has a maximum shift
    length of 8 hours and can handle up to 200 "care-minutes" per shift.
    Find routes that respect all time windows and balance workload.

Real-world considerations modeled:
    - Patient acuity-based visit durations (15-60 min)
    - Geographic clustering (patients in same neighborhood)
    - Time window constraints (medication schedules, therapy windows)
    - Nurse shift length and workload capacity
    - Travel time between patient homes

Industry context:
    Home healthcare is the fastest-growing segment of healthcare delivery,
    with over 5 million Americans receiving home health services annually
    (CMS, 2022). Route optimization can reduce nurse driving time by
    20-30%, increasing direct patient care hours by 15-25% (Fikar &
    Hirsch, 2017). Poor routing leads to nurse burnout, missed visits,
    and medication non-adherence.

References:
    Fikar, C. & Hirsch, P. (2017). Home health care routing and
    scheduling: A review. Computers & Operations Research, 77, 86-95.
    https://doi.org/10.1016/j.cor.2016.07.019

    Begur, S.V., Miller, D.M. & Weaver, J.R. (1997). An integrated
    spatial DSS for scheduling and routing home-health-care nurses.
    Interfaces, 27(4), 35-48.
    https://doi.org/10.1287/inte.27.4.35

    Benzarti, E., Sahin, E. & Dallery, Y. (2013). Operations management
    applied to home care services: Analysis of the districting problem.
    Decision Support Systems, 55(2), 587-598.
    https://doi.org/10.1016/j.dss.2012.10.015
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

# Agency office (depot)
AGENCY = {"name": "Home Health Agency Office", "coords": (50, 50)}

# 15 homebound patients
PATIENTS = [
    {"name": "Mrs. Thompson (diabetes)",   "coords": (35, 72), "visit_min": 45, "acuity": 4,
     "window": (480, 600),  "type": "insulin_management"},
    {"name": "Mr. Davis (wound care)",     "coords": (28, 65), "visit_min": 60, "acuity": 5,
     "window": (480, 660),  "type": "wound_care"},
    {"name": "Mrs. Chen (post-hip)",       "coords": (42, 78), "visit_min": 40, "acuity": 3,
     "window": (540, 720),  "type": "physical_therapy"},
    {"name": "Mr. Williams (COPD)",        "coords": (55, 82), "visit_min": 30, "acuity": 3,
     "window": (480, 720),  "type": "respiratory"},
    {"name": "Mrs. Garcia (CHF)",          "coords": (62, 70), "visit_min": 45, "acuity": 4,
     "window": (540, 660),  "type": "cardiac_monitoring"},
    {"name": "Mr. Brown (IV antibiotics)", "coords": (70, 65), "visit_min": 50, "acuity": 5,
     "window": (480, 600),  "type": "infusion"},
    {"name": "Mrs. Lee (palliative)",      "coords": (75, 55), "visit_min": 60, "acuity": 5,
     "window": (600, 780),  "type": "palliative"},
    {"name": "Mr. Miller (post-stroke)",   "coords": (65, 40), "visit_min": 45, "acuity": 4,
     "window": (540, 720),  "type": "rehabilitation"},
    {"name": "Mrs. Wilson (ostomy)",       "coords": (48, 35), "visit_min": 35, "acuity": 3,
     "window": (600, 780),  "type": "wound_care"},
    {"name": "Mr. Moore (dialysis prep)",  "coords": (38, 28), "visit_min": 40, "acuity": 4,
     "window": (480, 600),  "type": "renal"},
    {"name": "Mrs. Taylor (medication)",   "coords": (25, 40), "visit_min": 20, "acuity": 2,
     "window": (480, 780),  "type": "medication_management"},
    {"name": "Mr. Anderson (vitals)",      "coords": (20, 55), "visit_min": 15, "acuity": 1,
     "window": (480, 780),  "type": "vital_signs"},
    {"name": "Mrs. Jackson (catheter)",    "coords": (30, 50), "visit_min": 30, "acuity": 3,
     "window": (540, 720),  "type": "catheter_care"},
    {"name": "Mr. White (mental health)",  "coords": (58, 25), "visit_min": 45, "acuity": 3,
     "window": (600, 780),  "type": "psychiatric"},
    {"name": "Mrs. Harris (newborn)",      "coords": (45, 60), "visit_min": 40, "acuity": 3,
     "window": (540, 660),  "type": "maternal_child"},
]


def create_home_visit_instance() -> dict:
    """Create a home healthcare routing instance.

    Returns:
        Dictionary with TSP and VRPTW instance data.
    """
    n = len(PATIENTS)
    depot_coords = np.array(AGENCY["coords"], dtype=float)
    patient_coords = np.array([p["coords"] for p in PATIENTS], dtype=float)
    all_coords = np.vstack([depot_coords.reshape(1, 2), patient_coords])

    # Distance matrix (km, with road winding factor)
    n_total = n + 1
    dist = np.zeros((n_total, n_total))
    for i in range(n_total):
        for j in range(n_total):
            euclidean = np.sqrt(np.sum((all_coords[i] - all_coords[j]) ** 2))
            dist[i][j] = euclidean * 1.3  # road factor

    # Time windows (in minutes from midnight)
    time_windows = np.zeros((n_total, 2))
    time_windows[0] = [420, 960]  # agency open 7am-4pm
    for i, p in enumerate(PATIENTS):
        time_windows[i + 1] = p["window"]

    # Service times (visit duration in minutes, used as travel-time units)
    service_times = np.zeros(n_total)
    for i, p in enumerate(PATIENTS):
        service_times[i + 1] = p["visit_min"]

    # Demands (care-minutes for capacity constraint)
    demands = np.zeros(n_total)
    for i, p in enumerate(PATIENTS):
        demands[i + 1] = p["visit_min"]

    return {
        "n_patients": n,
        "coords": all_coords,
        "distance_matrix": dist,
        "time_windows": time_windows,
        "service_times": service_times,
        "demands": demands[1:],  # customer demands only
        "capacity": 200,  # max care-minutes per nurse shift
        "patients": PATIENTS,
        "agency": AGENCY,
    }


def solve_home_visits(verbose: bool = True) -> dict:
    """Solve home healthcare routing using TSP and VRPTW.

    Returns:
        Dictionary with routing results.
    """
    data = create_home_visit_instance()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    results = {}

    # ── Problem 1: TSP (single nurse, best route) ─────────────────────────
    tsp_dir = os.path.join(base_dir, "problems", "routing", "tsp")
    tsp_inst = _load_mod("tsp_inst_hh", os.path.join(tsp_dir, "instance.py"))
    tsp_nn = _load_mod("tsp_nn_hh", os.path.join(tsp_dir, "heuristics", "nearest_neighbor.py"))
    tsp_ls = _load_mod("tsp_ls_hh", os.path.join(tsp_dir, "metaheuristics", "local_search.py"))
    tsp_sa = _load_mod("tsp_sa_hh", os.path.join(tsp_dir, "metaheuristics", "simulated_annealing.py"))

    tsp_instance = tsp_inst.TSPInstance(
        n=data["n_patients"] + 1,  # depot + patients
        distance_matrix=data["distance_matrix"],
        coords=data["coords"],
        name="home_visits_tsp",
    )

    nn_sol = tsp_nn.nearest_neighbor(tsp_instance, start=0)
    nn_ms = tsp_nn.nearest_neighbor_multistart(tsp_instance)
    opt2_sol = tsp_ls.two_opt(tsp_instance, initial_tour=nn_ms.tour)
    sa_sol = tsp_sa.simulated_annealing(tsp_instance, max_iterations=50_000, seed=42)

    results["tsp"] = {}
    for name, sol in [("NN", nn_sol), ("NN-Multistart", nn_ms),
                      ("2-opt", opt2_sol), ("SA", sa_sol)]:
        results["tsp"][name] = {
            "distance": sol.distance,
            "tour": sol.tour,
        }

    # ── Problem 2: VRPTW (team with time windows) ─────────────────────────
    vrptw_dir = os.path.join(base_dir, "problems", "routing", "vrptw")
    vrptw_inst = _load_mod("vrptw_inst_hh", os.path.join(vrptw_dir, "instance.py"))
    vrptw_sol_mod = _load_mod("vrptw_si_hh", os.path.join(vrptw_dir, "heuristics", "solomon_insertion.py"))

    vrptw_instance = vrptw_inst.VRPTWInstance(
        n=data["n_patients"],
        capacity=data["capacity"],
        demands=data["demands"],
        distance_matrix=data["distance_matrix"],
        time_windows=data["time_windows"],
        service_times=data["service_times"],
        coords=data["coords"],
        name="home_visits_vrptw",
    )

    si_sol = vrptw_sol_mod.solomon_insertion(vrptw_instance)
    nn_tw_sol = vrptw_sol_mod.nearest_neighbor_tw(vrptw_instance)

    results["vrptw"] = {}
    for name, sol in [("Solomon-I1", si_sol), ("NN-TW", nn_tw_sol)]:
        results["vrptw"][name] = {
            "distance": sol.distance,
            "routes": sol.routes,
            "n_nurses": len([r for r in sol.routes if r]),
        }

    if verbose:
        total_care = sum(p["visit_min"] for p in PATIENTS)
        print("=" * 70)
        print("HOME HEALTHCARE NURSE ROUTING")
        print(f"  {data['n_patients']} patients, {total_care} total care-minutes")
        print("=" * 70)

        # Patient summary
        print("\n  Patient roster:")
        for i, p in enumerate(PATIENTS):
            h1, m1 = divmod(p["window"][0], 60)
            h2, m2 = divmod(p["window"][1], 60)
            print(f"    {i+1:2d}. {p['name']:35s} {p['visit_min']:2d}min "
                  f"(acuity {p['acuity']}) [{h1:02d}:{m1:02d}-{h2:02d}:{m2:02d}]")

        # TSP results
        best_tsp = min(results["tsp"], key=lambda k: results["tsp"][k]["distance"])
        print(f"\n--- Single Nurse Route (TSP) ---")
        for method, res in results["tsp"].items():
            marker = " (best)" if method == best_tsp else ""
            drive_time = res["distance"] / 40 * 60  # 40 km/h → minutes
            care_pct = total_care / (total_care + drive_time) * 100
            print(f"  {method}{marker}: {res['distance']:.1f} km, "
                  f"~{drive_time:.0f} min driving, {care_pct:.0f}% care time")

        # VRPTW results
        print(f"\n--- Team Routing with Time Windows (VRPTW) ---")
        for method, res in results["vrptw"].items():
            print(f"\n  {method}: {res['distance']:.1f} km total, "
                  f"{res['n_nurses']} nurses needed")
            for i, route in enumerate(res["routes"]):
                if not route:
                    continue
                care = sum(PATIENTS[c - 1]["visit_min"] for c in route)
                route_dist = vrptw_instance.route_distance(route)
                names = [PATIENTS[c - 1]["name"][:20] for c in route]
                print(f"    Nurse {i+1}: {len(route)} visits, "
                      f"{care} care-min, {route_dist:.1f} km")
                for c in route:
                    p = PATIENTS[c - 1]
                    h1, m1 = divmod(p["window"][0], 60)
                    h2, m2 = divmod(p["window"][1], 60)
                    print(f"      → {p['name']:35s} ({p['visit_min']}min, "
                          f"{h1:02d}:{m1:02d}-{h2:02d}:{m2:02d})")

    return results


if __name__ == "__main__":
    solve_home_visits()
