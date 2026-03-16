"""
Real-World Application: Fertilizer Delivery Route Optimization.

Domain: Precision agriculture / Fertilizer application logistics
Model: VRPTW (Vehicle Routing Problem with Time Windows)

Scenario:
    A fertilizer service company operates 3 spreader trucks from a central
    depot (fertilizer warehouse). During spring growing season, 10 farm
    fields need fertilizer application. Each field has a time window when
    application is most effective — some crops (e.g., corn) need early
    morning application before heat causes volatilization, while others
    (e.g., alfalfa, soybeans) can be serviced any time during the day.

    Each truck carries up to 20 tons of fertilizer. Fields require varying
    amounts (2-8 tons) depending on acreage and soil nutrient levels. The
    goal is to minimize total travel distance while respecting all time
    windows and truck capacities.

Real-world considerations modeled:
    - Crop-specific application time windows (nitrogen volatilization risk)
    - Truck fertilizer capacity (weight-limited spreader trucks)
    - Service time at each field (loading, calibration, application)
    - Geographic spread of fields across a farming region
    - Depot operating hours (planning horizon: 6 AM to 6 PM)
    - Soil test-based fertilizer quantities per field

Industry context:
    Precision timing of fertilizer application increases nitrogen uptake
    efficiency by 20-30% (Scharf et al., 2002). In the US alone, farmers
    spend over $25 billion annually on fertilizer (USDA ERS, 2023).
    Route optimization for custom applicators can reduce fuel costs by
    15-25% and ensure time-sensitive applications are completed within
    optimal agronomic windows, reducing nutrient runoff and improving
    crop yields.

References:
    Scharf, P.C., Kitchen, N.R., Sudduth, K.A., Davis, J.G., Hubbard,
    V.C. & Lory, J.A. (2005). Field-scale variability in optimal
    nitrogen fertilizer rate for corn. Agronomy Journal, 97(2), 452-461.
    https://doi.org/10.2134/agronj2005.0452

    Bochtis, D.D. & Sorensen, C.G. (2009). The vehicle routing problem
    in field logistics part I. Biosystems Engineering, 104(4), 447-457.
    https://doi.org/10.1016/j.biosystemseng.2009.09.003

    Seyyedhasani, H. & Dvorak, J.S. (2017). Using the Vehicle Routing
    Problem to reduce field completion times with multiple machines.
    Computers and Electronics in Agriculture, 134, 142-150.
    https://doi.org/10.1016/j.compag.2016.11.010
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


# -- Domain Data ---------------------------------------------------------------

# Fertilizer warehouse (depot)
DEPOT = {"name": "AgriChem Fertilizer Depot", "coords": (50, 50)}

# 10 farm fields requiring fertilizer application
FIELDS = [
    {"name": "Henderson Corn North",   "coords": (30, 85), "tons": 7, "crop": "corn",
     "window": (360, 480),  "service_min": 45, "notes": "Early AM — reduce N volatilization"},
    {"name": "Henderson Corn South",   "coords": (35, 75), "tons": 6, "crop": "corn",
     "window": (360, 480),  "service_min": 40, "notes": "Early AM — reduce N volatilization"},
    {"name": "Williams Wheat Field",   "coords": (20, 60), "tons": 5, "crop": "wheat",
     "window": (360, 600),  "service_min": 35, "notes": "Morning preferred"},
    {"name": "Garcia Soybean East",    "coords": (75, 70), "tons": 3, "crop": "soybean",
     "window": (360, 720),  "service_min": 25, "notes": "Flexible window"},
    {"name": "Garcia Soybean West",    "coords": (70, 65), "tons": 4, "crop": "soybean",
     "window": (360, 720),  "service_min": 30, "notes": "Flexible window"},
    {"name": "Thompson Alfalfa",       "coords": (60, 30), "tons": 3, "crop": "alfalfa",
     "window": (420, 720),  "service_min": 25, "notes": "After morning dew dries"},
    {"name": "Miller Corn Field",      "coords": (25, 40), "tons": 8, "crop": "corn",
     "window": (360, 480),  "service_min": 50, "notes": "Early AM — large field"},
    {"name": "Davis Pasture",          "coords": (80, 45), "tons": 2, "crop": "grass",
     "window": (480, 720),  "service_min": 20, "notes": "After livestock moved"},
    {"name": "Brown Potato Field",     "coords": (45, 20), "tons": 5, "crop": "potato",
     "window": (420, 600),  "service_min": 35, "notes": "Mid-morning optimal"},
    {"name": "Lee Sunflower Plot",     "coords": (65, 90), "tons": 4, "crop": "sunflower",
     "window": (360, 600),  "service_min": 30, "notes": "Morning application"},
]

# Truck fleet
TRUCKS = [
    {"id": "Spreader-1", "capacity_tons": 20},
    {"id": "Spreader-2", "capacity_tons": 20},
    {"id": "Spreader-3", "capacity_tons": 20},
]


def create_instance() -> dict:
    """Create a fertilizer delivery VRPTW instance.

    Returns:
        Dictionary with VRPTW instance data.
    """
    n = len(FIELDS)
    depot_coords = np.array(DEPOT["coords"], dtype=float)
    field_coords = np.array([f["coords"] for f in FIELDS], dtype=float)
    all_coords = np.vstack([depot_coords.reshape(1, 2), field_coords])

    # Distance matrix (km, with rural road winding factor 1.4)
    n_total = n + 1
    dist = np.zeros((n_total, n_total))
    for i in range(n_total):
        for j in range(n_total):
            euclidean = np.sqrt(np.sum((all_coords[i] - all_coords[j]) ** 2))
            dist[i][j] = euclidean * 1.4  # rural road factor

    # Time windows (minutes from midnight)
    time_windows = np.zeros((n_total, 2))
    time_windows[0] = [360, 1080]  # depot: 6 AM to 6 PM
    for i, f in enumerate(FIELDS):
        time_windows[i + 1] = f["window"]

    # Service times (minutes for application at each field)
    service_times = np.zeros(n_total)
    for i, f in enumerate(FIELDS):
        service_times[i + 1] = f["service_min"]

    # Demands (tons of fertilizer)
    demands = np.array([f["tons"] for f in FIELDS], dtype=float)

    return {
        "n_fields": n,
        "coords": all_coords,
        "distance_matrix": dist,
        "time_windows": time_windows,
        "service_times": service_times,
        "demands": demands,
        "capacity": 20,  # tons per truck
        "fields": FIELDS,
        "depot": DEPOT,
        "trucks": TRUCKS,
    }


def solve(verbose: bool = True) -> dict:
    """Solve fertilizer delivery routing using VRPTW.

    Returns:
        Dictionary with routing results.
    """
    data = create_instance()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vrptw_dir = os.path.join(base_dir, "problems", "routing", "vrptw")

    vrptw_inst = _load_mod("vrptw_inst_fert", os.path.join(vrptw_dir, "instance.py"))
    vrptw_sol = _load_mod("vrptw_si_fert", os.path.join(vrptw_dir, "heuristics", "solomon_insertion.py"))

    vrptw_instance = vrptw_inst.VRPTWInstance(
        n=data["n_fields"],
        capacity=data["capacity"],
        demands=data["demands"],
        distance_matrix=data["distance_matrix"],
        time_windows=data["time_windows"],
        service_times=data["service_times"],
        coords=data["coords"],
        name="fertilizer_delivery",
    )

    si_sol = vrptw_sol.solomon_insertion(vrptw_instance)
    nn_tw_sol = vrptw_sol.nearest_neighbor_tw(vrptw_instance)

    results = {}
    for name, sol in [("Solomon-I1", si_sol), ("NN-TW", nn_tw_sol)]:
        results[name] = {
            "distance": sol.distance,
            "routes": sol.routes,
            "n_trucks": len([r for r in sol.routes if r]),
        }

    if verbose:
        total_tons = sum(f["tons"] for f in FIELDS)
        print("=" * 70)
        print("FERTILIZER DELIVERY ROUTE OPTIMIZATION (VRPTW)")
        print(f"  {data['n_fields']} fields, {total_tons} tons total fertilizer")
        print(f"  {len(TRUCKS)} trucks, {TRUCKS[0]['capacity_tons']} tons capacity each")
        print("=" * 70)

        # Field summary
        print("\n  Field schedule:")
        for i, f in enumerate(FIELDS):
            h1, m1 = divmod(f["window"][0], 60)
            h2, m2 = divmod(f["window"][1], 60)
            print(f"    {i+1:2d}. {f['name']:28s} {f['tons']:2d}t "
                  f"({f['crop']:10s}) [{h1:02d}:{m1:02d}-{h2:02d}:{m2:02d}] "
                  f"{f['service_min']}min")

        # Routing results
        for method, res in results.items():
            fuel_cost = res["distance"] * 0.45  # $0.45/km for heavy truck
            print(f"\n--- {method} ---")
            print(f"  Total distance: {res['distance']:.1f} km, "
                  f"{res['n_trucks']} trucks used, "
                  f"est. fuel cost: ${fuel_cost:.2f}")
            for i, route in enumerate(res["routes"]):
                if not route:
                    continue
                load = sum(FIELDS[c - 1]["tons"] for c in route)
                route_dist = vrptw_instance.route_distance(route)
                print(f"    Truck {i+1}: {len(route)} fields, "
                      f"{load} tons, {route_dist:.1f} km")
                for c in route:
                    f = FIELDS[c - 1]
                    h1, m1 = divmod(f["window"][0], 60)
                    h2, m2 = divmod(f["window"][1], 60)
                    print(f"      -> {f['name']:28s} ({f['tons']}t, "
                          f"{h1:02d}:{m1:02d}-{h2:02d}:{m2:02d})")

    return results


if __name__ == "__main__":
    solve()
