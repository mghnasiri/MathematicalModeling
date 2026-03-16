"""
Real-World Application: Harvest Crop Transport Optimization.

Domain: Agricultural logistics / Grain transport and storage
Model: CVRP (Capacitated Vehicle Routing Problem)

Scenario:
    During harvest season, a grain elevator (central depot) dispatches 4
    trucks to collect harvested crops from 12 farm fields. Each field has
    a known quantity of grain ready for pickup (3-14 tons). Trucks have a
    15-ton weight capacity. The goal is to route all trucks to collect
    from every field and return to the elevator, minimizing total travel
    distance (fuel cost).

    Fields are spread across a 100x100 km farming region with the grain
    elevator near the center. Different crops (wheat, corn, soybeans,
    barley) have different harvest quantities based on yield and acreage.

Real-world considerations modeled:
    - Truck weight capacity constraints (15 tons per truck)
    - Geographic spread of fields across farming region
    - Varying harvest quantities by crop type and field size
    - Rural road distances (winding factor applied to Euclidean distance)
    - Fuel cost estimation based on distance and truck fuel economy
    - Multiple collection trips may be required for large harvests

Industry context:
    Transport costs represent 5-10% of total crop value for commodity
    grains (USDA, 2020). During peak harvest, efficient truck routing
    reduces fuel consumption by 15-25% and minimizes field waiting time,
    which is critical when weather threatens unharvested crops. Custom
    harvest operations in the US Midwest routinely manage fleets of 4-8
    trucks serving 10-20 fields within a 50-mile radius.

References:
    Bochtis, D.D., Sorensen, C.G. & Busato, P. (2014). Advances in
    agricultural machinery management: A review. Biosystems Engineering,
    126, 69-81.
    https://doi.org/10.1016/j.biosystemseng.2014.07.012

    Oksanen, T. & Visala, A. (2009). Coverage path planning algorithms
    for agricultural field machines. Journal of Field Robotics, 26(8),
    651-668.
    https://doi.org/10.1002/rob.20300

    Clarke, G. & Wright, J.W. (1964). Scheduling of vehicles from a
    central depot to a number of delivery points. Operations Research,
    12(4), 568-581.
    https://doi.org/10.1287/opre.12.4.568
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

# Grain elevator (depot) — central collection point
ELEVATOR = {"name": "Prairie View Grain Elevator", "coords": (50, 50)}

# 12 farm fields with harvested crop quantities
FIELDS = [
    {"name": "Anderson Wheat North",  "coords": (30, 90), "tons": 12, "crop": "wheat",    "acres": 160},
    {"name": "Anderson Wheat South",  "coords": (35, 78), "tons": 10, "crop": "wheat",    "acres": 130},
    {"name": "Baker Corn East",       "coords": (80, 75), "tons": 14, "crop": "corn",     "acres": 200},
    {"name": "Baker Corn West",       "coords": (70, 68), "tons": 11, "crop": "corn",     "acres": 160},
    {"name": "Clark Soybean",         "coords": (85, 45), "tons": 8,  "crop": "soybean",  "acres": 120},
    {"name": "Davis Barley",          "coords": (75, 25), "tons": 6,  "crop": "barley",   "acres": 90},
    {"name": "Evans Corn Field",      "coords": (55, 15), "tons": 13, "crop": "corn",     "acres": 180},
    {"name": "Foster Wheat Plot",     "coords": (30, 20), "tons": 7,  "crop": "wheat",    "acres": 100},
    {"name": "Green Soybean North",   "coords": (15, 65), "tons": 9,  "crop": "soybean",  "acres": 130},
    {"name": "Harris Barley Field",   "coords": (20, 45), "tons": 5,  "crop": "barley",   "acres": 75},
    {"name": "Irving Corn Plot",      "coords": (45, 85), "tons": 3,  "crop": "corn",     "acres": 45},
    {"name": "Johnson Wheat Field",   "coords": (60, 35), "tons": 10, "crop": "wheat",    "acres": 140},
]

# Truck fleet
TRUCK_CAPACITY = 15  # tons per truck
NUM_TRUCKS = 4
FUEL_COST_PER_KM = 0.55  # $/km for loaded grain truck


def create_instance() -> dict:
    """Create a crop transport CVRP instance.

    Returns:
        Dictionary with CVRP instance data.
    """
    n = len(FIELDS)
    depot_coords = np.array(ELEVATOR["coords"], dtype=float)
    field_coords = np.array([f["coords"] for f in FIELDS], dtype=float)
    all_coords = np.vstack([depot_coords.reshape(1, 2), field_coords])

    # Distance matrix (km, with rural road winding factor 1.35)
    n_total = n + 1
    dist = np.zeros((n_total, n_total))
    for i in range(n_total):
        for j in range(n_total):
            euclidean = np.sqrt(np.sum((all_coords[i] - all_coords[j]) ** 2))
            dist[i][j] = euclidean * 1.35  # rural road factor

    # Demands (tons of grain to pick up at each field)
    demands = np.array([f["tons"] for f in FIELDS], dtype=float)

    return {
        "n_fields": n,
        "coords": all_coords,
        "distance_matrix": dist,
        "demands": demands,
        "capacity": TRUCK_CAPACITY,
        "num_trucks": NUM_TRUCKS,
        "fields": FIELDS,
        "elevator": ELEVATOR,
    }


def solve(verbose: bool = True) -> dict:
    """Solve crop transport routing using CVRP.

    Returns:
        Dictionary with routing results.
    """
    data = create_instance()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cvrp_dir = os.path.join(base_dir, "problems", "routing", "cvrp")

    cvrp_inst = _load_mod("cvrp_inst_crop", os.path.join(cvrp_dir, "instance.py"))
    cw_mod = _load_mod("cvrp_cw_crop", os.path.join(cvrp_dir, "heuristics", "clarke_wright.py"))
    sa_mod = _load_mod("cvrp_sa_crop", os.path.join(cvrp_dir, "metaheuristics", "simulated_annealing.py"))

    instance = cvrp_inst.CVRPInstance(
        n=data["n_fields"],
        coords=data["coords"],
        demands=data["demands"],
        capacity=data["capacity"],
        distance_matrix=data["distance_matrix"],
        name="crop_transport",
    )

    # Clarke-Wright savings heuristic
    cw_sol = cw_mod.clarke_wright_savings(instance)

    # Simulated annealing improvement
    sa_sol = sa_mod.simulated_annealing(
        instance, max_iterations=30_000, seed=42
    )

    results = {}
    for name, sol in [("Clarke-Wright", cw_sol), ("SA", sa_sol)]:
        results[name] = {
            "distance": sol.distance,
            "routes": sol.routes,
            "n_trucks": len(sol.routes),
        }

    if verbose:
        total_tons = sum(f["tons"] for f in FIELDS)
        total_acres = sum(f["acres"] for f in FIELDS)
        print("=" * 70)
        print("HARVEST CROP TRANSPORT OPTIMIZATION (CVRP)")
        print(f"  {data['n_fields']} fields, {total_tons} tons total harvest")
        print(f"  {total_acres} total acres, {NUM_TRUCKS} trucks @ "
              f"{TRUCK_CAPACITY}t capacity")
        print("=" * 70)

        # Field summary
        print("\n  Field inventory:")
        for i, f in enumerate(FIELDS):
            print(f"    {i+1:2d}. {f['name']:25s} {f['tons']:2d}t "
                  f"({f['crop']:8s}, {f['acres']:3d} acres)")

        # Minimum trips needed
        min_trips = 0
        for f in FIELDS:
            min_trips += max(1, -(-f["tons"] // TRUCK_CAPACITY))
        print(f"\n  Minimum trips needed: {min_trips} "
              f"(theoretical: ceil({total_tons}/{TRUCK_CAPACITY}) = "
              f"{-(-total_tons // TRUCK_CAPACITY)})")

        # Routing results
        for method, res in results.items():
            fuel_cost = res["distance"] * FUEL_COST_PER_KM
            print(f"\n--- {method} ---")
            print(f"  Total distance: {res['distance']:.1f} km, "
                  f"{res['n_trucks']} routes, "
                  f"est. fuel cost: ${fuel_cost:.2f}")
            for i, route in enumerate(res["routes"]):
                load = sum(data["demands"][c - 1] for c in route)
                util = load / TRUCK_CAPACITY * 100
                crops = [FIELDS[c - 1]["crop"] for c in route]
                crop_summary = ", ".join(sorted(set(crops)))
                route_dist = instance.route_distance(route)
                print(f"    Route {i+1}: {len(route)} fields, "
                      f"{load:.0f}/{TRUCK_CAPACITY}t ({util:.0f}%), "
                      f"{route_dist:.1f} km — [{crop_summary}]")
                for c in route:
                    f = FIELDS[c - 1]
                    print(f"      -> {f['name']:25s} ({f['tons']}t {f['crop']})")

        # Cost comparison
        cw_cost = results["Clarke-Wright"]["distance"] * FUEL_COST_PER_KM
        sa_cost = results["SA"]["distance"] * FUEL_COST_PER_KM
        savings = cw_cost - sa_cost
        if savings > 0:
            print(f"\n  SA improvement: ${savings:.2f} fuel savings "
                  f"({savings / cw_cost * 100:.1f}% reduction)")

    return results


if __name__ == "__main__":
    solve()
