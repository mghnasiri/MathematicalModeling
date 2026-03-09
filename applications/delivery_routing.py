"""
Real-World Application: Last-Mile Delivery Route Optimization.

Domain: E-commerce grocery delivery / Food delivery fleet
Model: CVRP + VRPTW

Scenario:
    A grocery delivery company operates from a central warehouse in a
    metropolitan area. On a typical morning, 25 orders need delivery
    within customer-specified time windows. The fleet has refrigerated
    vans with 300 kg capacity. Each delivery takes ~10 minutes for
    unloading and customer handoff.

    Objective: Minimize total distance (fuel cost) while respecting
    vehicle capacity and customer time windows.

Real-world considerations modeled:
    - Customer time windows (morning, lunch, afternoon slots)
    - Vehicle capacity (weight-limited refrigerated vans)
    - Service time at each stop
    - Depot opening hours (planning horizon)
    - Geographic clustering (urban neighborhoods)

Industry context:
    Last-mile delivery accounts for 40-50% of total supply chain costs
    (Gevaers et al., 2014). Route optimization typically yields 10-30%
    distance savings over manual planning.

References:
    Solomon, M.M. (1987). Algorithms for the vehicle routing and
    scheduling problems with time window constraints. Operations
    Research, 35(2), 254-265.
    https://doi.org/10.1287/opre.35.2.254

    Gevaers, R., Van de Voorde, E. & Vanelslander, T. (2014).
    Cost modelling and simulation of last-mile characteristics in
    an innovative B2C supply chain environment with implications on
    urban areas and cities. Procedia - Social and Behavioral Sciences,
    125, 398-411.
    https://doi.org/10.1016/j.sbspro.2014.01.1483

    Toth, P. & Vigo, D. (2014). Vehicle Routing: Problems, Methods,
    and Applications. 2nd ed. SIAM, Philadelphia.
    https://doi.org/10.1137/1.9781611973594
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

# Neighborhood clusters around a central warehouse
NEIGHBORHOODS = {
    "Downtown": {"center": (50, 50), "radius": 8},
    "Northside": {"center": (50, 80), "radius": 10},
    "Eastside": {"center": (80, 50), "radius": 10},
    "Southside": {"center": (50, 20), "radius": 10},
    "Westside": {"center": (20, 50), "radius": 10},
}

# Time window slots (minutes from 8:00 AM)
TIME_SLOTS = {
    "Morning (8-10 AM)": (0, 120),
    "Late Morning (10-12 PM)": (120, 240),
    "Lunch (12-2 PM)": (240, 360),
    "Afternoon (2-4 PM)": (360, 480),
}


def create_delivery_instance(
    n_customers: int = 20, seed: int = 42
) -> dict:
    """Create a grocery delivery instance with realistic parameters.

    Args:
        n_customers: Number of delivery stops.
        seed: Random seed.

    Returns:
        Dictionary with coordinates, demands, time windows, etc.
    """
    rng = np.random.default_rng(seed)

    # Depot at center
    depot = np.array([50.0, 50.0])

    # Generate customers in neighborhood clusters
    coords = [depot]
    neighborhoods = []
    for _ in range(n_customers):
        hood_name = rng.choice(list(NEIGHBORHOODS.keys()))
        hood = NEIGHBORHOODS[hood_name]
        angle = rng.uniform(0, 2 * np.pi)
        r = rng.uniform(0, hood["radius"])
        x = hood["center"][0] + r * np.cos(angle)
        y = hood["center"][1] + r * np.sin(angle)
        coords.append(np.array([x, y]))
        neighborhoods.append(hood_name)

    coords = np.array(coords)

    # Demands (kg): typical grocery order 5-40 kg
    demands = np.zeros(n_customers + 1, dtype=int)
    demands[1:] = rng.integers(5, 41, size=n_customers)

    # Time windows based on customer preference slots
    slot_names = list(TIME_SLOTS.keys())
    slot_weights = [0.3, 0.25, 0.25, 0.2]
    earliest = np.zeros(n_customers + 1, dtype=int)
    latest = np.full(n_customers + 1, 480, dtype=int)

    customer_slots = []
    for i in range(n_customers):
        slot = rng.choice(slot_names, p=slot_weights)
        e, l = TIME_SLOTS[slot]
        earliest[i + 1] = e
        latest[i + 1] = l
        customer_slots.append(slot)

    # Depot: full planning horizon
    earliest[0] = 0
    latest[0] = 480

    # Service time: 10 minutes per stop
    service_times = np.zeros(n_customers + 1, dtype=int)
    service_times[1:] = 10

    # Distance matrix (Euclidean, scaled to km)
    n = n_customers + 1
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i][j] = np.sqrt(np.sum((coords[i] - coords[j]) ** 2))

    return {
        "n_customers": n_customers,
        "coords": coords,
        "demands": demands,
        "capacity": 300,  # kg per van
        "earliest": earliest,
        "latest": latest,
        "service_times": service_times,
        "distance_matrix": dist,
        "neighborhoods": neighborhoods,
        "customer_slots": customer_slots,
        "depot": depot,
    }


def solve_cvrp(data: dict, verbose: bool = True) -> dict:
    """Solve the delivery problem as CVRP (capacity only).

    Returns:
        Dictionary with solution details.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cvrp_dir = os.path.join(base_dir, "problems", "routing", "cvrp")

    cvrp_inst = _load_mod("cvrp_inst_app", os.path.join(cvrp_dir, "instance.py"))
    cw_mod = _load_mod("cvrp_cw_app", os.path.join(cvrp_dir, "heuristics", "clarke_wright.py"))
    sa_mod = _load_mod("cvrp_sa_app", os.path.join(cvrp_dir, "metaheuristics", "simulated_annealing.py"))

    # CVRPInstance demands shape is (n,) — customers only, no depot
    instance = cvrp_inst.CVRPInstance(
        n=data["n_customers"],
        coords=data["coords"],
        demands=data["demands"][1:],  # exclude depot
        capacity=data["capacity"],
        distance_matrix=data["distance_matrix"],
        name="grocery_delivery",
    )

    # Clarke-Wright savings
    cw_sol = cw_mod.clarke_wright_savings(instance)

    # SA improvement
    sa_sol = sa_mod.simulated_annealing(
        instance, max_iterations=20_000, seed=42
    )

    results = {
        "Clarke-Wright": {
            "distance": cw_sol.distance,
            "routes": cw_sol.routes,
            "n_vehicles": len(cw_sol.routes),
        },
        "SA": {
            "distance": sa_sol.distance,
            "routes": sa_sol.routes,
            "n_vehicles": len(sa_sol.routes),
        },
    }

    if verbose:
        print("=" * 70)
        print("GROCERY DELIVERY ROUTING (CVRP)")
        print(f"  {data['n_customers']} customers, capacity={data['capacity']} kg")
        print("=" * 70)

        for method, res in results.items():
            fuel_cost = res["distance"] * 0.15  # $0.15 per km
            print(f"\n{method}: distance = {res['distance']:.1f} km, "
                  f"vehicles = {res['n_vehicles']}, "
                  f"est. fuel = ${fuel_cost:.2f}")
            for i, route in enumerate(res["routes"]):
                load = sum(data["demands"][c] for c in route)
                hoods = [data["neighborhoods"][c - 1] for c in route]
                hood_summary = ", ".join(sorted(set(hoods)))
                print(f"    Route {i+1}: {len(route)} stops, "
                      f"{load} kg — [{hood_summary}]")

    return results


def solve_vrptw(data: dict, verbose: bool = True) -> dict:
    """Solve the delivery problem as VRPTW (capacity + time windows).

    Returns:
        Dictionary with solution details.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vrptw_dir = os.path.join(base_dir, "problems", "routing", "vrptw")

    vrptw_inst = _load_mod("vrptw_inst_app", os.path.join(vrptw_dir, "instance.py"))
    solomon_mod = _load_mod("vrptw_sol_app", os.path.join(vrptw_dir, "heuristics", "solomon_insertion.py"))

    # VRPTWInstance uses time_windows (n+1, 2) and demands (n,)
    n = data["n_customers"]
    time_windows = np.column_stack([data["earliest"], data["latest"]])
    instance = vrptw_inst.VRPTWInstance(
        n=n,
        coords=data["coords"],
        demands=data["demands"][1:],  # exclude depot
        capacity=data["capacity"],
        distance_matrix=data["distance_matrix"],
        time_windows=time_windows,
        service_times=data["service_times"],
        name="grocery_delivery_tw",
    )

    # Solomon I1 insertion
    sol = solomon_mod.solomon_insertion(instance)

    results = {
        "Solomon-I1": {
            "distance": sol.distance,
            "routes": sol.routes,
            "n_vehicles": len(sol.routes),
        },
    }

    if verbose:
        print("\n" + "=" * 70)
        print("GROCERY DELIVERY ROUTING WITH TIME WINDOWS (VRPTW)")
        print(f"  {data['n_customers']} customers, capacity={data['capacity']} kg")
        print("=" * 70)

        for method, res in results.items():
            print(f"\n{method}: distance = {res['distance']:.1f} km, "
                  f"vehicles = {res['n_vehicles']}")
            for i, route in enumerate(res["routes"]):
                load = sum(data["demands"][c] for c in route)
                windows = [(data["customer_slots"][c - 1]) for c in route]
                window_str = ", ".join(sorted(set(windows)))
                print(f"    Route {i+1}: {len(route)} stops, "
                      f"{load} kg — [{window_str}]")

    return results


if __name__ == "__main__":
    data = create_delivery_instance(n_customers=20, seed=42)
    solve_cvrp(data, verbose=True)
    solve_vrptw(data, verbose=True)
