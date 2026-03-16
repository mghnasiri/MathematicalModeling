#!/usr/bin/env python3
"""
OpenRouteService Integration Demo — Real-World OR Problems.

This script demonstrates how to create and solve Operations Research
problems using real-world road-network data from OpenRouteService,
and visualize results on interactive maps.

Run: python examples/ors_demo.py

Generated maps are saved as HTML files in the examples/ directory.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Add project root
_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _ROOT)

from shared.api.openrouteservice import ORSClient
from shared.visualization.map_viz import (
    plot_tsp_tour,
    plot_vrp_routes,
    plot_facility_location,
    plot_distance_matrix,
)


def demo_tsp_german_cities():
    """TSP tour through German cities with real road distances."""
    print("=" * 60)
    print("Demo 1: TSP — German Cities Tour")
    print("=" * 60)

    # City coordinates [longitude, latitude]
    cities = {
        "Heidelberg":  [8.6946, 49.4058],
        "Mannheim":    [8.4669, 49.4875],
        "Karlsruhe":   [8.4037, 49.0069],
        "Stuttgart":   [9.1829, 48.7758],
        "Frankfurt":   [8.6821, 50.1109],
        "Darmstadt":   [8.6512, 49.8728],
    }

    city_names = list(cities.keys())
    coords = np.array(list(cities.values()))

    # Create TSP instance with real road distances
    from problems.routing.tsp.instance import TSPInstance
    inst = TSPInstance.from_ors(
        locations=coords.tolist(),
        metric="distance",
        name="german_cities_6",
    )

    print(f"  Created TSP with {inst.n} cities")
    print(f"  Distance matrix (km):")
    for i in range(inst.n):
        row = [f"{inst.distance_matrix[i][j] / 1000:.1f}" for j in range(inst.n)]
        print(f"    {city_names[i]:12s}: {', '.join(row)}")

    # Solve with nearest neighbor
    from problems.routing.tsp.heuristics.nearest_neighbor import nearest_neighbor
    sol = nearest_neighbor(inst)
    print(f"\n  Nearest Neighbor tour: {[city_names[i] for i in sol.tour]}")
    print(f"  Total distance: {sol.distance / 1000:.1f} km")

    # Try local search improvement
    from problems.routing.tsp.metaheuristics.local_search import two_opt
    improved = two_opt(inst, sol.tour)
    print(f"  After 2-opt: {[city_names[i] for i in improved.tour]}")
    print(f"  Improved distance: {improved.distance / 1000:.1f} km")

    # Visualize on map
    save_path = str(Path(__file__).parent / "tsp_german_cities.html")
    plot_tsp_tour(
        coords,
        tour=improved.tour,
        city_labels=city_names,
        title=f"TSP: German Cities ({improved.distance / 1000:.0f} km)",
        use_roads=True,
        save_path=save_path,
    )
    print(f"\n  Map saved to: {save_path}")


def demo_cvrp_delivery():
    """CVRP delivery problem with real distances."""
    print("\n" + "=" * 60)
    print("Demo 2: CVRP — Heidelberg Delivery Routes")
    print("=" * 60)

    # Depot (warehouse) and customer locations in Heidelberg
    depot = [8.6946, 49.4058]  # Heidelberg center
    customers = [
        [8.7062, 49.4100],  # Altstadt
        [8.6513, 49.4189],  # Neuenheim
        [8.6730, 49.4290],  # Handschuhsheim
        [8.7197, 49.3960],  # Rohrbach
        [8.6380, 49.3940],  # Kirchheim
        [8.7100, 49.4200],  # Ziegelhausen
    ]
    demands = [15, 20, 10, 25, 18, 12]

    from problems.routing.cvrp.instance import CVRPInstance
    inst = CVRPInstance.from_ors(
        depot=depot,
        customers=customers,
        demands=demands,
        capacity=50,
        metric="duration",  # Use travel time
        name="heidelberg_delivery",
    )

    print(f"  Created CVRP with {inst.n} customers")
    print(f"  Vehicle capacity: {inst.capacity}")
    print(f"  Demands: {inst.demands}")

    # Solve with Clarke-Wright
    from problems.routing.cvrp.heuristics.clarke_wright import clarke_wright
    sol = clarke_wright(inst)
    print(f"\n  Clarke-Wright solution:")
    print(f"    Routes: {sol.routes}")
    print(f"    Vehicles used: {sol.num_vehicles}")
    print(f"    Total travel time: {sol.distance:.0f} seconds")

    # Visualize
    all_coords = np.vstack([depot, customers])
    save_path = str(Path(__file__).parent / "cvrp_heidelberg.html")
    plot_vrp_routes(
        all_coords,
        sol.routes,
        demands=inst.demands,
        title=f"CVRP: Heidelberg ({sol.num_vehicles} vehicles)",
        use_roads=True,
        save_path=save_path,
    )
    print(f"  Map saved to: {save_path}")


def demo_facility_location():
    """Facility location with isochrones."""
    print("\n" + "=" * 60)
    print("Demo 3: Facility Location — Warehouse Placement")
    print("=" * 60)

    # Candidate warehouse locations
    facilities = [
        [8.6946, 49.4058],  # Heidelberg center
        [8.4669, 49.4875],  # Mannheim
        [8.4037, 49.0069],  # Karlsruhe
    ]
    fac_names = ["Heidelberg", "Mannheim", "Karlsruhe"]

    # Customer cities
    customers = [
        [8.6512, 49.8728],  # Darmstadt
        [9.1829, 48.7758],  # Stuttgart
        [8.2324, 49.9929],  # Wiesbaden
        [8.5473, 49.4740],  # Weinheim
        [8.5410, 49.3167],  # Wiesloch
    ]
    cust_names = ["Darmstadt", "Stuttgart", "Wiesbaden", "Weinheim", "Wiesloch"]

    fixed_costs = [5000, 6000, 5500]

    from problems.location_network.facility_location.instance import (
        FacilityLocationInstance,
    )
    inst = FacilityLocationInstance.from_ors(
        facilities=facilities,
        customers=customers,
        fixed_costs=fixed_costs,
        metric="duration",
        name="rhein_neckar_fl",
    )

    print(f"  Created UFLP: {inst.m} facilities, {inst.n} customers")
    print(f"  Fixed costs: {inst.fixed_costs}")
    print(f"  Assignment costs (travel time in seconds):")
    for i in range(inst.m):
        row = [f"{inst.assignment_costs[i][j]:.0f}" for j in range(inst.n)]
        print(f"    {fac_names[i]:12s}: {', '.join(row)}")

    # Solve with greedy
    from problems.location_network.facility_location.heuristics.greedy_facility import (
        greedy_add,
    )
    sol = greedy_add(inst)
    print(f"\n  Greedy solution:")
    print(f"    Open: {[fac_names[i] for i in sol.open_facilities]}")
    print(f"    Assignments: {[fac_names[sol.assignments[j]] for j in range(inst.n)]}")
    print(f"    Total cost: {sol.cost:.0f}")

    # Visualize with isochrones
    fac_arr = np.array(facilities)
    cust_arr = np.array(customers)
    save_path = str(Path(__file__).parent / "facility_location.html")
    plot_facility_location(
        fac_arr,
        cust_arr,
        open_facilities=sol.open_facilities,
        assignments=sol.assignments,
        facility_labels=fac_names,
        customer_labels=cust_names,
        isochrone_seconds=[1800, 3600],  # 30min, 60min
        title="Facility Location: Rhine-Neckar",
        save_path=save_path,
    )
    print(f"  Map saved to: {save_path}")


def demo_distance_matrix_heatmap():
    """Visualize a real distance matrix as a heatmap."""
    print("\n" + "=" * 60)
    print("Demo 4: Distance Matrix Heatmap")
    print("=" * 60)

    client = ORSClient()
    cities = {
        "Heidelberg":  [8.6946, 49.4058],
        "Mannheim":    [8.4669, 49.4875],
        "Karlsruhe":   [8.4037, 49.0069],
        "Stuttgart":   [9.1829, 48.7758],
        "Frankfurt":   [8.6821, 50.1109],
    }

    coords = np.array(list(cities.values()))
    matrix = client.distance_matrix(coords, metric="distance")

    # Convert to km
    matrix_km = matrix / 1000.0
    print(f"  Distance matrix (km):")
    names = list(cities.keys())
    for i in range(len(names)):
        row = [f"{matrix_km[i][j]:6.1f}" for j in range(len(names))]
        print(f"    {names[i]:12s}: {', '.join(row)}")

    import matplotlib
    matplotlib.use("Agg")
    save_path = str(Path(__file__).parent / "distance_heatmap.png")
    plot_distance_matrix(
        matrix_km,
        labels=names,
        title="Road Distances (km)",
        save_path=save_path,
    )
    print(f"  Heatmap saved to: {save_path}")


if __name__ == "__main__":
    print("OpenRouteService Integration Demo")
    print("Fetching real road-network data...\n")

    try:
        demo_tsp_german_cities()
        demo_cvrp_delivery()
        demo_facility_location()
        demo_distance_matrix_heatmap()
        print("\n" + "=" * 60)
        print("All demos complete! Open the HTML files in your browser.")
        print("=" * 60)
    except RuntimeError as e:
        print(f"\nAPI Error: {e}")
        print("Make sure you have a valid ORS API key.")
        print("Set it via: export ORS_API_KEY=your_key_here")
