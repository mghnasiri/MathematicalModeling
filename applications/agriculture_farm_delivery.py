"""
Real-World Application: Farm-to-Market Fresh Produce Delivery Routing.

Domain: Agricultural cooperative logistics / Fresh produce distribution
Model: CVRP + Stochastic VRP

Scenario:
    A cooperative of farms near Quebec City distributes fresh produce
    (vegetables, fruits, dairy, preserves) to 15 delivery points across
    the region. Delivery points include farmers markets, restaurant
    kitchens, grocery stores, and food banks. A fleet of refrigerated
    trucks (capacity 2000 kg each) departs daily from the cooperative's
    central distribution center (depot).

    Daily demand varies due to seasonal fluctuations, weather-driven
    consumption changes, and event-based spikes (e.g., weekend markets).
    The cooperative must decide routes before exact orders are finalized,
    making stochastic modeling essential for reliable service.

    Objective: Minimize total delivery distance (fuel and driver costs)
    while ensuring demand can be met with high probability despite
    daily variation.

Real-world considerations modeled:
    - Heterogeneous delivery points (markets, restaurants, stores, food banks)
    - Refrigerated truck capacity limits (2000 kg)
    - Demand uncertainty from daily order variation (±15-25%)
    - Chance constraints ensuring route feasibility at 90% confidence
    - Recourse costs when a truck must return to depot mid-route

Industry context:
    Short food supply chains reduce food miles and post-harvest losses.
    Route optimization for perishable goods is critical: refrigerated
    transport costs 1.5-2x dry freight, and spoilage risk grows with
    transit time. Canadian cooperatives serve 3,400+ farmers markets
    and 97,000 food service establishments (Agriculture and Agri-Food
    Canada, 2022).

References:
    Clarke, G. & Wright, J.W. (1964). Scheduling of vehicles from a
    central depot to a number of delivery points. Operations Research,
    12(4), 568-581.
    https://doi.org/10.1287/opre.12.4.568

    Bertsimas, D.J. (1992). A vehicle routing problem with stochastic
    demand. Operations Research, 40(3), 574-585.
    https://doi.org/10.1287/opre.40.3.574

    Bosona, T. & Gebresenbet, G. (2011). Cluster building and logistics
    network integration of local food supply chain. Biosystems
    Engineering, 108(4), 293-302.
    https://doi.org/10.1016/j.biosystemseng.2011.01.001

    Akkerman, R., Farahani, P. & Grunow, M. (2010). Quality, safety
    and sustainability in food distribution: a review of quantitative
    operations management approaches and challenges. OR Spectrum,
    32(4), 863-904.
    https://doi.org/10.1007/s00291-010-0223-2
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

# Delivery points: (name, type, base_demand_kg)
DELIVERY_POINTS = [
    # Farmers markets (300-500 kg)
    ("Marche du Vieux-Port", "farmers_market", 450),
    ("Marche de Sainte-Foy", "farmers_market", 380),
    ("Marche de Limoilou", "farmers_market", 320),
    ("Marche de Levis", "farmers_market", 500),
    # Restaurants (50-150 kg)
    ("Bistro Le Clocher Penche", "restaurant", 80),
    ("Restaurant Panache", "restaurant", 120),
    ("Chez Boulay", "restaurant", 95),
    ("Le Saint-Amour", "restaurant", 65),
    ("Cafe du Monde", "restaurant", 150),
    # Grocery stores (200-400 kg)
    ("Epicerie Bio Quebec", "grocery", 350),
    ("Marche Tradition Beauport", "grocery", 280),
    ("IGA Extra Charlesbourg", "grocery", 400),
    ("Metro Plus Sillery", "grocery", 220),
    # Food banks (100-250 kg)
    ("Moisson Quebec", "food_bank", 250),
    ("Banque Alimentaire Levis", "food_bank", 150),
]

# Delivery point types for reporting
POINT_TYPE_LABELS = {
    "farmers_market": "Farmers Market",
    "restaurant": "Restaurant",
    "grocery": "Grocery Store",
    "food_bank": "Food Bank",
}

# Vehicle parameters
TRUCK_CAPACITY_KG = 2000
FUEL_COST_PER_KM = 0.85  # CAD, refrigerated truck


def create_farm_delivery_instance(seed: int = 42) -> dict:
    """Create a farm-to-market delivery instance with realistic parameters.

    Generates 15 delivery points spread across a 50x50 km area around
    Quebec City, with demands reflecting typical daily orders for each
    type of establishment.

    Args:
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with coordinates, demands, delivery point metadata,
        and demand scenarios for the stochastic model.
    """
    rng = np.random.default_rng(seed)
    n_customers = len(DELIVERY_POINTS)

    # Depot: cooperative distribution center (roughly center of service area)
    depot = np.array([25.0, 25.0])

    # Generate realistic coordinates spread across 50x50 km area
    # Farmers markets: urban core (closer to center)
    # Restaurants: downtown cluster
    # Grocery stores: suburban spread
    # Food banks: varied locations
    location_profiles = {
        "farmers_market": {"center": (20.0, 30.0), "spread": 12.0},
        "restaurant": {"center": (28.0, 28.0), "spread": 6.0},
        "grocery": {"center": (25.0, 25.0), "spread": 15.0},
        "food_bank": {"center": (22.0, 20.0), "spread": 10.0},
    }

    coords = [depot]
    for name, point_type, base_demand in DELIVERY_POINTS:
        profile = location_profiles[point_type]
        cx, cy = profile["center"]
        spread = profile["spread"]
        x = np.clip(cx + rng.normal(0, spread), 0, 50)
        y = np.clip(cy + rng.normal(0, spread), 0, 50)
        coords.append(np.array([x, y]))

    coords = np.array(coords)

    # Deterministic demands (base values from DELIVERY_POINTS)
    demands = np.array(
        [dp[2] for dp in DELIVERY_POINTS], dtype=float
    )

    # Distance matrix (Euclidean, in km)
    n = n_customers + 1
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i][j] = np.sqrt(np.sum((coords[i] - coords[j]) ** 2))

    # Demand scenarios for stochastic model (50 scenarios)
    # Daily variation: markets ±20%, restaurants ±25%, grocery ±15%, food banks ±20%
    n_scenarios = 50
    variation = {
        "farmers_market": 0.20,
        "restaurant": 0.25,
        "grocery": 0.15,
        "food_bank": 0.20,
    }

    demand_scenarios = np.zeros((n_scenarios, n_customers))
    for s in range(n_scenarios):
        for c in range(n_customers):
            _, point_type, base = DELIVERY_POINTS[c]
            var = variation[point_type]
            noisy = base * (1.0 + rng.uniform(-var, var))
            demand_scenarios[s, c] = max(10.0, noisy)

    # Metadata
    names = [dp[0] for dp in DELIVERY_POINTS]
    types = [dp[1] for dp in DELIVERY_POINTS]

    return {
        "n_customers": n_customers,
        "coords": coords,
        "demands": demands,
        "capacity": TRUCK_CAPACITY_KG,
        "distance_matrix": dist,
        "demand_scenarios": demand_scenarios,
        "names": names,
        "types": types,
        "depot": depot,
    }


def solve_deterministic(data: dict, verbose: bool = True) -> dict:
    """Solve the farm delivery problem as deterministic CVRP.

    Uses Clarke-Wright savings, Sweep algorithm, and Simulated Annealing.

    Args:
        data: Instance data from create_farm_delivery_instance().
        verbose: If True, print detailed results.

    Returns:
        Dictionary with solution details per method.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cvrp_dir = os.path.join(base_dir, "problems", "routing", "cvrp")

    cvrp_inst = _load_mod(
        "cvrp_inst_farm", os.path.join(cvrp_dir, "instance.py")
    )
    cw_mod = _load_mod(
        "cvrp_cw_farm", os.path.join(cvrp_dir, "heuristics", "clarke_wright.py")
    )
    sweep_mod = _load_mod(
        "cvrp_sweep_farm", os.path.join(cvrp_dir, "heuristics", "sweep.py")
    )
    sa_mod = _load_mod(
        "cvrp_sa_farm",
        os.path.join(cvrp_dir, "metaheuristics", "simulated_annealing.py"),
    )

    instance = cvrp_inst.CVRPInstance(
        n=data["n_customers"],
        coords=data["coords"],
        demands=data["demands"],
        capacity=data["capacity"],
        distance_matrix=data["distance_matrix"],
        name="farm_delivery_deterministic",
    )

    # Clarke-Wright savings
    cw_sol = cw_mod.clarke_wright_savings(instance)

    # Sweep (multi-start)
    sweep_sol = sweep_mod.sweep_multistart(instance, num_starts=12)

    # Simulated Annealing improvement
    sa_sol = sa_mod.simulated_annealing(
        instance, max_iterations=30_000, seed=42
    )

    results = {
        "Clarke-Wright": {
            "distance": cw_sol.distance,
            "routes": cw_sol.routes,
            "n_vehicles": len(cw_sol.routes),
        },
        "Sweep": {
            "distance": sweep_sol.distance,
            "routes": sweep_sol.routes,
            "n_vehicles": len(sweep_sol.routes),
        },
        "SA": {
            "distance": sa_sol.distance,
            "routes": sa_sol.routes,
            "n_vehicles": len(sa_sol.routes),
        },
    }

    if verbose:
        print("=" * 70)
        print("FARM-TO-MARKET DELIVERY ROUTING (Deterministic CVRP)")
        print(
            f"  {data['n_customers']} delivery points, "
            f"truck capacity = {data['capacity']} kg"
        )
        total_demand = data["demands"].sum()
        min_trucks = int(np.ceil(total_demand / data["capacity"]))
        print(
            f"  Total demand = {total_demand:.0f} kg, "
            f"min trucks needed = {min_trucks}"
        )
        print("=" * 70)

        for method, res in results.items():
            fuel_cost = res["distance"] * FUEL_COST_PER_KM
            print(
                f"\n{method}: distance = {res['distance']:.1f} km, "
                f"vehicles = {res['n_vehicles']}, "
                f"est. fuel = ${fuel_cost:.2f} CAD"
            )
            for i, route in enumerate(res["routes"]):
                load = sum(data["demands"][c - 1] for c in route)
                stop_names = [data["names"][c - 1] for c in route]
                stop_types = [
                    POINT_TYPE_LABELS[data["types"][c - 1]] for c in route
                ]
                type_summary = ", ".join(sorted(set(stop_types)))
                print(
                    f"    Route {i + 1}: {len(route)} stops, "
                    f"{load:.0f} kg — [{type_summary}]"
                )
                for name in stop_names:
                    print(f"      - {name}")

    return results


def solve_stochastic(data: dict, verbose: bool = True) -> dict:
    """Solve the farm delivery problem as Stochastic VRP.

    Uses chance-constrained Clarke-Wright savings and Simulated Annealing
    with recourse penalty to handle demand uncertainty.

    Args:
        data: Instance data from create_farm_delivery_instance().
        verbose: If True, print detailed results.

    Returns:
        Dictionary with solution details per method.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    svrp_dir = os.path.join(
        base_dir, "problems", "stochastic_robust", "stochastic_vrp"
    )

    svrp_inst = _load_mod(
        "svrp_inst_farm", os.path.join(svrp_dir, "instance.py")
    )
    cc_cw_mod = _load_mod(
        "svrp_cccw_farm",
        os.path.join(svrp_dir, "heuristics", "chance_constrained_cw.py"),
    )
    sa_mod = _load_mod(
        "svrp_sa_farm",
        os.path.join(svrp_dir, "metaheuristics", "simulated_annealing.py"),
    )

    # Build StochasticVRPInstance
    n = data["n_customers"]
    n_vehicles = int(
        np.ceil(data["demands"].sum() / data["capacity"])
    ) + 1  # allow one extra truck

    instance = svrp_inst.StochasticVRPInstance(
        n_customers=n,
        coordinates=data["coords"],
        demand_scenarios=data["demand_scenarios"],
        vehicle_capacity=data["capacity"],
        n_vehicles=n_vehicles,
        alpha=0.10,  # 90% reliability
    )

    # Chance-constrained Clarke-Wright
    cc_sol = cc_cw_mod.chance_constrained_savings(instance)

    # Simulated Annealing with recourse
    sa_sol = sa_mod.simulated_annealing(
        instance, max_iterations=5000, seed=42
    )

    results = {
        "CC-Clarke-Wright": {
            "distance": cc_sol.total_distance,
            "expected_cost": cc_sol.expected_total_cost,
            "routes": cc_sol.routes,
            "n_vehicles": cc_sol.n_routes,
            "max_overflow": cc_sol.max_overflow_prob,
        },
        "SA-Recourse": {
            "distance": sa_sol.total_distance,
            "expected_cost": sa_sol.expected_total_cost,
            "routes": sa_sol.routes,
            "n_vehicles": sa_sol.n_routes,
            "max_overflow": sa_sol.max_overflow_prob,
        },
    }

    if verbose:
        print("\n" + "=" * 70)
        print("FARM-TO-MARKET DELIVERY ROUTING (Stochastic VRP)")
        print(
            f"  {n} delivery points, "
            f"truck capacity = {data['capacity']} kg, "
            f"alpha = 0.10 (90% reliability)"
        )
        print(
            f"  {data['demand_scenarios'].shape[0]} demand scenarios "
            f"modeling daily variation"
        )
        print("=" * 70)

        for method, res in results.items():
            fuel_cost = res["distance"] * FUEL_COST_PER_KM
            print(
                f"\n{method}: distance = {res['distance']:.1f} km, "
                f"vehicles = {res['n_vehicles']}, "
                f"est. fuel = ${fuel_cost:.2f} CAD"
            )
            print(
                f"    Expected total cost (dist + recourse) = "
                f"{res['expected_cost']:.1f} km-equiv"
            )
            print(
                f"    Max route overflow probability = "
                f"{res['max_overflow']:.3f} "
                f"({'OK' if res['max_overflow'] <= 0.10 + 1e-9 else 'EXCEEDS'})"
            )
            for i, route in enumerate(res["routes"]):
                overflow_p = instance.route_overflow_probability(route)
                stop_names = [data["names"][c - 1] for c in route]
                mean_load = sum(
                    instance.mean_demands[c - 1] for c in route
                )
                print(
                    f"    Route {i + 1}: {len(route)} stops, "
                    f"mean load = {mean_load:.0f} kg, "
                    f"P(overflow) = {overflow_p:.3f}"
                )
                for name in stop_names:
                    print(f"      - {name}")

    return results


if __name__ == "__main__":
    data = create_farm_delivery_instance(seed=42)

    det_results = solve_deterministic(data, verbose=True)
    sto_results = solve_stochastic(data, verbose=True)

    # Comparison summary
    print("\n" + "=" * 70)
    print("COMPARISON: Deterministic vs. Stochastic Routing")
    print("=" * 70)

    best_det = min(det_results.values(), key=lambda r: r["distance"])
    best_sto_method = min(sto_results, key=lambda m: sto_results[m]["distance"])
    best_sto = sto_results[best_sto_method]

    det_dist = best_det["distance"]
    sto_dist = best_sto["distance"]
    extra_pct = (sto_dist - det_dist) / det_dist * 100 if det_dist > 0 else 0

    print(f"\n  Best deterministic distance:  {det_dist:.1f} km "
          f"({best_det['n_vehicles']} trucks)")
    print(f"  Best stochastic distance:    {sto_dist:.1f} km "
          f"({best_sto['n_vehicles']} trucks)")
    print(f"  Extra distance for reliability: {extra_pct:+.1f}%")
    print(f"  Max overflow probability:    {best_sto['max_overflow']:.3f} "
          f"(target <= 0.10)")
    print(
        f"\n  The stochastic solution uses "
        f"{'more' if best_sto['n_vehicles'] > best_det['n_vehicles'] else 'the same number of'}"
        f" trucks but guarantees 90% route feasibility"
        f"\n  under daily demand variation, avoiding costly mid-route"
        f" depot returns."
    )
