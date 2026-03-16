"""
Farm-to-Market Fresh Produce Delivery Routing Problem

Domain: Agricultural cooperative logistics / Fresh produce distribution
Notation: CVRP_agri | capacity, perishability | min distance
          SVRP_agri | capacity, stochastic demand | min E[cost]

A cooperative of farms distributes fresh produce to delivery points
(farmers markets, restaurants, grocery stores, food banks) using
refrigerated trucks. Daily demand varies, making stochastic routing
essential for reliable service.

Deterministic model: CVRP with known demands.
Stochastic model: Stochastic VRP with chance-constrained routes.

Complexity: NP-hard (generalizes both TSP and Bin Packing).

References:
    Clarke, G. & Wright, J.W. (1964). Scheduling of vehicles from a
    central depot to a number of delivery points. Operations Research,
    12(4), 568-581. https://doi.org/10.1287/opre.12.4.568

    Bertsimas, D.J. (1992). A vehicle routing problem with stochastic
    demand. Operations Research, 40(3), 574-585.
    https://doi.org/10.1287/opre.40.3.574

    Bosona, T. & Gebresenbet, G. (2011). Cluster building and logistics
    network integration of local food supply chain. Biosystems
    Engineering, 108(4), 293-302.
    https://doi.org/10.1016/j.biosystemseng.2011.01.001
"""
from __future__ import annotations

import sys
import os
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


POINT_TYPE_LABELS = {
    "farmers_market": "Farmers Market",
    "restaurant": "Restaurant",
    "grocery": "Grocery Store",
    "food_bank": "Food Bank",
}


@dataclass
class DeliveryPoint:
    """A delivery point in the farm-to-market network.

    Args:
        name: Name of the delivery point.
        point_type: Type (farmers_market, restaurant, grocery, food_bank).
        base_demand_kg: Base daily demand in kg.
    """
    name: str
    point_type: str
    base_demand_kg: float


@dataclass
class FarmDeliveryInstance:
    """Farm-to-market delivery routing instance.

    Args:
        delivery_points: List of delivery point profiles.
        truck_capacity_kg: Capacity per refrigerated truck (kg).
        fuel_cost_per_km: Fuel cost ($/km) for refrigerated trucks.
        n_demand_scenarios: Number of stochastic demand scenarios.
        alpha: Reliability level for chance constraints (P(overflow) <= alpha).
        name: Instance name.
    """
    delivery_points: list[DeliveryPoint]
    truck_capacity_kg: float = 2000.0
    fuel_cost_per_km: float = 0.85
    n_demand_scenarios: int = 50
    alpha: float = 0.10
    name: str = "farm_delivery"

    @property
    def n_customers(self) -> int:
        return len(self.delivery_points)

    def generate_coordinates(self, seed: int = 42) -> np.ndarray:
        """Generate delivery point coordinates in a 50x50 km area.

        Depot is at (25, 25). Delivery points are clustered by type.

        Args:
            seed: Random seed.

        Returns:
            Array of shape (n_customers + 1, 2) with depot at index 0.
        """
        rng = np.random.default_rng(seed)
        depot = np.array([25.0, 25.0])

        location_profiles = {
            "farmers_market": {"center": (20.0, 30.0), "spread": 12.0},
            "restaurant": {"center": (28.0, 28.0), "spread": 6.0},
            "grocery": {"center": (25.0, 25.0), "spread": 15.0},
            "food_bank": {"center": (22.0, 20.0), "spread": 10.0},
        }

        coords = [depot]
        for dp in self.delivery_points:
            profile = location_profiles[dp.point_type]
            cx, cy = profile["center"]
            spread = profile["spread"]
            x = np.clip(cx + rng.normal(0, spread), 0, 50)
            y = np.clip(cy + rng.normal(0, spread), 0, 50)
            coords.append(np.array([x, y]))

        return np.array(coords)

    def generate_distance_matrix(self, coords: np.ndarray) -> np.ndarray:
        """Compute Euclidean distance matrix from coordinates.

        Args:
            coords: Array of shape (n+1, 2).

        Returns:
            Distance matrix of shape (n+1, n+1).
        """
        n = len(coords)
        dist = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dist[i][j] = np.sqrt(np.sum((coords[i] - coords[j]) ** 2))
        return dist

    def generate_demand_scenarios(self, seed: int = 42) -> np.ndarray:
        """Generate stochastic demand scenarios.

        Args:
            seed: Random seed.

        Returns:
            Array of shape (n_scenarios, n_customers).
        """
        rng = np.random.default_rng(seed)
        variation = {
            "farmers_market": 0.20,
            "restaurant": 0.25,
            "grocery": 0.15,
            "food_bank": 0.20,
        }
        scenarios = np.zeros((self.n_demand_scenarios, self.n_customers))
        for s in range(self.n_demand_scenarios):
            for c, dp in enumerate(self.delivery_points):
                var = variation[dp.point_type]
                noisy = dp.base_demand_kg * (1.0 + rng.uniform(-var, var))
                scenarios[s, c] = max(10.0, noisy)
        return scenarios

    @classmethod
    def quebec_cooperative(cls, n_demand_scenarios: int = 50) -> FarmDeliveryInstance:
        """Create the Quebec cooperative benchmark instance.

        15 delivery points across Quebec City region with realistic
        demand profiles for each establishment type.

        Args:
            n_demand_scenarios: Number of demand scenarios.

        Returns:
            FarmDeliveryInstance with 15 delivery points.
        """
        points = [
            DeliveryPoint("Marche du Vieux-Port", "farmers_market", 450),
            DeliveryPoint("Marche de Sainte-Foy", "farmers_market", 380),
            DeliveryPoint("Marche de Limoilou", "farmers_market", 320),
            DeliveryPoint("Marche de Levis", "farmers_market", 500),
            DeliveryPoint("Bistro Le Clocher Penche", "restaurant", 80),
            DeliveryPoint("Restaurant Panache", "restaurant", 120),
            DeliveryPoint("Chez Boulay", "restaurant", 95),
            DeliveryPoint("Le Saint-Amour", "restaurant", 65),
            DeliveryPoint("Cafe du Monde", "restaurant", 150),
            DeliveryPoint("Epicerie Bio Quebec", "grocery", 350),
            DeliveryPoint("Marche Tradition Beauport", "grocery", 280),
            DeliveryPoint("IGA Extra Charlesbourg", "grocery", 400),
            DeliveryPoint("Metro Plus Sillery", "grocery", 220),
            DeliveryPoint("Moisson Quebec", "food_bank", 250),
            DeliveryPoint("Banque Alimentaire Levis", "food_bank", 150),
        ]
        return cls(
            delivery_points=points,
            truck_capacity_kg=2000.0,
            fuel_cost_per_km=0.85,
            n_demand_scenarios=n_demand_scenarios,
            alpha=0.10,
            name="quebec_cooperative",
        )

    @classmethod
    def random(cls, n_customers: int = 10, seed: int = 42) -> FarmDeliveryInstance:
        """Generate a random farm delivery instance.

        Args:
            n_customers: Number of delivery points.
            seed: Random seed.

        Returns:
            Random FarmDeliveryInstance.
        """
        rng = np.random.default_rng(seed)
        types = ["farmers_market", "restaurant", "grocery", "food_bank"]
        points = []
        for i in range(n_customers):
            pt = rng.choice(types)
            demand_ranges = {
                "farmers_market": (200, 500),
                "restaurant": (50, 200),
                "grocery": (150, 400),
                "food_bank": (80, 250),
            }
            lo, hi = demand_ranges[pt]
            points.append(DeliveryPoint(
                name=f"Point_{i+1}",
                point_type=pt,
                base_demand_kg=float(rng.integers(lo, hi)),
            ))
        return cls(
            delivery_points=points,
            n_demand_scenarios=50,
            name=f"random_{n_customers}pts",
        )


@dataclass
class FarmDeliverySolution:
    """Solution to the farm delivery routing problem.

    Args:
        routes: List of routes, each a list of customer indices (1-based).
        total_distance: Total route distance (km).
        n_vehicles: Number of vehicles used.
        method: Algorithm used.
        expected_cost: Expected cost including recourse (stochastic only).
        max_overflow_prob: Maximum route overflow probability (stochastic only).
    """
    routes: list[list[int]]
    total_distance: float
    n_vehicles: int
    method: str
    expected_cost: float | None = None
    max_overflow_prob: float | None = None

    def __repr__(self) -> str:
        return (f"FarmDeliverySolution(method={self.method}, "
                f"distance={self.total_distance:.1f} km, "
                f"vehicles={self.n_vehicles})")


if __name__ == "__main__":
    inst = FarmDeliveryInstance.quebec_cooperative()
    print(f"Quebec cooperative: {inst.n_customers} delivery points")
    for dp in inst.delivery_points:
        print(f"  {dp.name}: {dp.point_type}, {dp.base_demand_kg} kg")
