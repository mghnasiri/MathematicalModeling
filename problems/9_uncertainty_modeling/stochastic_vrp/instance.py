"""
Stochastic Vehicle Routing Problem (SVRP)

Extends CVRP with stochastic customer demands. The vehicle capacity
constraint must hold with high probability across demand scenarios.

Two main approaches:
    - A priori routes: Design routes before demands are known; if a
      vehicle overflows, it returns to the depot (recourse).
    - Chance-constrained: Routes must be feasible with probability >= 1-alpha.

Notation: SVRP | stochastic demand | min E[cost]

Complexity: NP-hard (generalizes deterministic CVRP).

References:
    - Bertsimas, D.J. (1992). A vehicle routing problem with stochastic
      demand. Oper. Res., 40(3), 574-585.
      https://doi.org/10.1287/opre.40.3.574
    - Gendreau, M., Laporte, G. & Séguin, R. (1996). Stochastic vehicle
      routing. EJOR, 88(1), 3-12.
      https://doi.org/10.1016/0377-2217(95)00050-X
    - Laporte, G., Louveaux, F. & van Hamme, L. (2002). An integer
      L-shaped algorithm for the CVRP with stochastic demands.
      Oper. Res., 50(3), 415-423.
      https://doi.org/10.1287/opre.50.3.415.7751
"""
from __future__ import annotations

import sys
import os
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


@dataclass
class StochasticVRPInstance:
    """Stochastic VRP with demand uncertainty.

    Args:
        n_customers: Number of customers (not counting depot).
        coordinates: (n+1, 2) array — depot at index 0.
        demand_scenarios: (S, n) array — demand for each customer per scenario.
            Depot demand is always 0.
        vehicle_capacity: Capacity Q of each vehicle.
        n_vehicles: Maximum number of vehicles available.
        alpha: Maximum allowed route overflow probability.
        probabilities: Scenario probabilities (S,). Uniform if None.
    """
    n_customers: int
    coordinates: np.ndarray
    demand_scenarios: np.ndarray
    vehicle_capacity: float
    n_vehicles: int
    alpha: float = 0.1
    probabilities: np.ndarray | None = None

    def __post_init__(self):
        self.coordinates = np.asarray(self.coordinates, dtype=float)
        self.demand_scenarios = np.asarray(self.demand_scenarios, dtype=float)
        if self.probabilities is None:
            S = self.demand_scenarios.shape[0]
            self.probabilities = np.full(S, 1.0 / S)
        else:
            self.probabilities = np.asarray(self.probabilities, dtype=float)

    @property
    def n_scenarios(self) -> int:
        return self.demand_scenarios.shape[0]

    @property
    def mean_demands(self) -> np.ndarray:
        """Expected demand per customer."""
        return np.dot(self.probabilities, self.demand_scenarios)

    def distance(self, i: int, j: int) -> float:
        """Euclidean distance between nodes i and j."""
        diff = self.coordinates[i] - self.coordinates[j]
        return float(np.sqrt(np.dot(diff, diff)))

    def distance_matrix(self) -> np.ndarray:
        """Full distance matrix (n+1) x (n+1)."""
        n = len(self.coordinates)
        dm = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = self.distance(i, j)
                dm[i, j] = d
                dm[j, i] = d
        return dm

    def route_distance(self, route: list[int]) -> float:
        """Total distance of a route (depot -> customers -> depot).

        Args:
            route: List of customer indices (1-indexed, no depot).

        Returns:
            Total Euclidean distance.
        """
        if not route:
            return 0.0
        total = self.distance(0, route[0])
        for i in range(len(route) - 1):
            total += self.distance(route[i], route[i + 1])
        total += self.distance(route[-1], 0)
        return total

    def route_overflow_probability(self, route: list[int]) -> float:
        """Probability that total demand on route exceeds capacity.

        Args:
            route: Customer indices (1-indexed).

        Returns:
            P(sum of demands > Q).
        """
        if not route:
            return 0.0
        # Customer indices in demand_scenarios are 0-indexed (customer 1 = index 0)
        cust_indices = [c - 1 for c in route]
        loads = self.demand_scenarios[:, cust_indices].sum(axis=1)
        overflow = loads > self.vehicle_capacity + 1e-9
        return float(np.dot(overflow, self.probabilities))

    def solution_total_distance(self, routes: list[list[int]]) -> float:
        """Total distance across all routes."""
        return sum(self.route_distance(r) for r in routes)

    def expected_recourse_cost(self, routes: list[list[int]],
                                return_cost_factor: float = 2.0) -> float:
        """Expected extra cost from route failures (return trips to depot).

        When a route overflows, the vehicle must return to depot mid-route.
        Approximated as: for each scenario where overflow occurs, add
        return_cost_factor * average_distance as penalty.

        Args:
            routes: List of routes.
            return_cost_factor: Multiplier for penalty.

        Returns:
            Expected recourse cost.
        """
        total_recourse = 0.0
        for route in routes:
            if not route:
                continue
            avg_dist = self.route_distance(route) / max(len(route), 1)
            overflow_prob = self.route_overflow_probability(route)
            total_recourse += overflow_prob * return_cost_factor * avg_dist
        return total_recourse

    @classmethod
    def random(cls, n_customers: int = 10, n_scenarios: int = 20,
               n_vehicles: int = 3, seed: int = 42) -> StochasticVRPInstance:
        """Generate a random stochastic VRP instance."""
        rng = np.random.default_rng(seed)
        coordinates = rng.uniform(0, 100, (n_customers + 1, 2))
        mean_demands = rng.uniform(5, 25, n_customers)
        demand_scenarios = np.array([
            np.maximum(1, mean_demands + rng.normal(0, 5, n_customers))
            for _ in range(n_scenarios)
        ])
        capacity = float(mean_demands.sum() / n_vehicles * 1.3)
        return cls(
            n_customers=n_customers,
            coordinates=coordinates,
            demand_scenarios=demand_scenarios,
            vehicle_capacity=capacity,
            n_vehicles=n_vehicles,
        )


@dataclass
class StochasticVRPSolution:
    """Solution to the stochastic VRP.

    Args:
        routes: List of routes (each is a list of customer indices, 1-indexed).
        total_distance: Sum of route distances.
        expected_total_cost: Distance + expected recourse.
        max_overflow_prob: Max overflow probability across routes.
        n_routes: Number of routes used.
    """
    routes: list[list[int]]
    total_distance: float
    expected_total_cost: float
    max_overflow_prob: float
    n_routes: int

    def __repr__(self) -> str:
        return (f"StochasticVRPSolution(routes={self.n_routes}, "
                f"dist={self.total_distance:.1f}, "
                f"E[cost]={self.expected_total_cost:.1f}, "
                f"max_overflow={self.max_overflow_prob:.3f})")
