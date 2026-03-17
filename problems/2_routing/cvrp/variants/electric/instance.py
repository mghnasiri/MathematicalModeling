"""
Electric Vehicle Routing Problem (EVRP) — Instance and Solution.

Extends CVRP with battery-constrained electric vehicles. Vehicles have
limited battery range and must visit charging stations to recharge.
Minimize total distance while respecting capacity and energy constraints.

Complexity: NP-hard (generalizes CVRP).

References:
    Schneider, M., Stenger, A. & Goeke, D. (2014). The electric vehicle
    routing problem with time windows and recharging stations. Transportation
    Science, 48(4), 500-520. https://doi.org/10.1287/trsc.2013.0490

    Erdogan, S. & Miller-Hooks, E. (2012). A green vehicle routing problem.
    Transportation Research Part E, 48(1), 100-114.
    https://doi.org/10.1016/j.tre.2011.08.001
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


@dataclass
class EVRPInstance:
    """Electric Vehicle Routing Problem instance.

    Attributes:
        n: Number of customers (nodes 1..n).
        coords: Coordinates of all nodes, shape (n + 1 + num_stations, 2).
            Node 0 is the depot, nodes 1..n are customers,
            nodes n+1..n+num_stations are charging stations.
        demands: Demand at each customer, shape (n,). demands[i] for customer i+1.
        vehicle_capacity: Vehicle load capacity.
        battery_capacity: Maximum battery level (energy units).
        energy_rate: Energy consumed per unit distance.
        num_stations: Number of charging stations.
        name: Optional instance name.
    """

    n: int
    coords: np.ndarray
    demands: np.ndarray
    vehicle_capacity: float
    battery_capacity: float
    energy_rate: float
    num_stations: int
    name: str = ""

    def __post_init__(self):
        self.coords = np.asarray(self.coords, dtype=float)
        self.demands = np.asarray(self.demands, dtype=float)

    @property
    def total_nodes(self) -> int:
        """Total number of nodes (depot + customers + stations)."""
        return 1 + self.n + self.num_stations

    @property
    def station_nodes(self) -> list[int]:
        """Indices of charging station nodes."""
        return list(range(self.n + 1, self.n + 1 + self.num_stations))

    @property
    def customer_nodes(self) -> list[int]:
        """Indices of customer nodes."""
        return list(range(1, self.n + 1))

    def dist(self, i: int, j: int) -> float:
        """Euclidean distance between nodes i and j."""
        return float(np.linalg.norm(self.coords[i] - self.coords[j]))

    def energy_cost(self, i: int, j: int) -> float:
        """Energy consumed traveling from node i to node j."""
        return self.energy_rate * self.dist(i, j)

    def route_distance(self, route: list[int]) -> float:
        """Total distance of a route (depot -> ... -> depot)."""
        if not route:
            return 0.0
        d = self.dist(0, route[0])
        for k in range(len(route) - 1):
            d += self.dist(route[k], route[k + 1])
        d += self.dist(route[-1], 0)
        return d

    def route_feasible(self, route: list[int]) -> tuple[bool, str]:
        """Check if a route is capacity- and energy-feasible.

        Returns:
            (feasible, error_message)
        """
        # Check capacity
        load = sum(self.demands[c - 1] for c in route if 1 <= c <= self.n)
        if load > self.vehicle_capacity + 1e-6:
            return False, f"Load {load:.1f} > capacity {self.vehicle_capacity:.1f}"

        # Check energy
        battery = self.battery_capacity
        prev = 0
        for node in route:
            energy = self.energy_cost(prev, node)
            battery -= energy
            if battery < -1e-6:
                return False, f"Battery depleted at node {node}"
            if node in self.station_nodes:
                battery = self.battery_capacity  # Full recharge
            prev = node
        # Return to depot
        battery -= self.energy_cost(prev, 0)
        if battery < -1e-6:
            return False, "Battery depleted returning to depot"

        return True, ""

    @classmethod
    def random(
        cls,
        n: int = 8,
        num_stations: int = 2,
        vehicle_capacity: float = 50.0,
        battery_capacity: float = 80.0,
        energy_rate: float = 1.0,
        seed: int | None = None,
    ) -> EVRPInstance:
        rng = np.random.default_rng(seed)
        total = 1 + n + num_stations
        coords = rng.uniform(0, 100, size=(total, 2))
        coords[0] = [50, 50]  # Depot at center
        demands = rng.integers(5, 20, size=n).astype(float)
        return cls(
            n=n, coords=coords, demands=demands,
            vehicle_capacity=vehicle_capacity,
            battery_capacity=battery_capacity,
            energy_rate=energy_rate,
            num_stations=num_stations,
            name=f"random_evrp_{n}c_{num_stations}s",
        )


@dataclass
class EVRPSolution:
    """EVRP solution.

    Attributes:
        routes: List of routes, each a list of node indices.
        total_distance: Total travel distance.
    """

    routes: list[list[int]]
    total_distance: float

    def __repr__(self) -> str:
        return (f"EVRPSolution(routes={len(self.routes)}, "
                f"dist={self.total_distance:.1f})")


def validate_solution(
    instance: EVRPInstance, solution: EVRPSolution
) -> tuple[bool, list[str]]:
    errors = []

    # Check all customers visited exactly once
    visited = {}
    for route in solution.routes:
        for node in route:
            if 1 <= node <= instance.n:
                visited[node] = visited.get(node, 0) + 1

    for c in range(1, instance.n + 1):
        count = visited.get(c, 0)
        if count == 0:
            errors.append(f"Customer {c} not visited")
        elif count > 1:
            errors.append(f"Customer {c} visited {count} times")

    # Check each route feasibility
    for idx, route in enumerate(solution.routes):
        feasible, msg = instance.route_feasible(route)
        if not feasible:
            errors.append(f"Route {idx}: {msg}")

    # Check total distance
    actual = sum(instance.route_distance(r) for r in solution.routes)
    if abs(actual - solution.total_distance) > 1e-2:
        errors.append(f"Distance: {solution.total_distance:.2f} != {actual:.2f}")

    return len(errors) == 0, errors


def small_evrp_6() -> EVRPInstance:
    """Small EVRP instance with 6 customers and 2 charging stations."""
    coords = np.array([
        [50, 50],   # depot
        [20, 70],   # c1
        [30, 30],   # c2
        [70, 80],   # c3
        [80, 40],   # c4
        [60, 20],   # c5
        [40, 60],   # c6
        [45, 45],   # station 1 (near depot)
        [25, 55],   # station 2
    ], dtype=float)
    demands = np.array([10, 15, 8, 12, 10, 7], dtype=float)
    return EVRPInstance(
        n=6, coords=coords, demands=demands,
        vehicle_capacity=35.0, battery_capacity=80.0,
        energy_rate=1.0, num_stations=2,
        name="small_evrp_6",
    )


if __name__ == "__main__":
    inst = small_evrp_6()
    print(f"{inst.name}: n={inst.n}, stations={inst.num_stations}")
