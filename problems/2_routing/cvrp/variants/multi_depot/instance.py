"""
Multi-Depot Vehicle Routing Problem (MDVRP) — Instance and Solution.

Problem notation: MDVRP

Multiple depots, each with a fleet of vehicles. Each customer is assigned
to a depot and served by a vehicle from that depot. Minimize total distance.

Complexity: NP-hard (generalizes CVRP).

References:
    Cordeau, J.-F., Gendreau, M. & Laporte, G. (1997). A tabu search
    heuristic for periodic and multi-depot vehicle routing problems.
    Networks, 30(2), 105-119.
    https://doi.org/10.1002/(SICI)1097-0037(199709)30:2<105::AID-NET5>3.0.CO;2-G
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class MDVRPInstance:
    """Multi-Depot VRP instance.

    Nodes: 0..num_depots-1 are depots, num_depots..num_depots+n-1 are customers.

    Attributes:
        num_depots: Number of depots.
        n: Number of customers.
        capacity: Vehicle capacity.
        demands: Customer demands, shape (n,).
        distance_matrix: Full distance matrix, shape (num_depots+n, num_depots+n).
        coords: Optional coordinates.
        name: Optional instance name.
    """

    num_depots: int
    n: int
    capacity: float
    demands: np.ndarray
    distance_matrix: np.ndarray
    coords: np.ndarray | None = None
    name: str = ""

    def __post_init__(self):
        self.demands = np.asarray(self.demands, dtype=float)
        self.distance_matrix = np.asarray(self.distance_matrix, dtype=float)

    @classmethod
    def random(
        cls, num_depots: int = 2, n: int = 10, capacity: float = 80.0,
        demand_range: tuple[float, float] = (5.0, 25.0),
        seed: int | None = None,
    ) -> MDVRPInstance:
        rng = np.random.default_rng(seed)
        total = num_depots + n
        coords = rng.uniform(0, 100, size=(total, 2))
        demands = np.round(rng.uniform(demand_range[0], demand_range[1], size=n))
        dist = np.sqrt(np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=2))
        return cls(num_depots=num_depots, n=n, capacity=capacity,
                   demands=demands, distance_matrix=dist, coords=coords,
                   name=f"random_{num_depots}_{n}")

    def customer_idx(self, c: int) -> int:
        """Customer c (0-indexed) maps to node num_depots + c."""
        return self.num_depots + c

    def route_distance(self, depot: int, route: list[int]) -> float:
        """Route distance: depot -> customers -> depot."""
        if not route:
            return 0.0
        d = self.distance_matrix[depot][route[0]]
        for i in range(len(route) - 1):
            d += self.distance_matrix[route[i]][route[i + 1]]
        d += self.distance_matrix[route[-1]][depot]
        return d

    def route_demand(self, route: list[int]) -> float:
        """Total demand of route customers."""
        return sum(self.demands[c - self.num_depots] for c in route)


@dataclass
class MDVRPSolution:
    """Solution to an MDVRP instance.

    Attributes:
        depot_routes: depot_routes[d] = list of routes from depot d.
        total_distance: Total distance across all routes.
    """

    depot_routes: list[list[list[int]]]
    total_distance: float

    def __repr__(self) -> str:
        n_routes = sum(len(rs) for rs in self.depot_routes)
        return f"MDVRPSolution(distance={self.total_distance:.2f}, routes={n_routes})"


def validate_solution(
    instance: MDVRPInstance, solution: MDVRPSolution
) -> tuple[bool, list[str]]:
    errors = []
    if len(solution.depot_routes) != instance.num_depots:
        errors.append(f"Expected {instance.num_depots} depot route lists")
        return False, errors

    all_customers = []
    actual_dist = 0.0
    for d in range(instance.num_depots):
        for k, route in enumerate(solution.depot_routes[d]):
            demand = instance.route_demand(route)
            if demand > instance.capacity + 1e-10:
                errors.append(f"Depot {d} route {k}: demand {demand:.1f} > cap")
            all_customers.extend(route)
            actual_dist += instance.route_distance(d, route)

    expected = sorted(range(instance.num_depots, instance.num_depots + instance.n))
    if sorted(all_customers) != expected:
        errors.append("Not all customers visited exactly once")

    if abs(actual_dist - solution.total_distance) > 1e-4:
        errors.append(f"Reported dist {solution.total_distance:.2f} != actual {actual_dist:.2f}")

    return len(errors) == 0, errors


def small_mdvrp_2_6() -> MDVRPInstance:
    coords = np.array([
        [20, 50], [80, 50],  # 2 depots
        [10, 80], [30, 90], [15, 20], [40, 30],  # customers near depot 0
        [70, 80], [90, 20],  # customers near depot 1
    ], dtype=float)
    dist = np.sqrt(np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=2))
    demands = np.array([15, 20, 10, 25, 15, 20], dtype=float)
    return MDVRPInstance(
        num_depots=2, n=6, capacity=50,
        demands=demands, distance_matrix=dist, coords=coords,
        name="small_2_6",
    )


if __name__ == "__main__":
    inst = small_mdvrp_2_6()
    print(f"{inst.name}: {inst.num_depots} depots, {inst.n} customers")
