"""
Multi-Trip VRP (MTVRP) — Instance and Solution.

Extends CVRP by allowing vehicles to make multiple trips. After completing
a route, a vehicle returns to the depot and can start a new trip.
Minimize total distance with a limited number of vehicles.

Complexity: NP-hard (generalizes CVRP).

References:
    Taillard, E.D., Laporte, G. & Gendreau, M. (1996). Vehicle routeing with
    multiple use of vehicles. Journal of the Operational Research Society,
    47(8), 1065-1070. https://doi.org/10.1057/jors.1996.133

    Olivera, A. & Viera, O. (2007). Adaptive memory programming for the
    vehicle routing problem with multiple trips. Computers & Operations
    Research, 34(1), 28-47. https://doi.org/10.1016/j.cor.2005.02.044
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class MTVRPInstance:
    """Multi-Trip VRP instance.

    Attributes:
        n: Number of customers.
        num_vehicles: Number of available vehicles.
        coords: Node coordinates, shape (n + 1, 2). Node 0 is depot.
        demands: Customer demands, shape (n,).
        vehicle_capacity: Vehicle capacity.
        name: Optional instance name.
    """

    n: int
    num_vehicles: int
    coords: np.ndarray
    demands: np.ndarray
    vehicle_capacity: float
    name: str = ""

    def __post_init__(self):
        self.coords = np.asarray(self.coords, dtype=float)
        self.demands = np.asarray(self.demands, dtype=float)

    def dist(self, i: int, j: int) -> float:
        return float(np.linalg.norm(self.coords[i] - self.coords[j]))

    def route_distance(self, route: list[int]) -> float:
        if not route:
            return 0.0
        d = self.dist(0, route[0])
        for k in range(len(route) - 1):
            d += self.dist(route[k], route[k + 1])
        d += self.dist(route[-1], 0)
        return d

    def route_load(self, route: list[int]) -> float:
        return sum(self.demands[c - 1] for c in route)

    @classmethod
    def random(
        cls,
        n: int = 12,
        num_vehicles: int = 2,
        vehicle_capacity: float = 40.0,
        seed: int | None = None,
    ) -> MTVRPInstance:
        rng = np.random.default_rng(seed)
        coords = rng.uniform(0, 100, size=(n + 1, 2))
        coords[0] = [50, 50]
        demands = rng.integers(5, 15, size=n).astype(float)
        return cls(n=n, num_vehicles=num_vehicles, coords=coords,
                   demands=demands, vehicle_capacity=vehicle_capacity,
                   name=f"random_mtvrp_{n}_{num_vehicles}v")


@dataclass
class MTVRPSolution:
    """Multi-Trip VRP solution.

    Attributes:
        vehicle_trips: vehicle_trips[v] = list of routes for vehicle v.
        total_distance: Total travel distance.
    """

    vehicle_trips: list[list[list[int]]]
    total_distance: float

    def __repr__(self) -> str:
        n_trips = sum(len(trips) for trips in self.vehicle_trips)
        return f"MTVRPSolution(vehicles={len(self.vehicle_trips)}, trips={n_trips}, dist={self.total_distance:.1f})"


def validate_solution(
    instance: MTVRPInstance, solution: MTVRPSolution
) -> tuple[bool, list[str]]:
    errors = []

    if len(solution.vehicle_trips) > instance.num_vehicles:
        errors.append(f"Too many vehicles: {len(solution.vehicle_trips)} > {instance.num_vehicles}")

    # All customers visited exactly once
    visited = {}
    for v_trips in solution.vehicle_trips:
        for route in v_trips:
            for c in route:
                visited[c] = visited.get(c, 0) + 1

    for c in range(1, instance.n + 1):
        count = visited.get(c, 0)
        if count == 0:
            errors.append(f"Customer {c} not visited")
        elif count > 1:
            errors.append(f"Customer {c} visited {count} times")

    # Route capacity
    for v, trips in enumerate(solution.vehicle_trips):
        for t, route in enumerate(trips):
            load = instance.route_load(route)
            if load > instance.vehicle_capacity + 1e-6:
                errors.append(f"Vehicle {v} trip {t}: load {load:.1f} > cap {instance.vehicle_capacity:.1f}")

    # Distance
    actual = 0.0
    for trips in solution.vehicle_trips:
        for route in trips:
            actual += instance.route_distance(route)
    if abs(actual - solution.total_distance) > 1e-2:
        errors.append(f"Distance: {solution.total_distance:.2f} != {actual:.2f}")

    return len(errors) == 0, errors


def small_mtvrp_8() -> MTVRPInstance:
    coords = np.array([
        [50, 50],  # depot
        [20, 70], [30, 30], [70, 80], [80, 40],
        [60, 20], [40, 60], [25, 50], [65, 55],
    ], dtype=float)
    demands = np.array([10, 12, 8, 15, 10, 7, 11, 9], dtype=float)
    return MTVRPInstance(
        n=8, num_vehicles=2, coords=coords, demands=demands,
        vehicle_capacity=30.0, name="small_mtvrp_8",
    )


if __name__ == "__main__":
    inst = small_mtvrp_8()
    print(f"{inst.name}: n={inst.n}, vehicles={inst.num_vehicles}")
