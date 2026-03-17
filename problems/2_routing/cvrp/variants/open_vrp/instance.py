"""
Open Vehicle Routing Problem (OVRP) — Instance and Solution.

Problem notation: OVRP

Like CVRP but vehicles do not return to the depot after serving their
last customer. Minimizes total distance traveled across all routes.

Applications: courier services (drivers go home), school bus routing,
newspaper delivery, home healthcare visits.

Complexity: NP-hard (generalizes TSP).

References:
    Sariklis, D. & Powell, S. (2000). A heuristic method for the open
    vehicle routing problem. Journal of the Operational Research Society,
    51(5), 564-573. https://doi.org/10.1057/palgrave.jors.2600924

    Li, F., Golden, B. & Wasil, E. (2007). The open vehicle routing
    problem: Algorithms, large-scale test problems, and computational
    results. Computers & Operations Research, 34(10), 2918-2930.
    https://doi.org/10.1016/j.cor.2005.11.018
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class OVRPInstance:
    """Open Vehicle Routing Problem instance.

    Attributes:
        n: Number of customers (excluding depot).
        num_vehicles: Maximum number of vehicles.
        capacity: Vehicle capacity.
        demands: Customer demands, shape (n,). Index 0 is depot (demand=0).
        distance_matrix: (n+1, n+1) distance matrix (0 = depot).
        coords: Optional (n+1, 2) coordinates.
        name: Optional instance name.
    """

    n: int
    num_vehicles: int
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
        cls,
        n: int = 10,
        num_vehicles: int = 3,
        capacity: float = 100.0,
        demand_range: tuple[float, float] = (5.0, 30.0),
        coord_range: tuple[float, float] = (0.0, 100.0),
        seed: int | None = None,
    ) -> OVRPInstance:
        """Generate a random OVRP instance."""
        rng = np.random.default_rng(seed)
        coords = rng.uniform(coord_range[0], coord_range[1], size=(n + 1, 2))
        demands = np.zeros(n + 1)
        demands[1:] = np.round(rng.uniform(demand_range[0], demand_range[1], size=n))

        dist = np.sqrt(
            np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=2)
        )

        return cls(
            n=n, num_vehicles=num_vehicles, capacity=capacity,
            demands=demands, distance_matrix=dist, coords=coords,
            name=f"random_{n}",
        )

    def route_distance(self, route: list[int]) -> float:
        """Open route distance: depot -> customers (no return)."""
        if not route:
            return 0.0
        dist = self.distance_matrix[0][route[0]]
        for i in range(len(route) - 1):
            dist += self.distance_matrix[route[i]][route[i + 1]]
        return dist

    def route_demand(self, route: list[int]) -> float:
        """Total demand of a route."""
        return sum(self.demands[c] for c in route)


@dataclass
class OVRPSolution:
    """Solution to an OVRP instance.

    Attributes:
        routes: List of routes (each a list of customer indices).
        total_distance: Total distance across all routes.
    """

    routes: list[list[int]]
    total_distance: float

    def __repr__(self) -> str:
        return (
            f"OVRPSolution(distance={self.total_distance:.2f}, "
            f"routes={len(self.routes)})"
        )


def validate_solution(
    instance: OVRPInstance, solution: OVRPSolution
) -> tuple[bool, list[str]]:
    """Validate an OVRP solution."""
    errors = []

    all_customers = []
    for k, route in enumerate(solution.routes):
        for c in route:
            if c <= 0 or c > instance.n:
                errors.append(f"Route {k}: invalid customer {c}")
            all_customers.append(c)

        demand = instance.route_demand(route)
        if demand > instance.capacity + 1e-10:
            errors.append(
                f"Route {k}: demand {demand:.1f} > capacity {instance.capacity:.1f}"
            )

    if sorted(all_customers) != list(range(1, instance.n + 1)):
        errors.append("Not all customers visited exactly once")

    actual_dist = sum(instance.route_distance(r) for r in solution.routes)
    if abs(actual_dist - solution.total_distance) > 1e-4:
        errors.append(
            f"Reported distance {solution.total_distance:.2f} != "
            f"actual {actual_dist:.2f}"
        )

    return len(errors) == 0, errors


# ── Benchmark instances ──────────────────────────────────────────────────────


def small_ovrp_6() -> OVRPInstance:
    """6 customers, 2 vehicles, capacity 50."""
    coords = np.array([
        [50, 50],  # depot
        [20, 80], [80, 80], [90, 40],
        [70, 10], [30, 20], [10, 50],
    ], dtype=float)
    n = 6
    dist = np.sqrt(
        np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=2)
    )
    demands = np.array([0, 15, 20, 10, 25, 15, 10], dtype=float)
    return OVRPInstance(
        n=n, num_vehicles=2, capacity=50,
        demands=demands, distance_matrix=dist, coords=coords,
        name="small_6",
    )


if __name__ == "__main__":
    inst = small_ovrp_6()
    print(f"{inst.name}: {inst.n} customers, {inst.num_vehicles} vehicles")
