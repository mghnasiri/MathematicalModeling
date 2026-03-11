"""
Split Delivery Vehicle Routing Problem (SDVRP) — Instance and Solution.

Problem notation: SDVRP

Extends CVRP by allowing each customer to be served by multiple vehicles
(split deliveries). This can reduce the number of routes and total distance
when customer demands exceed vehicle capacity or when splitting is beneficial.

Complexity: NP-hard (generalizes CVRP when no splits are needed).

References:
    Dror, M. & Trudeau, P. (1989). Savings by split delivery routing.
    Transportation Science, 23(2), 141-145.
    https://doi.org/10.1287/trsc.23.2.141

    Archetti, C. & Speranza, M.G. (2012). Vehicle routing problems with
    split deliveries. International Transactions in Operational Research,
    19(1-2), 3-22. https://doi.org/10.1111/j.1475-3995.2011.00811.x
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


@dataclass
class SDVRPInstance:
    """Split Delivery VRP instance.

    Attributes:
        n: Number of customers (nodes 1..n; depot is node 0).
        coords: Coordinates of all nodes, shape (n+1, 2).
        demands: Customer demands, shape (n,). demands[i] for customer i+1.
        capacity: Vehicle capacity.
        name: Optional instance name.
    """

    n: int
    coords: np.ndarray
    demands: np.ndarray
    capacity: float
    name: str = ""

    def __post_init__(self):
        self.coords = np.asarray(self.coords, dtype=float)
        self.demands = np.asarray(self.demands, dtype=float)

    def distance(self, i: int, j: int) -> float:
        """Euclidean distance between nodes i and j."""
        return float(np.linalg.norm(self.coords[i] - self.coords[j]))

    def distance_matrix(self) -> np.ndarray:
        """Full distance matrix, shape (n+1, n+1)."""
        diff = self.coords[:, None, :] - self.coords[None, :, :]
        return np.sqrt((diff ** 2).sum(axis=2))

    @classmethod
    def random(
        cls,
        n: int = 10,
        capacity: float = 50.0,
        demand_range: tuple[int, int] = (5, 30),
        seed: int | None = None,
    ) -> SDVRPInstance:
        rng = np.random.default_rng(seed)
        coords = rng.uniform(0, 100, size=(n + 1, 2))
        demands = rng.integers(demand_range[0], demand_range[1] + 1, size=n).astype(float)
        return cls(n=n, coords=coords, demands=demands, capacity=capacity, name=f"random_{n}")

    def route_distance(self, route: list[int]) -> float:
        """Distance of a route (list of customer indices, depot implicit)."""
        if not route:
            return 0.0
        d = self.distance(0, route[0])
        for i in range(len(route) - 1):
            d += self.distance(route[i], route[i + 1])
        d += self.distance(route[-1], 0)
        return d


@dataclass
class SDVRPSolution:
    """SDVRP solution with split deliveries.

    Attributes:
        routes: List of routes, each route is list of (customer, quantity) pairs.
        total_distance: Total distance of all routes.
    """

    routes: list[list[tuple[int, float]]]
    total_distance: float

    def __repr__(self) -> str:
        return f"SDVRPSolution(routes={len(self.routes)}, dist={self.total_distance:.1f})"


def validate_solution(
    instance: SDVRPInstance, solution: SDVRPSolution
) -> tuple[bool, list[str]]:
    errors = []

    # Check all demands are met
    delivered = np.zeros(instance.n)
    for route in solution.routes:
        load = 0.0
        for cust, qty in route:
            if cust < 1 or cust > instance.n:
                errors.append(f"Invalid customer {cust}")
                continue
            if qty <= -1e-6:
                errors.append(f"Negative delivery {qty:.2f} to customer {cust}")
            delivered[cust - 1] += qty
            load += qty
        if load > instance.capacity + 1e-6:
            errors.append(f"Route overloaded: {load:.2f} > {instance.capacity}")

    for i in range(instance.n):
        if abs(delivered[i] - instance.demands[i]) > 1e-4:
            errors.append(
                f"Customer {i+1}: delivered {delivered[i]:.2f} != demand {instance.demands[i]:.2f}"
            )

    # Recompute distance
    actual_dist = 0.0
    for route in solution.routes:
        nodes = [c for c, _ in route]
        actual_dist += instance.route_distance(nodes)
    if abs(actual_dist - solution.total_distance) > 1e-2:
        errors.append(
            f"Distance mismatch: reported {solution.total_distance:.2f} != actual {actual_dist:.2f}"
        )

    return len(errors) == 0, errors


def small_sdvrp_6() -> SDVRPInstance:
    return SDVRPInstance(
        n=6,
        coords=np.array([
            [50, 50],  # depot
            [20, 70], [80, 70], [20, 30],
            [80, 30], [50, 90], [50, 10],
        ], dtype=float),
        demands=np.array([15, 25, 10, 20, 30, 15], dtype=float),
        capacity=40.0,
        name="small_6",
    )


if __name__ == "__main__":
    inst = small_sdvrp_6()
    print(f"{inst.name}: n={inst.n}, Q={inst.capacity}")
    print(f"  Demands: {inst.demands}")
