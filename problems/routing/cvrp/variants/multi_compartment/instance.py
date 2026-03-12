"""
Multi-Compartment VRP (MCVRP) — Instance and Solution.

Vehicles have multiple compartments, each with its own capacity. Each
customer demands specific product types, and products must be carried
in their designated compartments. Minimize total travel distance.

Complexity: NP-hard (generalizes CVRP).

References:
    Derigs, U., Gottlieb, J. & Kalkoff, J. (2011). Vehicle routing with
    compartments: Applications, modelling and heuristics. OR Spectrum, 33(4),
    885-914. https://doi.org/10.1007/s00291-009-0175-6

    Chajakis, E.D. & Guignard, M. (2003). Scheduling deliveries in vehicles
    with multiple compartments. Journal of Global Optimization, 26(1), 43-78.
    https://doi.org/10.1023/A:1023067016014
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class MCVRPInstance:
    """Multi-Compartment VRP instance.

    Attributes:
        n: Number of customers (nodes 1..n, depot is 0).
        num_compartments: Number of vehicle compartments (product types).
        coords: Node coordinates, shape (n + 1, 2).
        demands: Demand matrix, shape (n, num_compartments).
            demands[i][k] = demand of customer i+1 for product type k.
        compartment_capacities: Capacity per compartment, shape (num_compartments,).
        name: Optional instance name.
    """

    n: int
    num_compartments: int
    coords: np.ndarray
    demands: np.ndarray
    compartment_capacities: np.ndarray
    name: str = ""

    def __post_init__(self):
        self.coords = np.asarray(self.coords, dtype=float)
        self.demands = np.asarray(self.demands, dtype=float)
        self.compartment_capacities = np.asarray(self.compartment_capacities, dtype=float)

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

    def route_loads(self, route: list[int]) -> np.ndarray:
        """Compute per-compartment loads for a route."""
        loads = np.zeros(self.num_compartments)
        for c in route:
            loads += self.demands[c - 1]
        return loads

    @classmethod
    def random(
        cls,
        n: int = 10,
        num_compartments: int = 3,
        seed: int | None = None,
    ) -> MCVRPInstance:
        rng = np.random.default_rng(seed)
        coords = rng.uniform(0, 100, size=(n + 1, 2))
        coords[0] = [50, 50]
        demands = rng.integers(2, 12, size=(n, num_compartments)).astype(float)
        # Each customer only needs 1-2 product types
        for i in range(n):
            zero_mask = rng.random(num_compartments) < 0.4
            demands[i][zero_mask] = 0
        cap_per = np.ceil(demands.sum(axis=0) / 3 + demands.max(axis=0))
        return cls(n=n, num_compartments=num_compartments, coords=coords,
                   demands=demands, compartment_capacities=cap_per,
                   name=f"random_mcvrp_{n}c_{num_compartments}k")


@dataclass
class MCVRPSolution:
    """MCVRP solution.

    Attributes:
        routes: List of routes (each a list of customer indices).
        total_distance: Total travel distance.
    """

    routes: list[list[int]]
    total_distance: float

    def __repr__(self) -> str:
        return (f"MCVRPSolution(routes={len(self.routes)}, "
                f"dist={self.total_distance:.1f})")


def validate_solution(
    instance: MCVRPInstance, solution: MCVRPSolution
) -> tuple[bool, list[str]]:
    errors = []

    # All customers visited exactly once
    visited = {}
    for route in solution.routes:
        for c in route:
            visited[c] = visited.get(c, 0) + 1

    for c in range(1, instance.n + 1):
        count = visited.get(c, 0)
        if count == 0:
            errors.append(f"Customer {c} not visited")
        elif count > 1:
            errors.append(f"Customer {c} visited {count} times")

    # Compartment capacities
    for idx, route in enumerate(solution.routes):
        loads = instance.route_loads(route)
        for k in range(instance.num_compartments):
            if loads[k] > instance.compartment_capacities[k] + 1e-6:
                errors.append(
                    f"Route {idx}: compartment {k} load {loads[k]:.1f} > "
                    f"capacity {instance.compartment_capacities[k]:.1f}")

    # Distance check
    actual = sum(instance.route_distance(r) for r in solution.routes)
    if abs(actual - solution.total_distance) > 1e-2:
        errors.append(f"Distance: {solution.total_distance:.2f} != {actual:.2f}")

    return len(errors) == 0, errors


def small_mcvrp_6() -> MCVRPInstance:
    coords = np.array([
        [50, 50], [20, 70], [30, 30], [70, 80],
        [80, 40], [60, 20], [40, 60],
    ], dtype=float)
    demands = np.array([
        [8, 0, 5],   # c1
        [0, 10, 3],  # c2
        [6, 4, 0],   # c3
        [5, 0, 7],   # c4
        [0, 8, 4],   # c5
        [7, 3, 0],   # c6
    ], dtype=float)
    return MCVRPInstance(
        n=6, num_compartments=3, coords=coords, demands=demands,
        compartment_capacities=np.array([20.0, 18.0, 15.0]),
        name="small_mcvrp_6",
    )


if __name__ == "__main__":
    inst = small_mcvrp_6()
    print(f"{inst.name}: n={inst.n}, compartments={inst.num_compartments}")
