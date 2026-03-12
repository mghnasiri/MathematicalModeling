"""
VRP with Backhauls (VRPB) — Instance and Solution.

Extends CVRP with two customer types: linehaul (delivery) and backhaul
(pickup). All linehaul customers on a route must be served before any
backhaul customers. Minimize total travel distance.

Complexity: NP-hard (generalizes CVRP).

References:
    Toth, P. & Vigo, D. (1999). A heuristic algorithm for the symmetric and
    asymmetric vehicle routing problem with backhauls. European Journal of
    Operational Research, 113(3), 528-543.
    https://doi.org/10.1016/S0377-2217(98)00012-6

    Goetschalckx, M. & Jacobs-Blecha, C. (1989). The vehicle routing problem
    with backhauls. European Journal of Operational Research, 42(1), 39-51.
    https://doi.org/10.1016/0377-2217(89)90057-X
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class VRPBInstance:
    """VRP with Backhauls instance.

    Attributes:
        n_linehaul: Number of linehaul (delivery) customers.
        n_backhaul: Number of backhaul (pickup) customers.
        coords: Node coordinates, shape (1 + n_linehaul + n_backhaul, 2).
            Node 0 is depot, 1..n_linehaul are linehaul,
            n_linehaul+1..n_linehaul+n_backhaul are backhaul.
        demands: Demand at each customer (delivery or pickup), shape (n_total,).
        vehicle_capacity: Vehicle capacity.
        name: Optional instance name.
    """

    n_linehaul: int
    n_backhaul: int
    coords: np.ndarray
    demands: np.ndarray
    vehicle_capacity: float
    name: str = ""

    def __post_init__(self):
        self.coords = np.asarray(self.coords, dtype=float)
        self.demands = np.asarray(self.demands, dtype=float)

    @property
    def n_total(self) -> int:
        return self.n_linehaul + self.n_backhaul

    @property
    def linehaul_nodes(self) -> list[int]:
        return list(range(1, self.n_linehaul + 1))

    @property
    def backhaul_nodes(self) -> list[int]:
        return list(range(self.n_linehaul + 1, self.n_total + 1))

    def is_linehaul(self, node: int) -> bool:
        return 1 <= node <= self.n_linehaul

    def is_backhaul(self, node: int) -> bool:
        return self.n_linehaul < node <= self.n_total

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

    @classmethod
    def random(
        cls,
        n_linehaul: int = 6,
        n_backhaul: int = 3,
        vehicle_capacity: float = 50.0,
        seed: int | None = None,
    ) -> VRPBInstance:
        rng = np.random.default_rng(seed)
        n_total = n_linehaul + n_backhaul
        coords = rng.uniform(0, 100, size=(1 + n_total, 2))
        coords[0] = [50, 50]
        demands = rng.integers(5, 20, size=n_total).astype(float)
        return cls(
            n_linehaul=n_linehaul, n_backhaul=n_backhaul,
            coords=coords, demands=demands,
            vehicle_capacity=vehicle_capacity,
            name=f"random_vrpb_{n_linehaul}L_{n_backhaul}B",
        )


@dataclass
class VRPBSolution:
    """VRPB solution.

    Attributes:
        routes: List of routes. Each route: linehaul customers first, then backhaul.
        total_distance: Total travel distance.
    """

    routes: list[list[int]]
    total_distance: float

    def __repr__(self) -> str:
        return (f"VRPBSolution(routes={len(self.routes)}, "
                f"dist={self.total_distance:.1f})")


def validate_solution(
    instance: VRPBInstance, solution: VRPBSolution
) -> tuple[bool, list[str]]:
    errors = []

    # All customers visited exactly once
    visited = {}
    for route in solution.routes:
        for c in route:
            visited[c] = visited.get(c, 0) + 1

    for c in range(1, instance.n_total + 1):
        count = visited.get(c, 0)
        if count == 0:
            errors.append(f"Customer {c} not visited")
        elif count > 1:
            errors.append(f"Customer {c} visited {count} times")

    for idx, route in enumerate(solution.routes):
        # Check linehaul-before-backhaul constraint
        seen_backhaul = False
        for node in route:
            if instance.is_backhaul(node):
                seen_backhaul = True
            elif instance.is_linehaul(node) and seen_backhaul:
                errors.append(f"Route {idx}: linehaul {node} after backhaul")

        # Check capacity separately for linehaul and backhaul
        lh_load = sum(instance.demands[c - 1] for c in route if instance.is_linehaul(c))
        bh_load = sum(instance.demands[c - 1] for c in route if instance.is_backhaul(c))
        if lh_load > instance.vehicle_capacity + 1e-6:
            errors.append(f"Route {idx}: linehaul load {lh_load:.1f} > cap {instance.vehicle_capacity:.1f}")
        if bh_load > instance.vehicle_capacity + 1e-6:
            errors.append(f"Route {idx}: backhaul load {bh_load:.1f} > cap {instance.vehicle_capacity:.1f}")

    # Distance
    actual = sum(instance.route_distance(r) for r in solution.routes)
    if abs(actual - solution.total_distance) > 1e-2:
        errors.append(f"Distance: {solution.total_distance:.2f} != {actual:.2f}")

    return len(errors) == 0, errors


def small_vrpb_5() -> VRPBInstance:
    coords = np.array([
        [50, 50],  # depot
        [20, 70], [30, 30], [70, 80],  # linehaul
        [80, 40], [60, 20],            # backhaul
    ], dtype=float)
    demands = np.array([12, 15, 10, 8, 10], dtype=float)
    return VRPBInstance(
        n_linehaul=3, n_backhaul=2, coords=coords, demands=demands,
        vehicle_capacity=30.0, name="small_vrpb_5",
    )


if __name__ == "__main__":
    inst = small_vrpb_5()
    print(f"{inst.name}: {inst.n_linehaul}L, {inst.n_backhaul}B")
