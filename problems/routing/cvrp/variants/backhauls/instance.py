"""
Vehicle Routing Problem with Backhauls (VRPB) — Instance and Solution.

Problem notation: VRPB

Customers split into linehaul (delivery) and backhaul (pickup) sets.
Each route must serve all linehaul customers before any backhaul
customers (precedence constraint). Vehicle capacity applies to both
deliveries and pickups separately.

Applications: grocery distribution with empty returns, beverage
distribution with bottle collection, postal delivery with parcel
pickup.

Complexity: NP-hard (generalizes CVRP).

References:
    Goetschalckx, M. & Jacobs-Blecha, C. (1989). The vehicle routing
    problem with backhauls. European Journal of Operational Research,
    42(1), 39-51. https://doi.org/10.1016/0377-2217(89)90057-X

    Toth, P. & Vigo, D. (1999). A heuristic algorithm for the
    symmetric and asymmetric vehicle routing problems with backhauls.
    European Journal of Operational Research, 113(3), 528-543.
    https://doi.org/10.1016/S0377-2217(98)00022-8
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class VRPBInstance:
    """VRP with Backhauls instance.

    Customers 1..n_linehaul are linehaul (deliveries).
    Customers n_linehaul+1..n_linehaul+n_backhaul are backhaul (pickups).

    Attributes:
        n_linehaul: Number of linehaul customers.
        n_backhaul: Number of backhaul customers.
        n: Total customers (n_linehaul + n_backhaul).
        capacity: Vehicle capacity.
        demands: Demands for all customers + depot, shape (n+1,).
        distance_matrix: (n+1, n+1) distance matrix.
        coords: Optional coordinates.
        name: Optional instance name.
    """

    n_linehaul: int
    n_backhaul: int
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
        cls,
        n_linehaul: int = 6,
        n_backhaul: int = 4,
        capacity: float = 100.0,
        demand_range: tuple[float, float] = (10.0, 30.0),
        seed: int | None = None,
    ) -> VRPBInstance:
        """Generate a random VRPB instance."""
        rng = np.random.default_rng(seed)
        n = n_linehaul + n_backhaul
        coords = rng.uniform(0, 100, size=(n + 1, 2))
        demands = np.zeros(n + 1)
        demands[1:] = np.round(rng.uniform(demand_range[0], demand_range[1], size=n))
        dist = np.sqrt(np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=2))
        return cls(
            n_linehaul=n_linehaul, n_backhaul=n_backhaul, n=n,
            capacity=capacity, demands=demands, distance_matrix=dist,
            coords=coords, name=f"random_{n_linehaul}_{n_backhaul}",
        )

    def is_linehaul(self, c: int) -> bool:
        """Check if customer c is linehaul."""
        return 1 <= c <= self.n_linehaul

    def is_backhaul(self, c: int) -> bool:
        """Check if customer c is backhaul."""
        return self.n_linehaul < c <= self.n

    def route_distance(self, route: list[int]) -> float:
        """Total route distance (depot -> customers -> depot)."""
        if not route:
            return 0.0
        dist = self.distance_matrix[0][route[0]]
        for i in range(len(route) - 1):
            dist += self.distance_matrix[route[i]][route[i + 1]]
        dist += self.distance_matrix[route[-1]][0]
        return dist

    def route_precedence_feasible(self, route: list[int]) -> bool:
        """Check linehaul-before-backhaul precedence."""
        seen_backhaul = False
        for c in route:
            if self.is_backhaul(c):
                seen_backhaul = True
            elif self.is_linehaul(c) and seen_backhaul:
                return False
        return True


@dataclass
class VRPBSolution:
    """Solution to a VRPB instance."""

    routes: list[list[int]]
    total_distance: float

    def __repr__(self) -> str:
        return f"VRPBSolution(distance={self.total_distance:.2f}, routes={len(self.routes)})"


def validate_solution(
    instance: VRPBInstance, solution: VRPBSolution
) -> tuple[bool, list[str]]:
    """Validate a VRPB solution."""
    errors = []
    all_customers = []
    for k, route in enumerate(solution.routes):
        lh = [c for c in route if instance.is_linehaul(c)]
        bh = [c for c in route if instance.is_backhaul(c)]

        if not instance.route_precedence_feasible(route):
            errors.append(f"Route {k}: linehaul-before-backhaul violated")

        lh_demand = sum(instance.demands[c] for c in lh)
        bh_demand = sum(instance.demands[c] for c in bh)
        if lh_demand > instance.capacity + 1e-10:
            errors.append(f"Route {k}: linehaul demand {lh_demand:.1f} > cap")
        if bh_demand > instance.capacity + 1e-10:
            errors.append(f"Route {k}: backhaul demand {bh_demand:.1f} > cap")

        all_customers.extend(route)

    if sorted(all_customers) != list(range(1, instance.n + 1)):
        errors.append("Not all customers visited exactly once")

    actual_dist = sum(instance.route_distance(r) for r in solution.routes)
    if abs(actual_dist - solution.total_distance) > 1e-4:
        errors.append(f"Reported dist {solution.total_distance:.2f} != actual {actual_dist:.2f}")

    return len(errors) == 0, errors


def small_vrpb_4_3() -> VRPBInstance:
    """4 linehaul, 3 backhaul customers."""
    coords = np.array([
        [50, 50],  # depot
        [20, 80], [80, 80], [80, 20], [20, 20],  # linehaul
        [40, 60], [60, 40], [50, 90],             # backhaul
    ], dtype=float)
    n = 7
    dist = np.sqrt(np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=2))
    demands = np.array([0, 20, 25, 15, 30, 10, 20, 15], dtype=float)
    return VRPBInstance(
        n_linehaul=4, n_backhaul=3, n=n, capacity=60,
        demands=demands, distance_matrix=dist, coords=coords,
        name="small_4_3",
    )


if __name__ == "__main__":
    inst = small_vrpb_4_3()
    print(f"{inst.name}: {inst.n_linehaul} LH, {inst.n_backhaul} BH")
