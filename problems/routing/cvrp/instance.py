"""
Capacitated Vehicle Routing Problem (CVRP) — Instance and Solution definitions.

Problem notation: CVRP (m | C | Σd)

Given n customers with demands, a depot (node 0), a fleet of identical
vehicles each with capacity Q, and pairwise distances, find a set of
routes (one per vehicle) starting and ending at the depot that:
- Visits each customer exactly once
- Respects vehicle capacity constraints
- Minimizes total travel distance

Complexity: NP-hard (generalizes TSP and Bin Packing).

References:
    Dantzig, G.B. & Ramser, J.H. (1959). The truck dispatching problem.
    Management Science, 6(1), 80-91.
    https://doi.org/10.1287/mnsc.6.1.80

    Toth, P. & Vigo, D. (2014). Vehicle Routing: Problems, Methods,
    and Applications (2nd ed.). SIAM.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


@dataclass
class CVRPInstance:
    """Capacitated Vehicle Routing Problem instance.

    Attributes:
        n: Number of customers (excluding depot).
        capacity: Vehicle capacity Q.
        demands: Array of customer demands, shape (n,). Index i corresponds
            to customer i+1 (since depot is node 0).
        distance_matrix: (n+1) x (n+1) distance matrix. Node 0 is the depot,
            nodes 1..n are customers.
        coords: Optional (n+1, 2) array of coordinates (depot + customers).
        num_vehicles: Max number of vehicles (None = unlimited).
        name: Optional instance name.
    """

    n: int
    capacity: float
    demands: np.ndarray
    distance_matrix: np.ndarray
    coords: np.ndarray | None = None
    num_vehicles: int | None = None
    name: str = ""

    def __post_init__(self):
        self.demands = np.asarray(self.demands, dtype=float)
        self.distance_matrix = np.asarray(self.distance_matrix, dtype=float)

        if self.demands.shape != (self.n,):
            raise ValueError(
                f"demands shape {self.demands.shape} does not match n={self.n}"
            )
        if self.distance_matrix.shape != (self.n + 1, self.n + 1):
            raise ValueError(
                f"distance_matrix shape {self.distance_matrix.shape} "
                f"does not match n+1={self.n + 1}"
            )
        if np.any(self.demands > self.capacity):
            raise ValueError("Some customer demands exceed vehicle capacity")
        if self.coords is not None:
            self.coords = np.asarray(self.coords, dtype=float)
            if self.coords.shape != (self.n + 1, 2):
                raise ValueError(
                    f"coords shape {self.coords.shape} does not match n+1={self.n + 1}"
                )

    @classmethod
    def random(
        cls,
        n: int,
        capacity: float = 100.0,
        demand_range: tuple[float, float] = (5.0, 30.0),
        coord_range: tuple[float, float] = (0.0, 100.0),
        seed: int | None = None,
    ) -> CVRPInstance:
        """Generate a random CVRP instance with Euclidean distances.

        Args:
            n: Number of customers.
            capacity: Vehicle capacity.
            demand_range: Range for random demands.
            coord_range: Range for random coordinates.
            seed: Random seed for reproducibility.

        Returns:
            A random CVRPInstance.
        """
        rng = np.random.default_rng(seed)
        # Depot at center, customers random
        depot = np.array([[
            (coord_range[0] + coord_range[1]) / 2,
            (coord_range[0] + coord_range[1]) / 2,
        ]])
        customers = rng.uniform(coord_range[0], coord_range[1], size=(n, 2))
        coords = np.vstack([depot, customers])

        dist = np.sqrt(
            np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=2)
        )

        demands = rng.uniform(demand_range[0], demand_range[1], size=n)
        demands = np.round(demands).astype(float)
        # Ensure no demand exceeds capacity
        demands = np.minimum(demands, capacity)

        return cls(
            n=n,
            capacity=capacity,
            demands=demands,
            distance_matrix=dist,
            coords=coords,
            name=f"random_{n}",
        )

    @classmethod
    def from_coordinates(
        cls,
        coords: np.ndarray | list,
        demands: np.ndarray | list,
        capacity: float,
        name: str = "",
    ) -> CVRPInstance:
        """Create a CVRP instance from coordinates.

        Args:
            coords: (n+1, 2) array. First row is depot.
            demands: (n,) array of customer demands.
            capacity: Vehicle capacity.
            name: Optional instance name.

        Returns:
            A CVRPInstance with Euclidean distances.
        """
        coords = np.asarray(coords, dtype=float)
        n = len(coords) - 1
        dist = np.sqrt(
            np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=2)
        )
        return cls(
            n=n,
            capacity=capacity,
            demands=np.asarray(demands, dtype=float),
            distance_matrix=dist,
            coords=coords,
            name=name,
        )

    def route_distance(self, route: list[int]) -> float:
        """Compute distance of a single route (depot -> customers -> depot).

        Args:
            route: List of customer indices (1-indexed). Empty list = empty route.

        Returns:
            Total route distance including return to depot.
        """
        if not route:
            return 0.0
        dist = self.distance_matrix[0][route[0]]
        for i in range(len(route) - 1):
            dist += self.distance_matrix[route[i]][route[i + 1]]
        dist += self.distance_matrix[route[-1]][0]
        return dist

    def route_demand(self, route: list[int]) -> float:
        """Compute total demand served by a route.

        Args:
            route: List of customer indices (1-indexed).

        Returns:
            Total demand.
        """
        return sum(self.demands[c - 1] for c in route)

    def total_distance(self, routes: list[list[int]]) -> float:
        """Compute total distance across all routes.

        Args:
            routes: List of routes, each a list of customer indices (1-indexed).

        Returns:
            Total distance.
        """
        return sum(self.route_distance(r) for r in routes)


@dataclass
class CVRPSolution:
    """Solution to a CVRP instance.

    Attributes:
        routes: List of routes. Each route is a list of customer indices
            (1-indexed, not including depot).
        distance: Total distance across all routes.
    """

    routes: list[list[int]]
    distance: float

    @property
    def num_vehicles(self) -> int:
        """Number of vehicles (routes) used."""
        return len([r for r in self.routes if r])

    def __repr__(self) -> str:
        return (
            f"CVRPSolution(distance={self.distance:.2f}, "
            f"vehicles={self.num_vehicles}, "
            f"routes={self.routes})"
        )


def validate_solution(
    instance: CVRPInstance, solution: CVRPSolution
) -> tuple[bool, list[str]]:
    """Validate a CVRP solution.

    Args:
        instance: The CVRP instance.
        solution: The solution to validate.

    Returns:
        Tuple of (is_valid, list of error messages).
    """
    errors = []

    # Check all customers visited exactly once
    all_customers = []
    for route in solution.routes:
        all_customers.extend(route)

    expected = set(range(1, instance.n + 1))
    visited = set(all_customers)

    if len(all_customers) != len(set(all_customers)):
        errors.append("Some customers visited more than once")
    if visited != expected:
        missing = expected - visited
        extra = visited - expected
        if missing:
            errors.append(f"Unvisited customers: {missing}")
        if extra:
            errors.append(f"Invalid customer indices: {extra}")

    # Check capacity constraints
    for i, route in enumerate(solution.routes):
        demand = instance.route_demand(route)
        if demand > instance.capacity + 1e-10:
            errors.append(
                f"Route {i} exceeds capacity: {demand:.1f} > {instance.capacity:.1f}"
            )

    # Check vehicle limit
    if instance.num_vehicles is not None:
        if solution.num_vehicles > instance.num_vehicles:
            errors.append(
                f"Uses {solution.num_vehicles} vehicles, "
                f"limit is {instance.num_vehicles}"
            )

    return len(errors) == 0, errors


# ── Benchmark instances ──────────────────────────────────────────────────────


def small6() -> CVRPInstance:
    """6-customer instance with known structure.

    Depot at center, customers in two clusters.
    Capacity = 15, demands = [5, 5, 5, 5, 5, 5].
    Optimal uses 2 vehicles.
    """
    coords = [
        [50, 50],   # depot
        [20, 70],   # customer 1
        [30, 80],   # customer 2
        [10, 60],   # customer 3
        [80, 30],   # customer 4
        [70, 20],   # customer 5
        [90, 40],   # customer 6
    ]
    demands = [5, 5, 5, 5, 5, 5]
    return CVRPInstance.from_coordinates(coords, demands, capacity=15.0, name="small6")


def christofides1() -> CVRPInstance:
    """Christofides instance 1 — 5 customers, capacity 6.

    A small instance used in the original Clarke-Wright paper examples.
    """
    dist = [
        [0, 3, 5, 4, 6, 7],
        [3, 0, 6, 3, 7, 8],
        [5, 6, 0, 7, 3, 4],
        [4, 3, 7, 0, 6, 7],
        [6, 7, 3, 6, 0, 2],
        [7, 8, 4, 7, 2, 0],
    ]
    demands = [2, 3, 2, 3, 2]
    return CVRPInstance(
        n=5,
        capacity=6.0,
        demands=np.array(demands, dtype=float),
        distance_matrix=np.array(dist, dtype=float),
        name="christofides1",
    )


def medium12() -> CVRPInstance:
    """12-customer instance with mixed demands. Capacity=40."""
    rng = np.random.default_rng(123)
    coords = np.vstack([
        [[50, 50]],  # depot
        rng.uniform(0, 100, size=(12, 2))
    ])
    demands = np.array([10, 15, 8, 12, 7, 20, 5, 18, 9, 14, 11, 6], dtype=float)
    dist = np.sqrt(
        np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=2)
    )
    return CVRPInstance(
        n=12,
        capacity=40.0,
        demands=demands,
        distance_matrix=dist,
        coords=coords,
        name="medium12",
    )


if __name__ == "__main__":
    inst = small6()
    print(f"small6: {inst.n} customers, capacity={inst.capacity}")
    print(f"  demands: {inst.demands}")

    c1 = christofides1()
    print(f"\nchristofides1: {c1.n} customers, capacity={c1.capacity}")
    print(f"  demands: {c1.demands}")

    m12 = medium12()
    print(f"\nmedium12: {m12.n} customers, capacity={m12.capacity}")
    print(f"  demands: {m12.demands}")
    print(f"  total demand: {m12.demands.sum()}")
