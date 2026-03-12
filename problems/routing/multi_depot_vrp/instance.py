"""
Multi-Depot Vehicle Routing Problem (MDVRP) — Instance and Solution.

Problem notation: MDVRP (m depots | C | Sigma d)

Given multiple depots, each with a fleet of vehicles, and n customers
with demands, find routes starting and ending at their respective depots
that visit each customer exactly once, respect vehicle capacity, and
minimize total travel distance.

Complexity: NP-hard (generalizes CVRP).

References:
    Cordeau, J.-F., Gendreau, M. & Laporte, G. (1997). A tabu search
    heuristic for periodic and multi-depot vehicle routing problems.
    Networks, 30(2), 105-119.
    https://doi.org/10.1002/(SICI)1097-0037(199709)30:2<105::AID-NET5>3.0.CO;2-G

    Montoya-Torres, J.R., Franco, J.L., Isaza, S.N., Jimenez, H.F. &
    Herazo-Padilla, N. (2015). A literature review on the vehicle routing
    problem with multiple depots. Computers & Industrial Engineering,
    79, 115-129.
    https://doi.org/10.1016/j.cie.2014.10.029
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class MDVRPInstance:
    """Multi-Depot Vehicle Routing Problem instance.

    Attributes:
        n_customers: Number of customers.
        n_depots: Number of depots.
        capacity: Vehicle capacity Q (same for all vehicles).
        demands: Array of customer demands, shape (n_customers,).
        depot_coords: (n_depots, 2) array of depot coordinates.
        customer_coords: (n_customers, 2) array of customer coordinates.
        distance_matrix: (n_depots + n_customers) x (n_depots + n_customers)
            distance matrix. Nodes 0..n_depots-1 are depots,
            nodes n_depots..n_depots+n_customers-1 are customers.
        vehicles_per_depot: Max vehicles per depot (None = unlimited).
        name: Optional instance name.
    """

    n_customers: int
    n_depots: int
    capacity: float
    demands: np.ndarray
    depot_coords: np.ndarray
    customer_coords: np.ndarray
    distance_matrix: np.ndarray
    vehicles_per_depot: int | None = None
    name: str = ""

    def __post_init__(self):
        self.demands = np.asarray(self.demands, dtype=float)
        self.depot_coords = np.asarray(self.depot_coords, dtype=float)
        self.customer_coords = np.asarray(self.customer_coords, dtype=float)
        self.distance_matrix = np.asarray(self.distance_matrix, dtype=float)

        if self.demands.shape != (self.n_customers,):
            raise ValueError(
                f"demands shape {self.demands.shape} != ({self.n_customers},)"
            )
        if self.depot_coords.shape != (self.n_depots, 2):
            raise ValueError(
                f"depot_coords shape {self.depot_coords.shape} "
                f"!= ({self.n_depots}, 2)"
            )
        if self.customer_coords.shape != (self.n_customers, 2):
            raise ValueError(
                f"customer_coords shape {self.customer_coords.shape} "
                f"!= ({self.n_customers}, 2)"
            )
        total = self.n_depots + self.n_customers
        if self.distance_matrix.shape != (total, total):
            raise ValueError(
                f"distance_matrix shape {self.distance_matrix.shape} "
                f"!= ({total}, {total})"
            )
        if np.any(self.demands > self.capacity):
            raise ValueError("Some customer demands exceed vehicle capacity")

    def customer_node(self, customer_idx: int) -> int:
        """Convert customer index (0-based) to node index in distance matrix."""
        return self.n_depots + customer_idx

    def depot_node(self, depot_idx: int) -> int:
        """Convert depot index (0-based) to node index in distance matrix."""
        return depot_idx

    @classmethod
    def random(
        cls,
        n_customers: int,
        n_depots: int = 2,
        capacity: float = 100.0,
        demand_range: tuple[float, float] = (5.0, 30.0),
        coord_range: tuple[float, float] = (0.0, 100.0),
        seed: int | None = None,
    ) -> MDVRPInstance:
        """Generate a random MDVRP instance with Euclidean distances.

        Args:
            n_customers: Number of customers.
            n_depots: Number of depots.
            capacity: Vehicle capacity.
            demand_range: Range for random demands.
            coord_range: Range for random coordinates.
            seed: Random seed for reproducibility.

        Returns:
            A random MDVRPInstance.
        """
        rng = np.random.default_rng(seed)

        depot_coords = rng.uniform(
            coord_range[0], coord_range[1], size=(n_depots, 2)
        )
        customer_coords = rng.uniform(
            coord_range[0], coord_range[1], size=(n_customers, 2)
        )

        all_coords = np.vstack([depot_coords, customer_coords])
        dist = np.sqrt(
            np.sum(
                (all_coords[:, None, :] - all_coords[None, :, :]) ** 2, axis=2
            )
        )

        demands = np.round(
            rng.uniform(demand_range[0], demand_range[1], size=n_customers)
        ).astype(float)
        demands = np.minimum(demands, capacity)

        return cls(
            n_customers=n_customers,
            n_depots=n_depots,
            capacity=capacity,
            demands=demands,
            depot_coords=depot_coords,
            customer_coords=customer_coords,
            distance_matrix=dist,
            name=f"random_mdvrp_{n_customers}_{n_depots}",
        )

    def route_distance(self, depot_idx: int, route: list[int]) -> float:
        """Compute distance of a route from a specific depot.

        Args:
            depot_idx: Depot index (0-based).
            route: List of customer indices (0-based).

        Returns:
            Total route distance.
        """
        if not route:
            return 0.0
        d_node = self.depot_node(depot_idx)
        c_nodes = [self.customer_node(c) for c in route]
        dist = self.distance_matrix[d_node][c_nodes[0]]
        for i in range(len(c_nodes) - 1):
            dist += self.distance_matrix[c_nodes[i]][c_nodes[i + 1]]
        dist += self.distance_matrix[c_nodes[-1]][d_node]
        return dist

    def route_demand(self, route: list[int]) -> float:
        """Compute total demand of a route (customer indices 0-based)."""
        return sum(self.demands[c] for c in route)

    def total_distance(
        self, depot_routes: dict[int, list[list[int]]]
    ) -> float:
        """Compute total distance across all depots and routes.

        Args:
            depot_routes: Dict mapping depot_idx -> list of routes.
                Each route is a list of customer indices (0-based).
        """
        total = 0.0
        for depot_idx, routes in depot_routes.items():
            for route in routes:
                total += self.route_distance(depot_idx, route)
        return total


@dataclass
class MDVRPSolution:
    """Solution to a MDVRP instance.

    Attributes:
        depot_routes: Dict mapping depot_idx -> list of routes.
            Each route is a list of customer indices (0-based).
        distance: Total distance across all routes.
    """

    depot_routes: dict[int, list[list[int]]]
    distance: float

    @property
    def num_vehicles(self) -> int:
        return sum(
            len([r for r in routes if r])
            for routes in self.depot_routes.values()
        )

    def __repr__(self) -> str:
        return (
            f"MDVRPSolution(distance={self.distance:.2f}, "
            f"vehicles={self.num_vehicles}, "
            f"depots={list(self.depot_routes.keys())})"
        )


def validate_solution(
    instance: MDVRPInstance, solution: MDVRPSolution
) -> tuple[bool, list[str]]:
    """Validate a MDVRP solution.

    Args:
        instance: The MDVRP instance.
        solution: The solution to validate.

    Returns:
        Tuple of (is_valid, list of error messages).
    """
    errors = []

    # Collect all visited customers
    all_customers = []
    for depot_idx, routes in solution.depot_routes.items():
        if depot_idx < 0 or depot_idx >= instance.n_depots:
            errors.append(f"Invalid depot index: {depot_idx}")
            continue
        for r_idx, route in enumerate(routes):
            all_customers.extend(route)
            # Check capacity
            demand = instance.route_demand(route)
            if demand > instance.capacity + 1e-10:
                errors.append(
                    f"Depot {depot_idx}, route {r_idx}: capacity exceeded "
                    f"({demand:.1f} > {instance.capacity:.1f})"
                )
            # Check valid customer indices
            for c in route:
                if c < 0 or c >= instance.n_customers:
                    errors.append(
                        f"Depot {depot_idx}, route {r_idx}: "
                        f"invalid customer index {c}"
                    )

    expected = set(range(instance.n_customers))
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

    return len(errors) == 0, errors


# ── Benchmark instances ──────────────────────────────────────────────────────


def small_mdvrp4() -> MDVRPInstance:
    """Small MDVRP: 2 depots, 4 customers.

    Depots at (20, 50) and (80, 50). Customers clustered near depots.
    """
    depot_coords = np.array([[20, 50], [80, 50]], dtype=float)
    customer_coords = np.array([
        [10, 60],   # customer 0 — near depot 0
        [25, 40],   # customer 1 — near depot 0
        [75, 60],   # customer 2 — near depot 1
        [85, 40],   # customer 3 — near depot 1
    ], dtype=float)

    all_coords = np.vstack([depot_coords, customer_coords])
    dist = np.sqrt(
        np.sum((all_coords[:, None, :] - all_coords[None, :, :]) ** 2, axis=2)
    )

    return MDVRPInstance(
        n_customers=4,
        n_depots=2,
        capacity=40.0,
        demands=np.array([10.0, 15.0, 12.0, 8.0]),
        depot_coords=depot_coords,
        customer_coords=customer_coords,
        distance_matrix=dist,
        name="small_mdvrp4",
    )


def medium_mdvrp8() -> MDVRPInstance:
    """Medium MDVRP: 3 depots, 8 customers.

    Depots at triangle vertices, customers scattered.
    """
    depot_coords = np.array([
        [50, 90],  # depot 0
        [10, 10],  # depot 1
        [90, 10],  # depot 2
    ], dtype=float)
    customer_coords = np.array([
        [40, 80],  # 0
        [60, 80],  # 1
        [20, 20],  # 2
        [30, 30],  # 3
        [80, 20],  # 4
        [70, 30],  # 5
        [50, 50],  # 6
        [50, 10],  # 7
    ], dtype=float)

    all_coords = np.vstack([depot_coords, customer_coords])
    dist = np.sqrt(
        np.sum((all_coords[:, None, :] - all_coords[None, :, :]) ** 2, axis=2)
    )

    return MDVRPInstance(
        n_customers=8,
        n_depots=3,
        capacity=30.0,
        demands=np.array([8.0, 10.0, 7.0, 12.0, 9.0, 11.0, 6.0, 8.0]),
        depot_coords=depot_coords,
        customer_coords=customer_coords,
        distance_matrix=dist,
        name="medium_mdvrp8",
    )


if __name__ == "__main__":
    inst = small_mdvrp4()
    print(f"small_mdvrp4: {inst.n_customers} customers, {inst.n_depots} depots")
    print(f"  demands: {inst.demands}")

    inst2 = medium_mdvrp8()
    print(f"\nmedium_mdvrp8: {inst2.n_customers} customers, {inst2.n_depots} depots")
