"""
Location-Routing Problem (LRP) -- Instance and Solution definitions.

Problem notation: LRP (m | Q, U_j | min fixed + routing)

The Location-Routing Problem jointly optimizes facility (depot) location
and vehicle routing decisions. Given m candidate depot locations with
fixed opening costs and capacities, n customers with demands, a vehicle
capacity Q, and a distance matrix, select depots to open, assign
customers to depots, and design vehicle routes for each depot to minimize
total fixed cost plus total routing distance.

Complexity: NP-hard (generalizes both UFLP and CVRP).

References:
    Laporte, G. (1988). Location-routing problems. In: Golden, B.L.
    & Assad, A.A. (eds) Vehicle Routing: Methods and Studies, North-
    Holland, 163-198.

    Nagy, G. & Salhi, S. (2007). Location-routing: Issues, models and
    methods. European Journal of Operational Research, 177(2), 649-672.
    https://doi.org/10.1016/j.ejor.2006.04.004

    Prodhon, C. & Prins, C. (2014). A survey of recent research on
    location-routing problems. European Journal of Operational Research,
    238(1), 1-17.
    https://doi.org/10.1016/j.ejor.2014.01.005

    Drexl, M. & Schneider, M. (2015). A survey of variants and
    extensions of the location-routing problem. European Journal of
    Operational Research, 241(2), 283-308.
    https://doi.org/10.1016/j.ejor.2014.08.030
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


@dataclass
class LRPInstance:
    """Location-Routing Problem instance.

    The distance matrix is indexed as follows:
    - Indices 0..m-1 correspond to candidate depots.
    - Indices m..m+n-1 correspond to customers.
    The full matrix has shape (m + n, m + n).

    Attributes:
        m: Number of candidate depot locations.
        n: Number of customers.
        fixed_costs: Array of depot opening costs, shape (m,).
        capacities: Array of depot capacities, shape (m,). Each depot j
            can serve at most U_j total demand.
        demands: Array of customer demands, shape (n,). Customer i has
            demand d_i.
        vehicle_capacity: Vehicle capacity Q. Each route serves at most Q
            total demand.
        distance_matrix: Pairwise distances, shape (m + n, m + n).
            Rows/cols 0..m-1 are depots, m..m+n-1 are customers.
        coords: Optional (m + n, 2) coordinates (depots first, then
            customers).
        name: Optional instance name.
    """

    m: int
    n: int
    fixed_costs: np.ndarray
    capacities: np.ndarray
    demands: np.ndarray
    vehicle_capacity: float
    distance_matrix: np.ndarray
    coords: np.ndarray | None = None
    name: str = ""

    def __post_init__(self):
        self.fixed_costs = np.asarray(self.fixed_costs, dtype=float)
        self.capacities = np.asarray(self.capacities, dtype=float)
        self.demands = np.asarray(self.demands, dtype=float)
        self.distance_matrix = np.asarray(self.distance_matrix, dtype=float)

        if self.fixed_costs.shape != (self.m,):
            raise ValueError(
                f"fixed_costs shape {self.fixed_costs.shape} != ({self.m},)"
            )
        if self.capacities.shape != (self.m,):
            raise ValueError(
                f"capacities shape {self.capacities.shape} != ({self.m},)"
            )
        if self.demands.shape != (self.n,):
            raise ValueError(
                f"demands shape {self.demands.shape} != ({self.n},)"
            )
        total_nodes = self.m + self.n
        if self.distance_matrix.shape != (total_nodes, total_nodes):
            raise ValueError(
                f"distance_matrix shape {self.distance_matrix.shape} "
                f"!= ({total_nodes}, {total_nodes})"
            )
        if self.coords is not None:
            self.coords = np.asarray(self.coords, dtype=float)
            if self.coords.shape != (total_nodes, 2):
                raise ValueError(
                    f"coords shape {self.coords.shape} "
                    f"!= ({total_nodes}, 2)"
                )

    def customer_node(self, cust_idx: int) -> int:
        """Map customer index (0-based) to distance matrix node index.

        Args:
            cust_idx: Customer index in [0, n).

        Returns:
            Node index in the distance matrix (m + cust_idx).
        """
        return self.m + cust_idx

    def depot_node(self, depot_idx: int) -> int:
        """Map depot index (0-based) to distance matrix node index.

        Args:
            depot_idx: Depot index in [0, m).

        Returns:
            Node index in the distance matrix (depot_idx).
        """
        return depot_idx

    def route_distance(self, depot_idx: int, route: list[int]) -> float:
        """Compute distance of a single route from a depot.

        A route is a list of customer indices (0-based). The vehicle
        travels depot -> customers -> depot.

        Args:
            depot_idx: Depot index in [0, m).
            route: List of customer indices (0-based).

        Returns:
            Total route distance including return to depot.
        """
        if not route:
            return 0.0
        d = self.depot_node(depot_idx)
        nodes = [self.customer_node(c) for c in route]
        dist = self.distance_matrix[d][nodes[0]]
        for i in range(len(nodes) - 1):
            dist += self.distance_matrix[nodes[i]][nodes[i + 1]]
        dist += self.distance_matrix[nodes[-1]][d]
        return dist

    def route_demand(self, route: list[int]) -> float:
        """Compute total demand served by a route.

        Args:
            route: List of customer indices (0-based).

        Returns:
            Sum of customer demands on the route.
        """
        return float(sum(self.demands[c] for c in route))

    @classmethod
    def random(
        cls,
        m: int,
        n: int,
        vehicle_capacity: float = 50.0,
        fixed_cost_range: tuple[float, float] = (200.0, 800.0),
        demand_range: tuple[float, float] = (5.0, 20.0),
        coord_range: tuple[float, float] = (0.0, 100.0),
        seed: int | None = None,
    ) -> LRPInstance:
        """Generate a random LRP instance with Euclidean distances.

        Args:
            m: Number of candidate depots.
            n: Number of customers.
            vehicle_capacity: Vehicle capacity Q.
            fixed_cost_range: Range for depot fixed opening costs.
            demand_range: Range for customer demands.
            coord_range: Range for random coordinates.
            seed: Random seed for reproducibility.

        Returns:
            A random LRPInstance.
        """
        rng = np.random.default_rng(seed)
        depot_coords = rng.uniform(
            coord_range[0], coord_range[1], size=(m, 2)
        )
        cust_coords = rng.uniform(
            coord_range[0], coord_range[1], size=(n, 2)
        )
        coords = np.vstack([depot_coords, cust_coords])

        dist = np.sqrt(
            np.sum(
                (coords[:, None, :] - coords[None, :, :]) ** 2, axis=2
            )
        )

        fixed_costs = np.round(
            rng.uniform(fixed_cost_range[0], fixed_cost_range[1], size=m)
        ).astype(float)

        demands = np.round(
            rng.uniform(demand_range[0], demand_range[1], size=n)
        ).astype(float)

        # Depot capacities: enough that a few depots can cover all demand
        total_demand = demands.sum()
        cap_per_depot = total_demand / max(1, m // 2)
        capacities = np.round(
            rng.uniform(cap_per_depot * 0.8, cap_per_depot * 1.5, size=m)
        ).astype(float)

        return cls(
            m=m,
            n=n,
            fixed_costs=fixed_costs,
            capacities=capacities,
            demands=demands,
            vehicle_capacity=vehicle_capacity,
            distance_matrix=dist,
            coords=coords,
            name=f"random_{m}_{n}",
        )

    @classmethod
    def from_coordinates(
        cls,
        depot_coords: np.ndarray | list,
        customer_coords: np.ndarray | list,
        fixed_costs: np.ndarray | list,
        capacities: np.ndarray | list,
        demands: np.ndarray | list,
        vehicle_capacity: float,
        name: str = "",
    ) -> LRPInstance:
        """Create an LRP instance from coordinates.

        Args:
            depot_coords: (m, 2) array of depot coordinates.
            customer_coords: (n, 2) array of customer coordinates.
            fixed_costs: (m,) array of depot opening costs.
            capacities: (m,) array of depot capacities.
            demands: (n,) array of customer demands.
            vehicle_capacity: Vehicle capacity Q.
            name: Optional instance name.

        Returns:
            An LRPInstance with Euclidean distances.
        """
        depot_coords = np.asarray(depot_coords, dtype=float)
        customer_coords = np.asarray(customer_coords, dtype=float)
        coords = np.vstack([depot_coords, customer_coords])

        dist = np.sqrt(
            np.sum(
                (coords[:, None, :] - coords[None, :, :]) ** 2, axis=2
            )
        )

        m = len(depot_coords)
        n = len(customer_coords)

        return cls(
            m=m,
            n=n,
            fixed_costs=np.asarray(fixed_costs, dtype=float),
            capacities=np.asarray(capacities, dtype=float),
            demands=np.asarray(demands, dtype=float),
            vehicle_capacity=vehicle_capacity,
            distance_matrix=dist,
            coords=coords,
            name=name,
        )


@dataclass
class LRPSolution:
    """Solution to a Location-Routing Problem.

    Attributes:
        open_depots: List of opened depot indices (0-based).
        routes: Dict mapping depot index to list of routes. Each route
            is a list of customer indices (0-based).
        cost: Total cost (fixed opening costs + total routing distance).
    """

    open_depots: list[int]
    routes: dict[int, list[list[int]]]
    cost: float

    @property
    def num_vehicles(self) -> int:
        """Total number of vehicles (routes) used across all depots."""
        return sum(
            len(depot_routes)
            for depot_routes in self.routes.values()
        )

    @property
    def fixed_cost(self) -> float:
        """Fixed cost component (extracted from routes is not possible;
        stored separately if needed). This returns 0 as a fallback;
        use compute_cost for accurate decomposition."""
        return 0.0

    def __repr__(self) -> str:
        return (
            f"LRPSolution(cost={self.cost:.2f}, "
            f"depots={self.open_depots}, "
            f"vehicles={self.num_vehicles})"
        )


def compute_cost(
    instance: LRPInstance, solution: LRPSolution
) -> tuple[float, float, float]:
    """Compute detailed cost breakdown for an LRP solution.

    Args:
        instance: The LRP instance.
        solution: The LRP solution.

    Returns:
        Tuple of (total_cost, fixed_cost, routing_cost).
    """
    fixed = float(sum(instance.fixed_costs[d] for d in solution.open_depots))
    routing = 0.0
    for depot_idx, depot_routes in solution.routes.items():
        for route in depot_routes:
            routing += instance.route_distance(depot_idx, route)
    return fixed + routing, fixed, routing


def validate_solution(
    instance: LRPInstance, solution: LRPSolution
) -> tuple[bool, list[str]]:
    """Validate an LRP solution.

    Checks:
    - All depot indices are valid and open.
    - Every customer is visited exactly once.
    - Vehicle capacity is respected on each route.
    - Depot capacity is respected (total demand assigned <= U_j).
    - Reported cost matches computed cost.

    Args:
        instance: The LRP instance.
        solution: The solution to validate.

    Returns:
        Tuple of (is_valid, list of error messages).
    """
    errors: list[str] = []

    # Check depot indices
    open_set = set(solution.open_depots)
    for d in solution.open_depots:
        if d < 0 or d >= instance.m:
            errors.append(f"Invalid depot index: {d}")

    # Check routes only from open depots
    for d in solution.routes:
        if d not in open_set:
            errors.append(f"Routes assigned to closed depot {d}")

    # Check all customers visited exactly once
    all_customers: list[int] = []
    for depot_idx, depot_routes in solution.routes.items():
        for route in depot_routes:
            all_customers.extend(route)

    expected = set(range(instance.n))
    visited = set(all_customers)

    if len(all_customers) != len(set(all_customers)):
        duplicates = [
            c for c in all_customers if all_customers.count(c) > 1
        ]
        errors.append(f"Customers visited more than once: {set(duplicates)}")
    if visited != expected:
        missing = expected - visited
        extra = visited - expected
        if missing:
            errors.append(f"Unvisited customers: {missing}")
        if extra:
            errors.append(f"Invalid customer indices: {extra}")

    # Check vehicle capacity on each route
    for depot_idx, depot_routes in solution.routes.items():
        for r_idx, route in enumerate(depot_routes):
            demand = instance.route_demand(route)
            if demand > instance.vehicle_capacity + 1e-10:
                errors.append(
                    f"Depot {depot_idx} route {r_idx}: demand "
                    f"{demand:.1f} > vehicle capacity "
                    f"{instance.vehicle_capacity:.1f}"
                )

    # Check depot capacity
    for depot_idx in solution.open_depots:
        depot_routes = solution.routes.get(depot_idx, [])
        total_demand = sum(
            instance.route_demand(route) for route in depot_routes
        )
        if total_demand > instance.capacities[depot_idx] + 1e-10:
            errors.append(
                f"Depot {depot_idx}: total demand {total_demand:.1f} > "
                f"capacity {instance.capacities[depot_idx]:.1f}"
            )

    # Check cost
    if not errors:
        actual_cost, _, _ = compute_cost(instance, solution)
        if abs(actual_cost - solution.cost) > 1e-4:
            errors.append(
                f"Reported cost {solution.cost:.2f} != "
                f"actual {actual_cost:.2f}"
            )

    return len(errors) == 0, errors


# -- Benchmark instances -------------------------------------------------------


def small_lrp_3_8() -> LRPInstance:
    """Small LRP: 3 candidate depots, 8 customers.

    Layout:
    - Depot 0 at (10, 50), low cost, moderate capacity
    - Depot 1 at (50, 50), medium cost, high capacity
    - Depot 2 at (90, 50), high cost, moderate capacity
    - Customers spread across the plane
    - Vehicle capacity Q = 30
    """
    depot_coords = np.array([
        [10.0, 50.0],
        [50.0, 50.0],
        [90.0, 50.0],
    ])
    customer_coords = np.array([
        [5.0, 70.0],    # customer 0 — near depot 0
        [15.0, 30.0],   # customer 1 — near depot 0
        [25.0, 60.0],   # customer 2 — between depot 0 and 1
        [45.0, 80.0],   # customer 3 — near depot 1
        [55.0, 20.0],   # customer 4 — near depot 1
        [75.0, 70.0],   # customer 5 — between depot 1 and 2
        [85.0, 30.0],   # customer 6 — near depot 2
        [95.0, 60.0],   # customer 7 — near depot 2
    ])
    fixed_costs = np.array([150.0, 250.0, 200.0])
    capacities = np.array([60.0, 100.0, 60.0])
    demands = np.array([8.0, 10.0, 7.0, 12.0, 9.0, 11.0, 8.0, 10.0])
    vehicle_capacity = 30.0

    return LRPInstance.from_coordinates(
        depot_coords=depot_coords,
        customer_coords=customer_coords,
        fixed_costs=fixed_costs,
        capacities=capacities,
        demands=demands,
        vehicle_capacity=vehicle_capacity,
        name="small_3_8",
    )


def medium_lrp_5_15() -> LRPInstance:
    """Medium LRP: 5 candidate depots, 15 customers. Seed-generated."""
    return LRPInstance.random(5, 15, seed=42)


if __name__ == "__main__":
    inst = small_lrp_3_8()
    print(f"{inst.name}: m={inst.m}, n={inst.n}")
    print(f"  fixed costs: {inst.fixed_costs}")
    print(f"  capacities:  {inst.capacities}")
    print(f"  demands:     {inst.demands}")
    print(f"  total demand: {inst.demands.sum():.0f}")
    print(f"  vehicle Q:   {inst.vehicle_capacity}")
    print()

    inst2 = medium_lrp_5_15()
    print(f"{inst2.name}: m={inst2.m}, n={inst2.n}")
    print(f"  total demand: {inst2.demands.sum():.0f}")
