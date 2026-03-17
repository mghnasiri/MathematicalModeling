"""
Dial-a-Ride Problem (DARP) — Instance and Solution definitions.

Problem notation: DARP (m vehicles | pickup-delivery, time windows | min distance)

Given a fleet of m vehicles and n ride requests, each with a pickup location,
delivery location, and time window, find routes that serve all requests while
respecting vehicle capacity, time windows, and maximum ride time constraints.

Complexity: NP-hard (generalizes VRP with Pickup and Delivery).

References:
    Cordeau, J.-F. & Laporte, G. (2007). The dial-a-ride problem: models
    and algorithms. Annals of Operations Research, 153(1), 29-46.
    https://doi.org/10.1007/s10479-007-0170-8

    Jaw, J.J., Odoni, A.R., Psaraftis, H.N. & Wilson, N.H.M. (1986).
    A heuristic algorithm for the multi-vehicle advance request
    dial-a-ride problem with time windows. Transportation Research
    Part B, 20(3), 243-257.
    https://doi.org/10.1016/0191-2615(86)90020-2
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


@dataclass
class DARPRequest:
    """A single ride request.

    Attributes:
        pickup: Pickup node index.
        delivery: Delivery node index.
        load: Number of passengers (or load units).
        earliest_pickup: Earliest pickup time.
        latest_pickup: Latest pickup time.
        earliest_delivery: Earliest delivery time.
        latest_delivery: Latest delivery time.
        max_ride_time: Maximum time from pickup to delivery.
        service_time: Service time at pickup and delivery.
    """

    pickup: int
    delivery: int
    load: int = 1
    earliest_pickup: float = 0.0
    latest_pickup: float = float("inf")
    earliest_delivery: float = 0.0
    latest_delivery: float = float("inf")
    max_ride_time: float = float("inf")
    service_time: float = 0.0


@dataclass
class DARPInstance:
    """Dial-a-Ride Problem instance.

    Attributes:
        n_requests: Number of ride requests.
        n_vehicles: Number of vehicles.
        vehicle_capacity: Capacity per vehicle.
        requests: List of DARPRequest objects.
        coordinates: (2*n_requests + 2, 2) array — depot_start, pickups,
            deliveries, depot_end.
        distance_matrix: Precomputed distance matrix.
        planning_horizon: Maximum time horizon.
        name: Optional instance name.
    """

    n_requests: int
    n_vehicles: int
    vehicle_capacity: int
    requests: list[DARPRequest]
    coordinates: np.ndarray
    distance_matrix: np.ndarray
    planning_horizon: float = 480.0
    name: str = ""

    def __post_init__(self):
        self.coordinates = np.asarray(self.coordinates, dtype=float)
        self.distance_matrix = np.asarray(self.distance_matrix, dtype=float)

    @property
    def n_nodes(self) -> int:
        """Total nodes: depot_start + n pickups + n deliveries + depot_end."""
        return 2 * self.n_requests + 2

    @property
    def depot_start(self) -> int:
        return 0

    @property
    def depot_end(self) -> int:
        return 2 * self.n_requests + 1

    def pickup_node(self, req_idx: int) -> int:
        """Node index for request pickup (1-indexed)."""
        return req_idx + 1

    def delivery_node(self, req_idx: int) -> int:
        """Node index for request delivery."""
        return self.n_requests + req_idx + 1

    def distance(self, i: int, j: int) -> float:
        return self.distance_matrix[i][j]

    @classmethod
    def random(
        cls,
        n_requests: int = 5,
        n_vehicles: int = 2,
        vehicle_capacity: int = 3,
        area_size: float = 100.0,
        planning_horizon: float = 480.0,
        seed: int | None = None,
    ) -> DARPInstance:
        """Generate a random DARP instance.

        Args:
            n_requests: Number of ride requests.
            n_vehicles: Number of vehicles.
            vehicle_capacity: Vehicle capacity.
            area_size: Coordinate range [0, area_size].
            planning_horizon: Planning horizon T.
            seed: Random seed.

        Returns:
            A random DARPInstance.
        """
        rng = np.random.default_rng(seed)

        n_nodes = 2 * n_requests + 2
        coords = np.zeros((n_nodes, 2))

        # Depot at center
        depot = np.array([area_size / 2, area_size / 2])
        coords[0] = depot
        coords[-1] = depot

        # Random pickup and delivery locations
        for i in range(n_requests):
            coords[i + 1] = rng.uniform(0, area_size, 2)  # pickup
            coords[n_requests + i + 1] = rng.uniform(0, area_size, 2)  # delivery

        # Compute distance matrix (Euclidean)
        diff = coords[:, None, :] - coords[None, :, :]
        dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))

        # Generate requests with time windows
        requests = []
        for i in range(n_requests):
            p_node = i + 1
            d_node = n_requests + i + 1
            travel = dist_matrix[p_node][d_node]

            earliest_p = rng.uniform(0, planning_horizon * 0.5)
            latest_p = earliest_p + rng.uniform(30, 90)
            earliest_d = earliest_p + travel
            latest_d = latest_p + travel + rng.uniform(30, 60)
            max_ride = travel * 2.5

            requests.append(DARPRequest(
                pickup=p_node,
                delivery=d_node,
                load=int(rng.integers(1, vehicle_capacity)),
                earliest_pickup=round(earliest_p, 1),
                latest_pickup=round(min(latest_p, planning_horizon), 1),
                earliest_delivery=round(earliest_d, 1),
                latest_delivery=round(min(latest_d, planning_horizon), 1),
                max_ride_time=round(max_ride, 1),
                service_time=round(rng.uniform(1, 5), 1),
            ))

        return cls(
            n_requests=n_requests,
            n_vehicles=n_vehicles,
            vehicle_capacity=vehicle_capacity,
            requests=requests,
            coordinates=coords,
            distance_matrix=dist_matrix,
            planning_horizon=planning_horizon,
            name=f"random_darp_{n_requests}_{n_vehicles}",
        )

    def route_distance(self, route: list[int]) -> float:
        """Compute total distance of a route (list of node indices).

        Route should start and end at depot nodes.
        """
        if len(route) < 2:
            return 0.0
        total = 0.0
        for i in range(len(route) - 1):
            total += self.distance(route[i], route[i + 1])
        return total


@dataclass
class DARPSolution:
    """Solution to a Dial-a-Ride Problem.

    Attributes:
        routes: List of routes, one per vehicle.
            Each route is a list of node indices (starts/ends at depot).
        total_distance: Total distance across all routes.
        n_served: Number of requests served.
    """

    routes: list[list[int]]
    total_distance: float
    n_served: int

    def __repr__(self) -> str:
        return (
            f"DARPSolution(distance={self.total_distance:.1f}, "
            f"served={self.n_served}, vehicles={len(self.routes)})"
        )


def validate_solution(
    instance: DARPInstance, solution: DARPSolution
) -> tuple[bool, list[str]]:
    """Validate a DARP solution.

    Checks: all requests served, precedence (pickup before delivery),
    same vehicle for pickup/delivery, capacity constraints.

    Args:
        instance: The DARP instance.
        solution: The solution to validate.

    Returns:
        Tuple of (is_valid, list of error messages).
    """
    errors = []

    # Track which request's pickup/delivery appears in which route
    pickup_route: dict[int, int] = {}
    delivery_route: dict[int, int] = {}

    for r_idx, route in enumerate(solution.routes):
        # Check route starts/ends at depot
        if route and route[0] != instance.depot_start:
            errors.append(f"Route {r_idx}: doesn't start at depot")
        if route and route[-1] != instance.depot_end:
            errors.append(f"Route {r_idx}: doesn't end at depot")

        # Track pickups and deliveries
        visited_pickups = set()
        for node in route:
            for req_idx in range(instance.n_requests):
                if node == instance.pickup_node(req_idx):
                    visited_pickups.add(req_idx)
                    pickup_route[req_idx] = r_idx
                elif node == instance.delivery_node(req_idx):
                    if req_idx not in visited_pickups:
                        errors.append(
                            f"Route {r_idx}: delivery of request {req_idx} "
                            f"before pickup"
                        )
                    delivery_route[req_idx] = r_idx

    # Check same vehicle for pickup and delivery
    for req_idx in range(instance.n_requests):
        if req_idx in pickup_route and req_idx in delivery_route:
            if pickup_route[req_idx] != delivery_route[req_idx]:
                errors.append(
                    f"Request {req_idx}: pickup and delivery on different routes"
                )

    # Check all requests served
    served = set(pickup_route.keys()) & set(delivery_route.keys())
    if len(served) != instance.n_requests:
        missing = set(range(instance.n_requests)) - served
        if missing:
            errors.append(f"Unserved requests: {missing}")

    return len(errors) == 0, errors


# ── Benchmark instances ──────────────────────────────────────────────────────


def small_darp3() -> DARPInstance:
    """Small DARP: 3 requests, 2 vehicles, capacity 2.

    Nodes: 0=depot_start, 1-3=pickups, 4-6=deliveries, 7=depot_end
    """
    coords = np.array([
        [50, 50],  # depot start
        [20, 80],  # pickup 0
        [80, 80],  # pickup 1
        [50, 20],  # pickup 2
        [30, 30],  # delivery 0
        [70, 30],  # delivery 1
        [50, 80],  # delivery 2
        [50, 50],  # depot end
    ], dtype=float)

    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))

    requests = [
        DARPRequest(pickup=1, delivery=4, load=1,
                    earliest_pickup=0, latest_pickup=100,
                    earliest_delivery=0, latest_delivery=200,
                    max_ride_time=150, service_time=5),
        DARPRequest(pickup=2, delivery=5, load=1,
                    earliest_pickup=0, latest_pickup=100,
                    earliest_delivery=0, latest_delivery=200,
                    max_ride_time=150, service_time=5),
        DARPRequest(pickup=3, delivery=6, load=1,
                    earliest_pickup=0, latest_pickup=100,
                    earliest_delivery=0, latest_delivery=200,
                    max_ride_time=150, service_time=5),
    ]

    return DARPInstance(
        n_requests=3,
        n_vehicles=2,
        vehicle_capacity=2,
        requests=requests,
        coordinates=coords,
        distance_matrix=dist,
        planning_horizon=480.0,
        name="small_darp3",
    )


if __name__ == "__main__":
    inst = small_darp3()
    print(f"small_darp3: {inst.n_requests} requests, "
          f"{inst.n_vehicles} vehicles, cap={inst.vehicle_capacity}")
    print(f"  nodes: {inst.n_nodes}")
