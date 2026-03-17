"""
VRP with Pickup and Delivery (VRPPD) — Instance and Solution.

Problem notation: VRPPD (m | C, PD | Sigma d)

Each request has a pickup location and a delivery location. A vehicle must
visit the pickup before the delivery, and the load must not exceed capacity
at any point. The objective is to minimize total travel distance.

Complexity: NP-hard (generalizes CVRP).

References:
    Savelsbergh, M.W.P. & Sol, M. (1995). The general pickup and delivery
    problem. Transportation Science, 29(1), 17-29.
    https://doi.org/10.1287/trsc.29.1.17

    Ropke, S. & Pisinger, D. (2006). An adaptive large neighborhood search
    heuristic for the pickup and delivery problem with time windows.
    Transportation Science, 40(4), 455-472.
    https://doi.org/10.1287/trsc.1050.0135
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class VRPPDInstance:
    """VRP with Pickup and Delivery instance.

    Attributes:
        n_requests: Number of pickup-delivery requests.
        capacity: Vehicle capacity Q.
        pickups: Array of pickup node indices, shape (n_requests,).
            Pickup nodes are 1..n_requests.
        deliveries: Array of delivery node indices, shape (n_requests,).
            Delivery nodes are n_requests+1..2*n_requests.
        loads: Array of load for each request, shape (n_requests,).
            Picked up at pickup node, dropped off at delivery node.
        distance_matrix: (2*n_requests+1) x (2*n_requests+1) distance matrix.
            Node 0 is the depot.
        coords: Optional (2*n_requests+1, 2) array of coordinates.
        name: Optional instance name.
    """

    n_requests: int
    capacity: float
    loads: np.ndarray
    distance_matrix: np.ndarray
    coords: np.ndarray | None = None
    name: str = ""

    def __post_init__(self):
        self.loads = np.asarray(self.loads, dtype=float)
        self.distance_matrix = np.asarray(self.distance_matrix, dtype=float)

        n = self.n_requests
        if self.loads.shape != (n,):
            raise ValueError(
                f"loads shape {self.loads.shape} != ({n},)"
            )
        total_nodes = 2 * n + 1
        if self.distance_matrix.shape != (total_nodes, total_nodes):
            raise ValueError(
                f"distance_matrix shape {self.distance_matrix.shape} "
                f"!= ({total_nodes}, {total_nodes})"
            )
        if np.any(self.loads > self.capacity):
            raise ValueError("Some request loads exceed vehicle capacity")
        if np.any(self.loads <= 0):
            raise ValueError("All request loads must be positive")
        if self.coords is not None:
            self.coords = np.asarray(self.coords, dtype=float)

    @property
    def pickups(self) -> list[int]:
        """Pickup node indices (1..n_requests)."""
        return list(range(1, self.n_requests + 1))

    @property
    def deliveries(self) -> list[int]:
        """Delivery node indices (n_requests+1..2*n_requests)."""
        return list(range(self.n_requests + 1, 2 * self.n_requests + 1))

    def pickup_of(self, request: int) -> int:
        """Return pickup node for request (0-indexed request)."""
        return request + 1

    def delivery_of(self, request: int) -> int:
        """Return delivery node for request (0-indexed request)."""
        return self.n_requests + request + 1

    def request_of_node(self, node: int) -> int:
        """Return request index (0-indexed) for a pickup or delivery node."""
        if 1 <= node <= self.n_requests:
            return node - 1
        elif self.n_requests + 1 <= node <= 2 * self.n_requests:
            return node - self.n_requests - 1
        else:
            raise ValueError(f"Node {node} is not a pickup or delivery node")

    def is_pickup(self, node: int) -> bool:
        """Check if a node is a pickup node."""
        return 1 <= node <= self.n_requests

    def is_delivery(self, node: int) -> bool:
        """Check if a node is a delivery node."""
        return self.n_requests + 1 <= node <= 2 * self.n_requests

    @classmethod
    def random(
        cls,
        n_requests: int,
        capacity: float = 50.0,
        load_range: tuple[float, float] = (5.0, 20.0),
        coord_range: tuple[float, float] = (0.0, 100.0),
        seed: int | None = None,
    ) -> VRPPDInstance:
        """Generate a random VRPPD instance with Euclidean distances.

        Args:
            n_requests: Number of pickup-delivery requests.
            capacity: Vehicle capacity.
            load_range: Range for random loads.
            coord_range: Range for random coordinates.
            seed: Random seed for reproducibility.

        Returns:
            A random VRPPDInstance.
        """
        rng = np.random.default_rng(seed)
        total_nodes = 2 * n_requests + 1

        depot = np.array([[
            (coord_range[0] + coord_range[1]) / 2,
            (coord_range[0] + coord_range[1]) / 2,
        ]])
        locations = rng.uniform(
            coord_range[0], coord_range[1], size=(2 * n_requests, 2)
        )
        coords = np.vstack([depot, locations])

        dist = np.sqrt(
            np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=2)
        )

        loads = np.round(
            rng.uniform(load_range[0], load_range[1], size=n_requests)
        ).astype(float)
        loads = np.minimum(loads, capacity)

        return cls(
            n_requests=n_requests,
            capacity=capacity,
            loads=loads,
            distance_matrix=dist,
            coords=coords,
            name=f"random_vrppd_{n_requests}",
        )

    def route_distance(self, route: list[int]) -> float:
        """Compute distance of a single route (depot -> nodes -> depot).

        Args:
            route: List of node indices (pickup/delivery).

        Returns:
            Total route distance.
        """
        if not route:
            return 0.0
        dist = self.distance_matrix[0][route[0]]
        for i in range(len(route) - 1):
            dist += self.distance_matrix[route[i]][route[i + 1]]
        dist += self.distance_matrix[route[-1]][0]
        return dist

    def total_distance(self, routes: list[list[int]]) -> float:
        """Compute total distance across all routes."""
        return sum(self.route_distance(r) for r in routes)

    def route_feasible(self, route: list[int]) -> tuple[bool, str]:
        """Check if a route is feasible (capacity + precedence).

        Args:
            route: List of node indices.

        Returns:
            Tuple of (is_feasible, error_message).
        """
        if not route:
            return True, ""

        # Check precedence: pickup before delivery for each request
        visited_pickups: set[int] = set()
        current_load = 0.0

        for node in route:
            if self.is_pickup(node):
                req = self.request_of_node(node)
                visited_pickups.add(req)
                current_load += self.loads[req]
                if current_load > self.capacity + 1e-10:
                    return False, f"Capacity exceeded at node {node}"
            elif self.is_delivery(node):
                req = self.request_of_node(node)
                if req not in visited_pickups:
                    return False, (
                        f"Delivery {node} before pickup {self.pickup_of(req)}"
                    )
                current_load -= self.loads[req]
            else:
                return False, f"Invalid node {node}"

        return True, ""


@dataclass
class VRPPDSolution:
    """Solution to a VRPPD instance.

    Attributes:
        routes: List of routes, each a list of node indices.
        distance: Total distance across all routes.
    """

    routes: list[list[int]]
    distance: float

    @property
    def num_vehicles(self) -> int:
        return len([r for r in self.routes if r])

    def __repr__(self) -> str:
        return (
            f"VRPPDSolution(distance={self.distance:.2f}, "
            f"vehicles={self.num_vehicles}, "
            f"routes={self.routes})"
        )


def validate_solution(
    instance: VRPPDInstance, solution: VRPPDSolution
) -> tuple[bool, list[str]]:
    """Validate a VRPPD solution.

    Args:
        instance: The VRPPD instance.
        solution: The solution to validate.

    Returns:
        Tuple of (is_valid, list of error messages).
    """
    errors = []
    n = instance.n_requests

    # Collect all visited nodes
    all_nodes = []
    for route in solution.routes:
        all_nodes.extend(route)

    # Check all pickup and delivery nodes visited exactly once
    expected = set(range(1, 2 * n + 1))
    visited = set(all_nodes)

    if len(all_nodes) != len(set(all_nodes)):
        errors.append("Some nodes visited more than once")
    if visited != expected:
        missing = expected - visited
        extra = visited - expected
        if missing:
            errors.append(f"Unvisited nodes: {missing}")
        if extra:
            errors.append(f"Invalid node indices: {extra}")

    # Check each route for feasibility
    for i, route in enumerate(solution.routes):
        feasible, msg = instance.route_feasible(route)
        if not feasible:
            errors.append(f"Route {i}: {msg}")

    # Check that pickup and delivery for same request are in same route
    for req in range(n):
        p = instance.pickup_of(req)
        d = instance.delivery_of(req)
        p_route = None
        d_route = None
        for idx, route in enumerate(solution.routes):
            if p in route:
                p_route = idx
            if d in route:
                d_route = idx
        if p_route is not None and d_route is not None and p_route != d_route:
            errors.append(
                f"Request {req}: pickup in route {p_route}, "
                f"delivery in route {d_route}"
            )

    return len(errors) == 0, errors


# ── Benchmark instances ──────────────────────────────────────────────────────


def small_vrppd3() -> VRPPDInstance:
    """Small 3-request VRPPD instance for testing.

    Depot at origin, 3 pickup-delivery pairs.
    Nodes: 0=depot, 1-3=pickups, 4-6=deliveries.
    """
    coords = np.array([
        [50, 50],  # depot
        [20, 80],  # pickup 1
        [80, 80],  # pickup 2
        [50, 20],  # pickup 3
        [30, 30],  # delivery 1
        [70, 30],  # delivery 2
        [50, 80],  # delivery 3
    ], dtype=float)

    dist = np.sqrt(
        np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=2)
    )

    return VRPPDInstance(
        n_requests=3,
        capacity=30.0,
        loads=np.array([10.0, 10.0, 10.0]),
        distance_matrix=dist,
        coords=coords,
        name="small_vrppd3",
    )


def medium_vrppd5() -> VRPPDInstance:
    """5-request VRPPD instance.

    Nodes: 0=depot, 1-5=pickups, 6-10=deliveries.
    """
    coords = np.array([
        [50, 50],  # depot
        [10, 90],  # pickup 1
        [90, 90],  # pickup 2
        [10, 10],  # pickup 3
        [90, 10],  # pickup 4
        [50, 90],  # pickup 5
        [30, 30],  # delivery 1
        [70, 30],  # delivery 2
        [30, 70],  # delivery 3
        [70, 70],  # delivery 4
        [50, 10],  # delivery 5
    ], dtype=float)

    dist = np.sqrt(
        np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=2)
    )

    return VRPPDInstance(
        n_requests=5,
        capacity=25.0,
        loads=np.array([8.0, 12.0, 7.0, 10.0, 9.0]),
        distance_matrix=dist,
        coords=coords,
        name="medium_vrppd5",
    )


if __name__ == "__main__":
    inst = small_vrppd3()
    print(f"small_vrppd3: {inst.n_requests} requests, capacity={inst.capacity}")
    print(f"  loads: {inst.loads}")
    print(f"  pickups: {inst.pickups}")
    print(f"  deliveries: {inst.deliveries}")

    inst2 = medium_vrppd5()
    print(f"\nmedium_vrppd5: {inst2.n_requests} requests, capacity={inst2.capacity}")
