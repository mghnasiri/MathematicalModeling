"""
Vehicle Routing Problem with Time Windows (VRPTW) — Instance and Solution.

Problem notation: VRPTW (m | C, TW | Σd)

Extends CVRP with time window constraints [e_i, l_i] for each customer i.
A vehicle must arrive at customer i no later than l_i. If it arrives
before e_i, it waits until e_i (no early service). Service takes s_i
time units. The depot also has a time window [e_0, l_0] (planning horizon).

Complexity: NP-hard (generalizes CVRP).

Two common objectives:
- Minimize total distance (primary)
- Hierarchical: minimize number of vehicles, then total distance

References:
    Solomon, M.M. (1987). Algorithms for the vehicle routing and
    scheduling problems with time window constraints. Operations
    Research, 35(2), 254-265.
    https://doi.org/10.1287/opre.35.2.254

    Desrochers, M., Desrosiers, J. & Solomon, M. (1992). A new
    optimization algorithm for the vehicle routing problem with
    time windows. Operations Research, 40(2), 342-354.
    https://doi.org/10.1287/opre.40.2.342
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class VRPTWInstance:
    """Vehicle Routing Problem with Time Windows instance.

    Attributes:
        n: Number of customers (excluding depot).
        capacity: Vehicle capacity Q.
        demands: Array of customer demands, shape (n,). Index i corresponds
            to customer i+1 (node 0 is the depot).
        distance_matrix: (n+1) x (n+1) distance matrix.
        time_windows: (n+1, 2) array of [earliest, latest] arrival times.
            Row 0 is the depot time window (planning horizon).
        service_times: (n+1,) array of service durations.
            service_times[0] = 0 for depot.
        coords: Optional (n+1, 2) array of coordinates.
        name: Optional instance name.
    """

    n: int
    capacity: float
    demands: np.ndarray
    distance_matrix: np.ndarray
    time_windows: np.ndarray
    service_times: np.ndarray
    coords: np.ndarray | None = None
    name: str = ""

    def __post_init__(self):
        self.demands = np.asarray(self.demands, dtype=float)
        self.distance_matrix = np.asarray(self.distance_matrix, dtype=float)
        self.time_windows = np.asarray(self.time_windows, dtype=float)
        self.service_times = np.asarray(self.service_times, dtype=float)

        if self.demands.shape != (self.n,):
            raise ValueError(
                f"demands shape {self.demands.shape} != ({self.n},)"
            )
        if self.distance_matrix.shape != (self.n + 1, self.n + 1):
            raise ValueError(
                f"distance_matrix shape {self.distance_matrix.shape} "
                f"!= ({self.n + 1}, {self.n + 1})"
            )
        if self.time_windows.shape != (self.n + 1, 2):
            raise ValueError(
                f"time_windows shape {self.time_windows.shape} "
                f"!= ({self.n + 1}, 2)"
            )
        if self.service_times.shape != (self.n + 1,):
            raise ValueError(
                f"service_times shape {self.service_times.shape} "
                f"!= ({self.n + 1},)"
            )
        if np.any(self.demands > self.capacity):
            raise ValueError("Some customer demands exceed vehicle capacity")
        if self.coords is not None:
            self.coords = np.asarray(self.coords, dtype=float)

    @classmethod
    def random(
        cls,
        n: int,
        capacity: float = 100.0,
        horizon: float = 500.0,
        demand_range: tuple[float, float] = (5.0, 25.0),
        tw_width_range: tuple[float, float] = (30.0, 100.0),
        service_time: float = 10.0,
        coord_range: tuple[float, float] = (0.0, 100.0),
        seed: int | None = None,
    ) -> VRPTWInstance:
        """Generate a random VRPTW instance with Euclidean distances.

        Args:
            n: Number of customers.
            capacity: Vehicle capacity.
            horizon: Planning horizon (depot closes at this time).
            demand_range: Range for random demands.
            tw_width_range: Range for time window widths.
            service_time: Uniform service time for all customers.
            coord_range: Range for random coordinates.
            seed: Random seed for reproducibility.

        Returns:
            A random VRPTWInstance.
        """
        rng = np.random.default_rng(seed)

        depot = np.array([[
            (coord_range[0] + coord_range[1]) / 2,
            (coord_range[0] + coord_range[1]) / 2,
        ]])
        customers = rng.uniform(coord_range[0], coord_range[1], size=(n, 2))
        coords = np.vstack([depot, customers])

        dist = np.sqrt(
            np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=2)
        )

        demands = np.round(
            rng.uniform(demand_range[0], demand_range[1], size=n)
        ).astype(float)
        demands = np.minimum(demands, capacity)

        # Generate time windows
        time_windows = np.zeros((n + 1, 2))
        time_windows[0] = [0.0, horizon]  # Depot

        service_times = np.zeros(n + 1)
        service_times[0] = 0.0

        for i in range(1, n + 1):
            # Earliest possible arrival from depot
            earliest_possible = dist[0][i]
            tw_width = rng.uniform(tw_width_range[0], tw_width_range[1])
            earliest = rng.uniform(
                earliest_possible,
                max(earliest_possible, horizon - tw_width - service_time)
            )
            latest = earliest + tw_width
            time_windows[i] = [earliest, min(latest, horizon)]
            service_times[i] = service_time

        return cls(
            n=n,
            capacity=capacity,
            demands=demands,
            distance_matrix=dist,
            time_windows=time_windows,
            service_times=service_times,
            coords=coords,
            name=f"random_{n}",
        )

    def travel_time(self, i: int, j: int) -> float:
        """Travel time from node i to node j (equals distance for unit speed)."""
        return self.distance_matrix[i][j]

    def route_distance(self, route: list[int]) -> float:
        """Compute distance of a single route (depot -> customers -> depot).

        Args:
            route: List of customer indices (1-indexed).

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

    def route_demand(self, route: list[int]) -> float:
        """Compute total demand served by a route."""
        return sum(self.demands[c - 1] for c in route)

    def route_schedule(self, route: list[int]) -> list[float]:
        """Compute the service start times for each customer in a route.

        Args:
            route: List of customer indices (1-indexed).

        Returns:
            List of service start times for each customer in the route.
        """
        if not route:
            return []
        times = []
        current_time = self.time_windows[0][0]  # Start at depot opening

        prev = 0
        for cust in route:
            arrival = current_time + self.travel_time(prev, cust)
            start = max(arrival, self.time_windows[cust][0])
            times.append(start)
            current_time = start + self.service_times[cust]
            prev = cust

        return times

    def route_feasible(self, route: list[int]) -> bool:
        """Check if a route is feasible (capacity and time windows).

        Args:
            route: List of customer indices (1-indexed).

        Returns:
            True if route is feasible.
        """
        if not route:
            return True

        # Check capacity
        if self.route_demand(route) > self.capacity + 1e-10:
            return False

        # Check time windows
        current_time = self.time_windows[0][0]
        prev = 0
        for cust in route:
            arrival = current_time + self.travel_time(prev, cust)
            if arrival > self.time_windows[cust][1] + 1e-10:
                return False  # Arrived after window closes
            start = max(arrival, self.time_windows[cust][0])
            current_time = start + self.service_times[cust]
            prev = cust

        # Check return to depot
        return_time = current_time + self.travel_time(prev, 0)
        if return_time > self.time_windows[0][1] + 1e-10:
            return False

        return True

    def total_distance(self, routes: list[list[int]]) -> float:
        """Compute total distance across all routes."""
        return sum(self.route_distance(r) for r in routes)


@dataclass
class VRPTWSolution:
    """Solution to a VRPTW instance.

    Attributes:
        routes: List of routes, each a list of customer indices (1-indexed).
        distance: Total distance across all routes.
    """

    routes: list[list[int]]
    distance: float

    @property
    def num_vehicles(self) -> int:
        return len([r for r in self.routes if r])

    def __repr__(self) -> str:
        return (
            f"VRPTWSolution(distance={self.distance:.2f}, "
            f"vehicles={self.num_vehicles}, "
            f"routes={self.routes})"
        )


def validate_solution(
    instance: VRPTWInstance, solution: VRPTWSolution
) -> tuple[bool, list[str]]:
    """Validate a VRPTW solution.

    Args:
        instance: The VRPTW instance.
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

    # Check each route
    for i, route in enumerate(solution.routes):
        if not route:
            continue

        # Capacity
        demand = instance.route_demand(route)
        if demand > instance.capacity + 1e-10:
            errors.append(
                f"Route {i}: capacity exceeded ({demand:.1f} > {instance.capacity:.1f})"
            )

        # Time windows
        current_time = instance.time_windows[0][0]
        prev = 0
        for cust in route:
            arrival = current_time + instance.travel_time(prev, cust)
            if arrival > instance.time_windows[cust][1] + 1e-10:
                errors.append(
                    f"Route {i}: customer {cust} arrival {arrival:.1f} "
                    f"> latest {instance.time_windows[cust][1]:.1f}"
                )
            start = max(arrival, instance.time_windows[cust][0])
            current_time = start + instance.service_times[cust]
            prev = cust

        return_time = current_time + instance.travel_time(prev, 0)
        if return_time > instance.time_windows[0][1] + 1e-10:
            errors.append(
                f"Route {i}: returns to depot at {return_time:.1f} "
                f"> depot closes at {instance.time_windows[0][1]:.1f}"
            )

    return len(errors) == 0, errors


# ── Benchmark instances ──────────────────────────────────────────────────────


def solomon_c101_mini() -> VRPTWInstance:
    """Small 8-customer instance inspired by Solomon's C101 structure.

    Clustered customers, tight time windows, capacity = 40.
    """
    coords = [
        [40, 50],   # depot
        [45, 68],   # customer 1
        [42, 66],   # customer 2
        [42, 65],   # customer 3
        [40, 69],   # customer 4
        [25, 35],   # customer 5
        [22, 30],   # customer 6
        [25, 30],   # customer 7
        [20, 35],   # customer 8
    ]
    demands = [10, 10, 20, 10, 10, 10, 20, 10]

    coords = np.array(coords, dtype=float)
    n = 8
    dist = np.sqrt(
        np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=2)
    )

    time_windows = np.array([
        [0, 300],     # depot
        [50, 100],    # customer 1
        [60, 110],    # customer 2
        [70, 120],    # customer 3
        [55, 105],    # customer 4
        [100, 170],   # customer 5
        [110, 180],   # customer 6
        [120, 190],   # customer 7
        [105, 175],   # customer 8
    ], dtype=float)

    service_times = np.array([0, 10, 10, 10, 10, 10, 10, 10, 10], dtype=float)

    return VRPTWInstance(
        n=n,
        capacity=40.0,
        demands=np.array(demands, dtype=float),
        distance_matrix=dist,
        time_windows=time_windows,
        service_times=service_times,
        coords=coords,
        name="solomon_c101_mini",
    )


def tight_tw5() -> VRPTWInstance:
    """5-customer instance with very tight time windows.

    Forces specific orderings due to narrow windows.
    """
    dist = np.array([
        [0, 10, 15, 20, 12, 18],
        [10, 0, 12, 22, 15, 20],
        [15, 12, 0, 10, 18, 14],
        [20, 22, 10, 0, 16, 11],
        [12, 15, 18, 16, 0, 9],
        [18, 20, 14, 11, 9, 0],
    ], dtype=float)

    demands = [5, 8, 6, 7, 4]

    time_windows = np.array([
        [0, 200],     # depot
        [10, 40],     # customer 1 — early
        [30, 60],     # customer 2
        [50, 80],     # customer 3
        [20, 50],     # customer 4
        [60, 90],     # customer 5
    ], dtype=float)

    service_times = np.array([0, 5, 5, 5, 5, 5], dtype=float)

    return VRPTWInstance(
        n=5,
        capacity=20.0,
        demands=np.array(demands, dtype=float),
        distance_matrix=dist,
        time_windows=time_windows,
        service_times=service_times,
        name="tight_tw5",
    )


if __name__ == "__main__":
    inst = solomon_c101_mini()
    print(f"solomon_c101_mini: {inst.n} customers, capacity={inst.capacity}")
    print(f"  demands: {inst.demands}")
    print(f"  time windows:\n{inst.time_windows}")

    inst2 = tight_tw5()
    print(f"\ntight_tw5: {inst2.n} customers, capacity={inst2.capacity}")
