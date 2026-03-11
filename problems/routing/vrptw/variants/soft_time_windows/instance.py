"""
VRPTW with Soft Time Windows — Instance and Solution.

Extends VRPTW by allowing time window violations with a penalty cost.
Early arrival still requires waiting, but late arrival incurs a penalty
proportional to the delay.

Complexity: NP-hard (generalizes VRPTW).

References:
    Taillard, E., Badeau, P., Gendreau, M., Guertin, F. & Potvin, J.Y.
    (1997). A tabu search heuristic for the vehicle routing problem with
    soft time windows. Transportation Science, 31(2), 170-186.
    https://doi.org/10.1287/trsc.31.2.170
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class SoftTWInstance:
    """VRPTW with Soft Time Windows instance.

    Attributes:
        n: Number of customers (1..n, depot is 0).
        coords: Coordinates, shape (n+1, 2).
        demands: Customer demands, shape (n,).
        capacity: Vehicle capacity.
        time_windows: [earliest, latest] per node, shape (n+1, 2).
        service_times: Service time per node, shape (n+1,).
        penalty_rate: Penalty cost per unit of late arrival.
        name: Optional instance name.
    """

    n: int
    coords: np.ndarray
    demands: np.ndarray
    capacity: float
    time_windows: np.ndarray
    service_times: np.ndarray
    penalty_rate: float = 1.0
    name: str = ""

    def __post_init__(self):
        self.coords = np.asarray(self.coords, dtype=float)
        self.demands = np.asarray(self.demands, dtype=float)
        self.time_windows = np.asarray(self.time_windows, dtype=float)
        self.service_times = np.asarray(self.service_times, dtype=float)

    def distance(self, i: int, j: int) -> float:
        return float(np.linalg.norm(self.coords[i] - self.coords[j]))

    @classmethod
    def random(
        cls,
        n: int = 10,
        capacity: float = 50.0,
        penalty_rate: float = 2.0,
        seed: int | None = None,
    ) -> SoftTWInstance:
        rng = np.random.default_rng(seed)
        coords = rng.uniform(0, 100, size=(n + 1, 2))
        demands = rng.integers(5, 20, size=n).astype(float)
        service = np.zeros(n + 1)
        service[1:] = rng.integers(2, 8, size=n).astype(float)
        tw = np.zeros((n + 1, 2))
        tw[0] = [0, 500]  # depot horizon
        for i in range(1, n + 1):
            e = rng.integers(0, 200)
            tw[i] = [e, e + rng.integers(30, 100)]
        return cls(n=n, coords=coords, demands=demands, capacity=capacity,
                   time_windows=tw, service_times=service,
                   penalty_rate=penalty_rate, name=f"random_{n}")

    def route_cost(self, route: list[int]) -> tuple[float, float, float]:
        """Compute distance, penalty, and total cost for a route.

        Returns:
            (distance, penalty, total_cost) tuple.
        """
        if not route:
            return 0.0, 0.0, 0.0
        dist = self.distance(0, route[0])
        time = dist
        penalty = 0.0

        for idx, cust in enumerate(route):
            e, l = self.time_windows[cust]
            if time < e:
                time = e  # wait
            if time > l:
                penalty += (time - l) * self.penalty_rate
            time += self.service_times[cust]
            if idx < len(route) - 1:
                d = self.distance(cust, route[idx + 1])
                dist += d
                time += d

        dist += self.distance(route[-1], 0)
        return dist, penalty, dist + penalty


@dataclass
class SoftTWSolution:
    """Soft TW VRP solution.

    Attributes:
        routes: List of routes (lists of customer indices).
        total_distance: Total travel distance.
        total_penalty: Total time window violation penalty.
        total_cost: distance + penalty.
    """

    routes: list[list[int]]
    total_distance: float
    total_penalty: float
    total_cost: float

    def __repr__(self) -> str:
        return (f"SoftTWSolution(routes={len(self.routes)}, "
                f"dist={self.total_distance:.1f}, pen={self.total_penalty:.1f})")


def validate_solution(
    instance: SoftTWInstance, solution: SoftTWSolution
) -> tuple[bool, list[str]]:
    errors = []
    visited = set()
    for route in solution.routes:
        load = sum(instance.demands[c - 1] for c in route)
        if load > instance.capacity + 1e-6:
            errors.append(f"Route overloaded: {load:.2f} > {instance.capacity}")
        for c in route:
            if c < 1 or c > instance.n:
                errors.append(f"Invalid customer {c}")
            if c in visited:
                errors.append(f"Customer {c} visited twice")
            visited.add(c)
    if len(visited) != instance.n:
        errors.append(f"Not all customers visited: {len(visited)}/{instance.n}")

    actual_dist = 0.0
    actual_pen = 0.0
    for route in solution.routes:
        d, p, _ = instance.route_cost(route)
        actual_dist += d
        actual_pen += p
    if abs(actual_dist - solution.total_distance) > 1e-2:
        errors.append(f"Distance: {solution.total_distance:.2f} != {actual_dist:.2f}")
    if abs(actual_pen - solution.total_penalty) > 1e-2:
        errors.append(f"Penalty: {solution.total_penalty:.2f} != {actual_pen:.2f}")

    return len(errors) == 0, errors


def small_softtw_6() -> SoftTWInstance:
    return SoftTWInstance(
        n=6,
        coords=np.array([
            [50, 50],
            [20, 70], [80, 70], [20, 30],
            [80, 30], [50, 90], [50, 10],
        ], dtype=float),
        demands=np.array([10, 8, 12, 7, 15, 9], dtype=float),
        capacity=30.0,
        time_windows=np.array([
            [0, 300],
            [10, 80], [20, 90], [30, 100],
            [10, 70], [40, 120], [20, 80],
        ], dtype=float),
        service_times=np.array([0, 5, 5, 5, 5, 5, 5], dtype=float),
        penalty_rate=2.0,
        name="small_6",
    )


if __name__ == "__main__":
    inst = small_softtw_6()
    print(f"{inst.name}: n={inst.n}, penalty_rate={inst.penalty_rate}")
