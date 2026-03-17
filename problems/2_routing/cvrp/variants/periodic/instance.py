"""
Periodic Vehicle Routing Problem (PVRP) — Instance and Solution.

Over a planning horizon of T periods, each customer must be visited a
specified number of times. Select visit day combinations and build
routes for each day.

Complexity: NP-hard (generalizes CVRP).

References:
    Christofides, N. & Beasley, J.E. (1984). The period routing problem.
    Networks, 14(2), 237-256.
    https://doi.org/10.1002/net.3230140205

    Cordeau, J.F., Gendreau, M. & Laporte, G. (1997). A tabu search
    heuristic for periodic and multi-depot vehicle routing problems.
    Networks, 30(2), 105-119.
    https://doi.org/10.1002/(SICI)1097-0037(199709)30:2<105::AID-NET5>3.0.CO;2-G
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class PVRPInstance:
    """Periodic VRP instance.

    Attributes:
        n: Number of customers (1..n, depot is 0).
        coords: Coordinates, shape (n+1, 2).
        demands: Per-visit demand per customer, shape (n,).
        capacity: Vehicle capacity.
        num_periods: Number of days in planning horizon.
        visit_freq: Required visits per customer, shape (n,).
        name: Optional instance name.
    """

    n: int
    coords: np.ndarray
    demands: np.ndarray
    capacity: float
    num_periods: int
    visit_freq: np.ndarray
    name: str = ""

    def __post_init__(self):
        self.coords = np.asarray(self.coords, dtype=float)
        self.demands = np.asarray(self.demands, dtype=float)
        self.visit_freq = np.asarray(self.visit_freq, dtype=int)

    def distance(self, i: int, j: int) -> float:
        return float(np.linalg.norm(self.coords[i] - self.coords[j]))

    @classmethod
    def random(
        cls,
        n: int = 8,
        num_periods: int = 3,
        capacity: float = 50.0,
        seed: int | None = None,
    ) -> PVRPInstance:
        rng = np.random.default_rng(seed)
        coords = rng.uniform(0, 100, size=(n + 1, 2))
        demands = rng.integers(5, 15, size=n).astype(float)
        visit_freq = rng.integers(1, min(num_periods, 3) + 1, size=n)
        return cls(n=n, coords=coords, demands=demands, capacity=capacity,
                   num_periods=num_periods, visit_freq=visit_freq,
                   name=f"random_{n}")

    def route_distance(self, route: list[int]) -> float:
        if not route:
            return 0.0
        d = self.distance(0, route[0])
        for i in range(len(route) - 1):
            d += self.distance(route[i], route[i + 1])
        d += self.distance(route[-1], 0)
        return d


@dataclass
class PVRPSolution:
    """Periodic VRP solution.

    Attributes:
        day_routes: day_routes[t] = list of routes for day t.
        total_distance: Total distance across all days.
    """

    day_routes: list[list[list[int]]]
    total_distance: float

    def __repr__(self) -> str:
        total_routes = sum(len(dr) for dr in self.day_routes)
        return f"PVRPSolution(days={len(self.day_routes)}, routes={total_routes}, dist={self.total_distance:.1f})"


def validate_solution(
    instance: PVRPInstance, solution: PVRPSolution
) -> tuple[bool, list[str]]:
    errors = []

    if len(solution.day_routes) != instance.num_periods:
        errors.append(f"Expected {instance.num_periods} days, got {len(solution.day_routes)}")

    # Count visits per customer
    visit_count = np.zeros(instance.n, dtype=int)
    for t, routes in enumerate(solution.day_routes):
        day_visited = set()
        for route in routes:
            load = sum(instance.demands[c - 1] for c in route)
            if load > instance.capacity + 1e-6:
                errors.append(f"Day {t}: route overloaded {load:.2f} > {instance.capacity}")
            for c in route:
                if c < 1 or c > instance.n:
                    errors.append(f"Invalid customer {c}")
                if c in day_visited:
                    errors.append(f"Day {t}: customer {c} visited twice")
                day_visited.add(c)
                visit_count[c - 1] += 1

    for i in range(instance.n):
        if visit_count[i] != instance.visit_freq[i]:
            errors.append(
                f"Customer {i+1}: visited {visit_count[i]} times, need {instance.visit_freq[i]}")

    actual_dist = 0.0
    for routes in solution.day_routes:
        for route in routes:
            actual_dist += instance.route_distance(route)
    if abs(actual_dist - solution.total_distance) > 1e-2:
        errors.append(f"Distance: {solution.total_distance:.2f} != {actual_dist:.2f}")

    return len(errors) == 0, errors


def small_pvrp_6() -> PVRPInstance:
    return PVRPInstance(
        n=6,
        coords=np.array([
            [50, 50],
            [20, 70], [80, 70], [20, 30],
            [80, 30], [50, 90], [50, 10],
        ], dtype=float),
        demands=np.array([8, 10, 6, 12, 7, 9], dtype=float),
        capacity=30.0,
        num_periods=3,
        visit_freq=np.array([2, 1, 3, 1, 2, 1]),
        name="small_6",
    )


if __name__ == "__main__":
    inst = small_pvrp_6()
    print(f"{inst.name}: n={inst.n}, periods={inst.num_periods}")
    print(f"  Visit freq: {inst.visit_freq}")
