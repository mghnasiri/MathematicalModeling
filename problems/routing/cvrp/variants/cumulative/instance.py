"""
Cumulative VRP (CumVRP) — Instance and Solution.

Minimize total arrival time at customers (sum of arrival times)
instead of total travel distance. Also known as the Minimum Latency
VRP or Multi-Vehicle Traveling Repairman Problem.

Applications: disaster relief, humanitarian logistics, service routing.

Complexity: NP-hard (generalizes Traveling Repairman Problem).

References:
    Ngueveu, S.U., Prins, C. & Wolfler Calvo, R. (2010). An effective
    memetic algorithm for the cumulative capacitated vehicle routing
    problem. Computers & Operations Research, 37(11), 1877-1885.
    https://doi.org/10.1016/j.cor.2009.06.014
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class CumVRPInstance:
    """Cumulative VRP instance.

    Attributes:
        n: Number of customers (1..n, depot is 0).
        coords: Coordinates, shape (n+1, 2).
        demands: Customer demands, shape (n,).
        capacity: Vehicle capacity.
        name: Optional instance name.
    """

    n: int
    coords: np.ndarray
    demands: np.ndarray
    capacity: float
    name: str = ""

    def __post_init__(self):
        self.coords = np.asarray(self.coords, dtype=float)
        self.demands = np.asarray(self.demands, dtype=float)

    def distance(self, i: int, j: int) -> float:
        return float(np.linalg.norm(self.coords[i] - self.coords[j]))

    @classmethod
    def random(
        cls,
        n: int = 10,
        capacity: float = 50.0,
        seed: int | None = None,
    ) -> CumVRPInstance:
        rng = np.random.default_rng(seed)
        coords = rng.uniform(0, 100, size=(n + 1, 2))
        demands = rng.integers(5, 20, size=n).astype(float)
        return cls(n=n, coords=coords, demands=demands, capacity=capacity,
                   name=f"random_{n}")

    def route_latency(self, route: list[int]) -> float:
        """Sum of arrival times at customers in a route."""
        if not route:
            return 0.0
        total_latency = 0.0
        time = 0.0
        prev = 0
        for c in route:
            time += self.distance(prev, c)
            total_latency += time
            prev = c
        return total_latency


@dataclass
class CumVRPSolution:
    """Cumulative VRP solution.

    Attributes:
        routes: List of routes.
        total_latency: Sum of all customer arrival times.
    """

    routes: list[list[int]]
    total_latency: float

    def __repr__(self) -> str:
        return f"CumVRPSolution(routes={len(self.routes)}, latency={self.total_latency:.1f})"


def validate_solution(
    instance: CumVRPInstance, solution: CumVRPSolution
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
    actual = sum(instance.route_latency(r) for r in solution.routes)
    if abs(actual - solution.total_latency) > 1e-2:
        errors.append(f"Latency: {solution.total_latency:.2f} != {actual:.2f}")
    return len(errors) == 0, errors


def small_cumvrp_6() -> CumVRPInstance:
    return CumVRPInstance(
        n=6,
        coords=np.array([
            [50, 50],
            [20, 70], [80, 70], [20, 30],
            [80, 30], [50, 90], [50, 10],
        ], dtype=float),
        demands=np.array([10, 8, 12, 7, 15, 9], dtype=float),
        capacity=30.0,
        name="small_6",
    )


if __name__ == "__main__":
    inst = small_cumvrp_6()
    print(f"{inst.name}: n={inst.n}")
