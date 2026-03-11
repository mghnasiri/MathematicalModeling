"""
Prize-Collecting TSP (PCTSP) — Instance and Solution.

Given n cities with prizes p_i and travel costs d_ij, select a subset
of cities to visit and find a tour through them. Minimize travel cost
minus collected prizes (or equivalently, maximize prizes minus travel cost).
A minimum prize threshold may be imposed.

Complexity: NP-hard (generalizes TSP when all cities must be visited).

References:
    Balas, E. (1989). The prize collecting traveling salesman problem.
    Networks, 19(6), 621-636.
    https://doi.org/10.1002/net.3230190602

    Goemans, M.X. & Williamson, D.P. (1995). A general approximation
    technique for constrained forest problems. SIAM Journal on Computing,
    24(2), 296-317. https://doi.org/10.1137/S0097539793242618
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class PCTSPInstance:
    """Prize-Collecting TSP instance.

    Attributes:
        n: Number of cities.
        coords: City coordinates, shape (n, 2).
        prizes: Prize at each city, shape (n,).
        min_prize: Minimum total prize to collect (0 = no constraint).
        name: Optional instance name.
    """

    n: int
    coords: np.ndarray
    prizes: np.ndarray
    min_prize: float = 0.0
    name: str = ""

    def __post_init__(self):
        self.coords = np.asarray(self.coords, dtype=float)
        self.prizes = np.asarray(self.prizes, dtype=float)

    def distance(self, i: int, j: int) -> float:
        return float(np.linalg.norm(self.coords[i] - self.coords[j]))

    def distance_matrix(self) -> np.ndarray:
        diff = self.coords[:, None, :] - self.coords[None, :, :]
        return np.sqrt((diff ** 2).sum(axis=2))

    @classmethod
    def random(
        cls,
        n: int = 12,
        prize_range: tuple[int, int] = (5, 50),
        min_prize_ratio: float = 0.5,
        seed: int | None = None,
    ) -> PCTSPInstance:
        rng = np.random.default_rng(seed)
        coords = rng.uniform(0, 100, size=(n, 2))
        prizes = rng.integers(prize_range[0], prize_range[1] + 1, size=n).astype(float)
        min_prize = float(prizes.sum() * min_prize_ratio)
        return cls(n=n, coords=coords, prizes=prizes, min_prize=min_prize,
                   name=f"random_{n}")

    def tour_cost(self, tour: list[int]) -> float:
        """Travel cost of a tour (cycle through selected cities)."""
        if len(tour) <= 1:
            return 0.0
        cost = 0.0
        for i in range(len(tour)):
            cost += self.distance(tour[i], tour[(i + 1) % len(tour)])
        return cost

    def tour_prize(self, tour: list[int]) -> float:
        """Total prize collected on tour."""
        return float(sum(self.prizes[i] for i in tour))

    def objective(self, tour: list[int]) -> float:
        """Objective: travel cost - collected prize (minimize)."""
        return self.tour_cost(tour) - self.tour_prize(tour)


@dataclass
class PCTSPSolution:
    """PCTSP solution.

    Attributes:
        tour: Ordered list of visited city indices.
        travel_cost: Total travel distance.
        total_prize: Total prize collected.
        objective: travel_cost - total_prize.
    """

    tour: list[int]
    travel_cost: float
    total_prize: float
    objective: float

    def __repr__(self) -> str:
        return (f"PCTSPSolution(cities={len(self.tour)}, "
                f"cost={self.travel_cost:.1f}, prize={self.total_prize:.1f})")


def validate_solution(
    instance: PCTSPInstance, solution: PCTSPSolution
) -> tuple[bool, list[str]]:
    errors = []
    if len(set(solution.tour)) != len(solution.tour):
        errors.append("Duplicate cities in tour")
    for c in solution.tour:
        if c < 0 or c >= instance.n:
            errors.append(f"City {c} out of range")
    actual_cost = instance.tour_cost(solution.tour)
    if abs(actual_cost - solution.travel_cost) > 1e-4:
        errors.append(f"Travel cost mismatch: {solution.travel_cost:.2f} != {actual_cost:.2f}")
    actual_prize = instance.tour_prize(solution.tour)
    if abs(actual_prize - solution.total_prize) > 1e-4:
        errors.append(f"Prize mismatch: {solution.total_prize:.2f} != {actual_prize:.2f}")
    if instance.min_prize > 0 and actual_prize < instance.min_prize - 1e-4:
        errors.append(f"Prize {actual_prize:.2f} < minimum {instance.min_prize:.2f}")
    return len(errors) == 0, errors


def small_pctsp_6() -> PCTSPInstance:
    return PCTSPInstance(
        n=6,
        coords=np.array([
            [0, 0], [10, 0], [20, 0],
            [0, 10], [10, 10], [20, 10],
        ], dtype=float),
        prizes=np.array([10, 5, 15, 8, 20, 3], dtype=float),
        min_prize=30.0,
        name="small_6",
    )


if __name__ == "__main__":
    inst = small_pctsp_6()
    print(f"{inst.name}: n={inst.n}, min_prize={inst.min_prize}")
    print(f"  Prizes: {inst.prizes}")
