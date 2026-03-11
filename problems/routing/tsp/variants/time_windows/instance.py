"""
TSP with Time Windows (TSPTW) — Instance and Solution.

Problem notation: TSPTW

Given n cities with pairwise distances and time windows [e_i, l_i],
find the shortest Hamiltonian cycle visiting each city exactly once
such that city i is visited within [e_i, l_i]. If the salesman arrives
before e_i, they wait until e_i.

Applications: scheduling deliveries/pickups within customer time slots,
technician routing, dial-a-ride services, aircraft landing scheduling.

Complexity: NP-hard (generalizes TSP).

References:
    Dumas, Y., Desrosiers, J., Gelinas, E. & Solomon, M.M. (1995).
    An optimal algorithm for the traveling salesman problem with time
    windows. Operations Research, 43(2), 367-371.
    https://doi.org/10.1287/opre.43.2.367

    Gendreau, M., Hertz, A., Laporte, G. & Stan, M. (1998). A
    generalized insertion heuristic for the traveling salesman problem
    with time windows. Operations Research, 46(3), 330-335.
    https://doi.org/10.1287/opre.46.3.330

    Ohlmann, J.W. & Thomas, B.W. (2007). A compressed-annealing
    heuristic for the traveling salesman problem with time windows.
    INFORMS Journal on Computing, 19(1), 80-90.
    https://doi.org/10.1287/ijoc.1050.0145
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class TSPTWInstance:
    """TSP with Time Windows instance.

    Attributes:
        n: Number of cities (including depot at index 0).
        distance_matrix: n x n distance/travel time matrix.
        time_windows: (n, 2) array of [earliest, latest] arrival times.
        service_times: (n,) array of service durations.
        coords: Optional (n, 2) coordinates.
        name: Optional instance name.
    """

    n: int
    distance_matrix: np.ndarray
    time_windows: np.ndarray
    service_times: np.ndarray
    coords: np.ndarray | None = None
    name: str = ""

    def __post_init__(self):
        self.distance_matrix = np.asarray(self.distance_matrix, dtype=float)
        self.time_windows = np.asarray(self.time_windows, dtype=float)
        self.service_times = np.asarray(self.service_times, dtype=float)

        if self.distance_matrix.shape != (self.n, self.n):
            raise ValueError(f"distance_matrix shape != ({self.n}, {self.n})")
        if self.time_windows.shape != (self.n, 2):
            raise ValueError(f"time_windows shape != ({self.n}, 2)")
        if self.service_times.shape != (self.n,):
            raise ValueError(f"service_times shape != ({self.n},)")

    @classmethod
    def random(
        cls,
        n: int,
        horizon: float = 500.0,
        tw_width_range: tuple[float, float] = (30.0, 100.0),
        service_time: float = 10.0,
        coord_range: tuple[float, float] = (0.0, 100.0),
        seed: int | None = None,
    ) -> TSPTWInstance:
        """Generate a random TSPTW instance.

        Args:
            n: Number of cities (including depot).
            horizon: Planning horizon.
            tw_width_range: Range for time window widths.
            service_time: Uniform service time.
            coord_range: Range for coordinates.
            seed: Random seed.

        Returns:
            A random TSPTWInstance.
        """
        rng = np.random.default_rng(seed)
        coords = rng.uniform(coord_range[0], coord_range[1], size=(n, 2))

        dist = np.sqrt(
            np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=2)
        )

        time_windows = np.zeros((n, 2))
        time_windows[0] = [0.0, horizon]  # Depot

        service_times = np.full(n, service_time)
        service_times[0] = 0.0

        for i in range(1, n):
            earliest_possible = dist[0][i]
            tw_width = rng.uniform(tw_width_range[0], tw_width_range[1])
            earliest = rng.uniform(
                earliest_possible,
                max(earliest_possible, horizon - tw_width - service_time),
            )
            latest = earliest + tw_width
            time_windows[i] = [earliest, min(latest, horizon)]

        return cls(
            n=n,
            distance_matrix=dist,
            time_windows=time_windows,
            service_times=service_times,
            coords=coords,
            name=f"random_{n}",
        )

    def tour_distance(self, tour: list[int]) -> float:
        """Total distance of a tour (cyclic)."""
        dist = 0.0
        for i in range(len(tour)):
            dist += self.distance_matrix[tour[i]][tour[(i + 1) % len(tour)]]
        return dist

    def tour_schedule(self, tour: list[int]) -> list[float]:
        """Compute arrival/service start times for each city in tour order."""
        if not tour:
            return []
        times = [max(0.0, self.time_windows[tour[0]][0])]

        for k in range(1, len(tour)):
            prev = tour[k - 1]
            curr = tour[k]
            arrival = times[k - 1] + self.service_times[prev] + self.distance_matrix[prev][curr]
            start = max(arrival, self.time_windows[curr][0])
            times.append(start)

        return times

    def tour_feasible(self, tour: list[int]) -> bool:
        """Check if tour satisfies all time window constraints."""
        if len(tour) != self.n:
            return False
        times = self.tour_schedule(tour)
        for k, city in enumerate(tour):
            if times[k] > self.time_windows[city][1] + 1e-10:
                return False
        # Check return to depot
        last = tour[-1]
        return_time = times[-1] + self.service_times[last] + self.distance_matrix[last][tour[0]]
        if return_time > self.time_windows[tour[0]][1] + 1e-10:
            return False
        return True


@dataclass
class TSPTWSolution:
    """Solution to a TSPTW instance.

    Attributes:
        tour: City visit order (Hamiltonian cycle).
        distance: Total tour distance.
        feasible: Whether all time windows are satisfied.
    """

    tour: list[int]
    distance: float
    feasible: bool = True

    def __repr__(self) -> str:
        return (
            f"TSPTWSolution(distance={self.distance:.2f}, "
            f"feasible={self.feasible}, tour={self.tour[:8]}...)"
        )


def validate_solution(
    instance: TSPTWInstance, solution: TSPTWSolution
) -> tuple[bool, list[str]]:
    """Validate a TSPTW solution."""
    errors = []

    if len(solution.tour) != instance.n:
        errors.append(f"Tour length {len(solution.tour)} != {instance.n}")
        return False, errors

    if set(solution.tour) != set(range(instance.n)):
        errors.append("Tour is not a valid permutation")
        return False, errors

    # Check time windows
    times = instance.tour_schedule(solution.tour)
    for k, city in enumerate(solution.tour):
        if times[k] > instance.time_windows[city][1] + 1e-10:
            errors.append(
                f"City {city}: visit at {times[k]:.1f} > "
                f"latest {instance.time_windows[city][1]:.1f}"
            )

    # Check return
    last = solution.tour[-1]
    first = solution.tour[0]
    return_time = (
        times[-1] + instance.service_times[last]
        + instance.distance_matrix[last][first]
    )
    if return_time > instance.time_windows[first][1] + 1e-10:
        errors.append(
            f"Return to depot at {return_time:.1f} > "
            f"deadline {instance.time_windows[first][1]:.1f}"
        )

    actual_dist = instance.tour_distance(solution.tour)
    if abs(actual_dist - solution.distance) > 1e-4:
        errors.append(
            f"Reported distance {solution.distance:.2f} != "
            f"actual {actual_dist:.2f}"
        )

    return len(errors) == 0, errors


# ── Benchmark instances ──────────────────────────────────────────────────────


def small_tsptw_5() -> TSPTWInstance:
    """5-city TSPTW with tight windows forcing specific ordering."""
    dist = np.array([
        [0, 10, 20, 15, 25],
        [10, 0, 12, 18, 20],
        [20, 12, 0, 10, 15],
        [15, 18, 10, 0, 8],
        [25, 20, 15, 8, 0],
    ], dtype=float)

    time_windows = np.array([
        [0, 200],     # depot (city 0)
        [10, 50],     # city 1 — early
        [40, 80],     # city 2
        [70, 120],    # city 3
        [100, 160],   # city 4
    ], dtype=float)

    return TSPTWInstance(
        n=5,
        distance_matrix=dist,
        time_windows=time_windows,
        service_times=np.array([0, 5, 5, 5, 5], dtype=float),
        name="small_5",
    )


if __name__ == "__main__":
    inst = small_tsptw_5()
    print(f"{inst.name}: {inst.n} cities")
    tour = [0, 1, 2, 3, 4]
    print(f"  Tour {tour}: dist={inst.tour_distance(tour):.1f}, "
          f"feasible={inst.tour_feasible(tour)}")
