"""
p-Median Problem — Instance and Solution definitions.

Problem notation: PMP (p-Median Problem)

Given n demand points (customers) and m candidate locations, open
exactly p facilities and assign each customer to its nearest open
facility to minimize total weighted distance.

Special case of UFLP where the number of open facilities is fixed
at p and there are no fixed opening costs.

Complexity: NP-hard for general p (Kariv & Hakimi, 1979).

References:
    Hakimi, S.L. (1964). Optimum locations of switching centers and
    the absolute centers and medians of a graph. Operations Research,
    12(3), 450-459.
    https://doi.org/10.1287/opre.12.3.450

    Kariv, O. & Hakimi, S.L. (1979). An algorithmic approach to
    network location problems. Part I: The p-centers. SIAM Journal
    on Applied Mathematics, 37(3), 513-538.
    https://doi.org/10.1137/0137040

    Reese, J. (2006). Solution methods for the p-median problem:
    An annotated bibliography. Networks, 48(3), 125-142.
    https://doi.org/10.1002/net.20128
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class PMedianInstance:
    """p-Median problem instance.

    Attributes:
        n: Number of demand points (customers).
        m: Number of candidate facility locations.
        p: Number of facilities to open.
        weights: Customer demand weights, shape (n,).
        distance_matrix: Distance from facility i to customer j, shape (m, n).
        coords_facilities: Optional (m, 2) coordinates.
        coords_customers: Optional (n, 2) coordinates.
        name: Optional instance name.
    """

    n: int
    m: int
    p: int
    weights: np.ndarray
    distance_matrix: np.ndarray
    coords_facilities: np.ndarray | None = None
    coords_customers: np.ndarray | None = None
    name: str = ""

    def __post_init__(self):
        self.weights = np.asarray(self.weights, dtype=float)
        self.distance_matrix = np.asarray(self.distance_matrix, dtype=float)

        if self.weights.shape != (self.n,):
            raise ValueError(
                f"weights shape {self.weights.shape} != ({self.n},)"
            )
        if self.distance_matrix.shape != (self.m, self.n):
            raise ValueError(
                f"distance_matrix shape {self.distance_matrix.shape} "
                f"!= ({self.m}, {self.n})"
            )
        if self.p < 1 or self.p > self.m:
            raise ValueError(f"p={self.p} must be in [1, m={self.m}]")

    @classmethod
    def random(
        cls,
        n: int,
        m: int | None = None,
        p: int = 3,
        coord_range: tuple[float, float] = (0.0, 100.0),
        seed: int | None = None,
    ) -> PMedianInstance:
        """Generate a random p-Median instance with Euclidean distances.

        Args:
            n: Number of customers.
            m: Number of candidate sites. If None, equals n.
            p: Number of facilities to open.
            coord_range: Range for coordinates.
            seed: Random seed.

        Returns:
            A random PMedianInstance.
        """
        if m is None:
            m = n
        rng = np.random.default_rng(seed)

        fac_coords = rng.uniform(coord_range[0], coord_range[1], size=(m, 2))
        cust_coords = rng.uniform(coord_range[0], coord_range[1], size=(n, 2))
        weights = np.ones(n)

        dist = np.sqrt(
            np.sum(
                (fac_coords[:, None, :] - cust_coords[None, :, :]) ** 2,
                axis=2,
            )
        )

        return cls(
            n=n, m=m, p=p,
            weights=weights,
            distance_matrix=dist,
            coords_facilities=fac_coords,
            coords_customers=cust_coords,
            name=f"random_{m}_{n}_p{p}",
        )

    @classmethod
    def from_ors(
        cls,
        facilities: list[list[float]] | list[str],
        customers: list[list[float]] | list[str],
        p: int,
        weights: list[float] | np.ndarray | None = None,
        metric: str = "distance",
        profile: str = "driving-car",
        api_key: str | None = None,
        name: str = "ors_pmedian",
    ) -> PMedianInstance:
        """Create a p-Median instance from real-world locations via ORS.

        Args:
            facilities: List of [lon, lat] or place names for candidates.
            customers: List of [lon, lat] or place names for demand points.
            p: Number of facilities to open.
            weights: Customer demand weights. If None, all equal to 1.
            metric: 'distance' (meters) or 'duration' (seconds).
            profile: Routing profile.
            api_key: ORS API key.
            name: Instance name.

        Returns:
            PMedianInstance with real road distances.
        """
        import sys
        from pathlib import Path
        _root = str(Path(__file__).resolve().parent.parent.parent.parent)
        if _root not in sys.path:
            sys.path.insert(0, _root)
        from shared.api.openrouteservice import ORSClient

        client = ORSClient(api_key=api_key, profile=profile)

        if facilities and isinstance(facilities[0], str):
            fac_coords = client.geocode_locations(facilities)
        else:
            fac_coords = np.asarray(facilities, dtype=float)

        if customers and isinstance(customers[0], str):
            cust_coords = client.geocode_locations(customers)
        else:
            cust_coords = np.asarray(customers, dtype=float)

        m = len(fac_coords)
        n = len(cust_coords)

        all_coords = np.vstack([fac_coords, cust_coords])
        result = client.matrix(
            all_coords.tolist(),
            metrics=[metric],
            sources=list(range(m)),
            destinations=list(range(m, m + n)),
        )
        key = "distances" if metric == "distance" else "durations"
        dist_matrix = result[key]

        if weights is None:
            weights = np.ones(n)

        return cls(
            n=n,
            m=m,
            p=p,
            weights=np.asarray(weights, dtype=float),
            distance_matrix=dist_matrix,
            coords_facilities=fac_coords,
            coords_customers=cust_coords,
            name=name,
        )

    def total_cost(
        self, open_facilities: list[int], assignments: list[int]
    ) -> float:
        """Compute total weighted distance.

        Args:
            open_facilities: List of opened facility indices.
            assignments: assignments[j] = facility index for customer j.

        Returns:
            Total weighted distance.
        """
        return sum(
            self.weights[j] * self.distance_matrix[assignments[j]][j]
            for j in range(self.n)
        )


@dataclass
class PMedianSolution:
    """Solution to a p-Median problem.

    Attributes:
        open_facilities: List of opened facility indices (length p).
        assignments: assignments[j] = facility for customer j.
        cost: Total weighted distance.
    """

    open_facilities: list[int]
    assignments: list[int]
    cost: float

    def __repr__(self) -> str:
        return (
            f"PMedianSolution(cost={self.cost:.1f}, "
            f"open={self.open_facilities})"
        )


def validate_solution(
    instance: PMedianInstance, solution: PMedianSolution
) -> tuple[bool, list[str]]:
    """Validate a p-Median solution."""
    errors = []

    if len(solution.open_facilities) != instance.p:
        errors.append(
            f"Opened {len(solution.open_facilities)} facilities, expected {instance.p}"
        )

    open_set = set(solution.open_facilities)
    for i in solution.open_facilities:
        if i < 0 or i >= instance.m:
            errors.append(f"Invalid facility index: {i}")

    if len(solution.assignments) != instance.n:
        errors.append(
            f"Assignments length {len(solution.assignments)} != {instance.n}"
        )
        return False, errors

    for j, fac in enumerate(solution.assignments):
        if fac not in open_set:
            errors.append(f"Customer {j} assigned to non-open facility {fac}")

    if errors:
        return False, errors

    actual_cost = instance.total_cost(
        solution.open_facilities, solution.assignments
    )
    if abs(actual_cost - solution.cost) > 1e-4:
        errors.append(
            f"Reported cost {solution.cost:.2f} != actual {actual_cost:.2f}"
        )

    return len(errors) == 0, errors


# ── Benchmark instances ──────────────────────────────────────────────────────


def small_pmedian_6_2() -> PMedianInstance:
    """6 locations (= customers), choose p=2 medians."""
    coords = np.array([
        [0, 0], [10, 0], [20, 0],
        [0, 10], [10, 10], [20, 10],
    ], dtype=float)
    n = m = 6
    dist = np.sqrt(
        np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=2)
    )
    return PMedianInstance(
        n=n, m=m, p=2,
        weights=np.ones(n),
        distance_matrix=dist,
        coords_facilities=coords,
        coords_customers=coords,
        name="small_6_2",
    )


if __name__ == "__main__":
    inst = small_pmedian_6_2()
    print(f"{inst.name}: n={inst.n}, m={inst.m}, p={inst.p}")
