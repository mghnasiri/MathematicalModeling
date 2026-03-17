"""
Traveling Salesman Problem (TSP) — Instance and Solution definitions.

Problem notation: TSP (symmetric) / ATSP (asymmetric)

Given a set of n cities and pairwise distances, find the shortest
Hamiltonian cycle (tour) visiting each city exactly once and returning
to the starting city.

Complexity: NP-hard (Karp, 1972).

References:
    Applegate, D.L., Bixby, R.E., Chvátal, V. & Cook, W.J. (2006).
    The Traveling Salesman Problem: A Computational Study.
    Princeton University Press.
    https://doi.org/10.1515/9781400841103

    Karp, R.M. (1972). Reducibility among combinatorial problems.
    In: Miller, R.E. & Thatcher, J.W. (eds) Complexity of Computer
    Computations, Plenum Press, New York, 85-103.
    https://doi.org/10.1007/978-1-4684-2001-2_9
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class TSPInstance:
    """Traveling Salesman Problem instance.

    Attributes:
        n: Number of cities.
        distance_matrix: n x n matrix of pairwise distances.
            distance_matrix[i][j] = distance from city i to city j.
            For symmetric TSP, distance_matrix[i][j] == distance_matrix[j][i].
        coords: Optional (n, 2) array of city coordinates (for visualization).
        name: Optional instance name.
    """

    n: int
    distance_matrix: np.ndarray
    coords: np.ndarray | None = None
    name: str = ""

    def __post_init__(self):
        self.distance_matrix = np.asarray(self.distance_matrix, dtype=float)
        if self.distance_matrix.shape != (self.n, self.n):
            raise ValueError(
                f"distance_matrix shape {self.distance_matrix.shape} "
                f"does not match n={self.n}"
            )
        if self.coords is not None:
            self.coords = np.asarray(self.coords, dtype=float)
            if self.coords.shape != (self.n, 2):
                raise ValueError(
                    f"coords shape {self.coords.shape} does not match n={self.n}"
                )

    @classmethod
    def random(
        cls,
        n: int,
        symmetric: bool = True,
        coord_range: tuple[float, float] = (0.0, 100.0),
        seed: int | None = None,
    ) -> TSPInstance:
        """Generate a random Euclidean TSP instance.

        Args:
            n: Number of cities.
            symmetric: If True, generate symmetric distances from coordinates.
            coord_range: Range for random coordinates.
            seed: Random seed for reproducibility.

        Returns:
            A random TSPInstance.
        """
        rng = np.random.default_rng(seed)
        coords = rng.uniform(coord_range[0], coord_range[1], size=(n, 2))

        if symmetric:
            dist = np.zeros((n, n))
            for i in range(n):
                for j in range(i + 1, n):
                    d = np.sqrt(np.sum((coords[i] - coords[j]) ** 2))
                    dist[i][j] = d
                    dist[j][i] = d
        else:
            dist = np.sqrt(
                np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=2)
            )
            # Add asymmetry by perturbing
            noise = rng.uniform(0.8, 1.2, size=(n, n))
            dist = dist * noise
            np.fill_diagonal(dist, 0.0)

        return cls(n=n, distance_matrix=dist, coords=coords, name=f"random_{n}")

    @classmethod
    def from_coordinates(
        cls, coords: np.ndarray | list, name: str = ""
    ) -> TSPInstance:
        """Create a symmetric TSP instance from city coordinates.

        Args:
            coords: (n, 2) array of city (x, y) coordinates.
            name: Optional instance name.

        Returns:
            A TSPInstance with Euclidean distances.
        """
        coords = np.asarray(coords, dtype=float)
        n = len(coords)
        dist = np.sqrt(
            np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=2)
        )
        return cls(n=n, distance_matrix=dist, coords=coords, name=name)

    @classmethod
    def from_ors(
        cls,
        locations: list[list[float]] | list[str],
        metric: str = "distance",
        profile: str = "driving-car",
        api_key: str | None = None,
        name: str = "ors_tsp",
    ) -> TSPInstance:
        """Create a TSP instance from real-world locations via OpenRouteService.

        Uses the ORS Matrix API to compute road-network distances or travel
        times between locations, producing a realistic asymmetric TSP instance.

        Args:
            locations: Either a list of [longitude, latitude] pairs, or
                a list of place name strings to geocode.
            metric: 'distance' (meters) or 'duration' (seconds).
            profile: Routing profile ('driving-car', 'driving-hgv',
                'cycling-regular', 'foot-walking').
            api_key: ORS API key. Falls back to ORS_API_KEY env var.
            name: Instance name.

        Returns:
            TSPInstance with real road distances and lon/lat coordinates.

        Example:
            >>> inst = TSPInstance.from_ors([
            ...     [8.681495, 49.41461],   # Heidelberg
            ...     [8.687872, 49.420318],  # Heidelberg Altstadt
            ...     [8.651177, 49.418865],  # Neuenheim
            ... ])
        """
        import sys
        from pathlib import Path
        _root = str(Path(__file__).resolve().parent.parent.parent.parent)
        if _root not in sys.path:
            sys.path.insert(0, _root)
        from shared.api.openrouteservice import ORSClient

        client = ORSClient(api_key=api_key, profile=profile)

        # Geocode if place names provided
        if locations and isinstance(locations[0], str):
            coords = client.geocode_locations(locations)
        else:
            coords = np.asarray(locations, dtype=float)

        # Get road-network distance/duration matrix
        matrix = client.distance_matrix(coords, metric=metric)

        return cls(
            n=len(coords),
            distance_matrix=matrix,
            coords=coords,
            name=name,
        )

    @classmethod
    def from_distance_matrix(
        cls, distance_matrix: np.ndarray | list, name: str = ""
    ) -> TSPInstance:
        """Create a TSP instance from a distance matrix.

        Args:
            distance_matrix: n x n matrix of pairwise distances.
            name: Optional instance name.

        Returns:
            A TSPInstance.
        """
        distance_matrix = np.asarray(distance_matrix, dtype=float)
        n = distance_matrix.shape[0]
        return cls(n=n, distance_matrix=distance_matrix, name=name)

    def is_symmetric(self) -> bool:
        """Check if the distance matrix is symmetric."""
        return np.allclose(self.distance_matrix, self.distance_matrix.T)

    def tour_distance(self, tour: list[int]) -> float:
        """Compute the total distance of a tour.

        Args:
            tour: List of city indices forming a Hamiltonian cycle.
                  The tour implicitly returns to tour[0] after tour[-1].

        Returns:
            Total tour distance.
        """
        dist = 0.0
        for i in range(len(tour)):
            dist += self.distance_matrix[tour[i]][tour[(i + 1) % len(tour)]]
        return dist


@dataclass
class TSPSolution:
    """Solution to a TSP instance.

    Attributes:
        tour: List of city indices forming the tour (Hamiltonian cycle).
        distance: Total tour distance.
    """

    tour: list[int]
    distance: float

    def __repr__(self) -> str:
        return (
            f"TSPSolution(distance={self.distance:.2f}, "
            f"tour={self.tour[:10]}{'...' if len(self.tour) > 10 else ''})"
        )


def validate_tour(instance: TSPInstance, tour: list[int]) -> tuple[bool, list[str]]:
    """Validate that a tour is a valid Hamiltonian cycle.

    Args:
        instance: The TSP instance.
        tour: The tour to validate.

    Returns:
        Tuple of (is_valid, list of error messages).
    """
    errors = []

    if len(tour) != instance.n:
        errors.append(
            f"Tour length {len(tour)} does not match n={instance.n}"
        )
        return False, errors

    if set(tour) != set(range(instance.n)):
        missing = set(range(instance.n)) - set(tour)
        extra = set(tour) - set(range(instance.n))
        if missing:
            errors.append(f"Missing cities: {missing}")
        if extra:
            errors.append(f"Invalid cities: {extra}")
        return False, errors

    return True, errors


# ── Benchmark instances ──────────────────────────────────────────────────────


def small4() -> TSPInstance:
    """4-city instance with known optimal tour.

    Optimal tour: [0, 1, 3, 2] with distance 8.0
    """
    dist = [
        [0, 1, 4, 3],
        [1, 0, 2, 3],
        [4, 2, 0, 1],
        [3, 3, 1, 0],
    ]
    return TSPInstance.from_distance_matrix(dist, name="small4")


def small5() -> TSPInstance:
    """5-city instance with known optimal tour.

    Optimal tour: [0, 1, 2, 4, 3] with distance 19.0
    """
    dist = [
        [0, 3, 4, 2, 7],
        [3, 0, 4, 6, 3],
        [4, 4, 0, 5, 8],
        [2, 6, 5, 0, 6],
        [7, 3, 8, 6, 0],
    ]
    return TSPInstance.from_distance_matrix(dist, name="small5")


def gr17() -> TSPInstance:
    """17-city instance from TSPLIB (Groetschel). Optimal: 2016."""
    dist = [
        [0, 633, 257, 91, 412, 150, 80, 134, 259, 505, 353, 324, 70, 211, 268, 246, 121],
        [633, 0, 390, 661, 227, 488, 572, 530, 555, 289, 282, 638, 567, 466, 420, 745, 518],
        [257, 390, 0, 228, 169, 112, 196, 154, 372, 262, 110, 437, 191, 74, 53, 472, 142],
        [91, 661, 228, 0, 383, 120, 77, 105, 175, 476, 324, 240, 27, 182, 239, 165, 92],
        [412, 227, 169, 383, 0, 267, 351, 309, 338, 196, 61, 421, 346, 243, 199, 528, 297],
        [150, 488, 112, 120, 267, 0, 63, 34, 264, 360, 208, 329, 83, 105, 123, 364, 35],
        [80, 572, 196, 77, 351, 63, 0, 29, 232, 444, 292, 297, 47, 150, 207, 332, 29],
        [134, 530, 154, 105, 309, 34, 29, 0, 249, 402, 250, 314, 68, 108, 165, 349, 36],
        [259, 555, 372, 175, 338, 264, 232, 249, 0, 495, 352, 95, 189, 326, 383, 202, 236],
        [505, 289, 262, 476, 196, 360, 444, 402, 495, 0, 154, 578, 439, 336, 240, 685, 390],
        [353, 282, 110, 324, 61, 208, 292, 250, 352, 154, 0, 435, 287, 184, 140, 542, 238],
        [324, 638, 437, 240, 421, 329, 297, 314, 95, 578, 435, 0, 254, 391, 448, 157, 301],
        [70, 567, 191, 27, 346, 83, 47, 68, 189, 439, 287, 254, 0, 145, 202, 289, 55],
        [211, 466, 74, 182, 243, 105, 150, 108, 326, 336, 184, 391, 145, 0, 57, 426, 96],
        [268, 420, 53, 239, 199, 123, 207, 165, 383, 240, 140, 448, 202, 57, 0, 483, 153],
        [246, 745, 472, 165, 528, 364, 332, 349, 202, 685, 542, 157, 289, 426, 483, 0, 336],
        [121, 518, 142, 92, 297, 35, 29, 36, 236, 390, 238, 301, 55, 96, 153, 336, 0],
    ]
    return TSPInstance.from_distance_matrix(dist, name="gr17")


if __name__ == "__main__":
    # Example: create a random instance
    inst = TSPInstance.random(10, seed=42)
    print(f"Random TSP: {inst.n} cities, symmetric={inst.is_symmetric()}")
    print(f"Distance matrix shape: {inst.distance_matrix.shape}")

    # Benchmark instances
    s4 = small4()
    print(f"\nsmall4: optimal tour distance = {s4.tour_distance([0, 1, 3, 2])}")

    s5 = small5()
    print(f"small5: optimal tour distance = {s5.tour_distance([0, 1, 2, 4, 3])}")

    g17 = gr17()
    print(f"gr17: {g17.n} cities, symmetric={g17.is_symmetric()}")
