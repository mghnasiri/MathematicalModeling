"""
Pickup and Delivery Problem (PDP) — Instance and Solution.

Problem notation: 1-PDTSP (single vehicle pickup and delivery TSP)

Given n pickup-delivery pairs (2n locations + depot), find the shortest
tour visiting all locations exactly once such that each pickup i is
visited before its corresponding delivery i+n.

Applications: courier services, ride-sharing, package pickup/delivery,
supply chain logistics, dial-a-ride.

Complexity: NP-hard (generalizes TSP).

References:
    Savelsbergh, M.W.P. & Sol, M. (1995). The general pickup and delivery
    problem. Transportation Science, 29(1), 17-29.
    https://doi.org/10.1287/trsc.29.1.17

    Renaud, J., Boctor, F.F. & Laporte, G. (2000). An improved petal
    heuristic for the vehicle routing problem. Journal of the Operational
    Research Society, 51(8), 923-928.
    https://doi.org/10.1057/palgrave.jors.2600988

    Ruland, K.S. & Rodin, E.Y. (1997). The pickup and delivery problem:
    Faces and branch-and-cut algorithm. Computers & Mathematics with
    Applications, 33(12), 1-13.
    https://doi.org/10.1016/S0898-1221(97)00090-2
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class PDPInstance:
    """Pickup and Delivery Problem instance.

    Locations 0 = depot, 1..num_pairs = pickups, num_pairs+1..2*num_pairs = deliveries.
    Pair i: pickup at i, delivery at i + num_pairs.

    Attributes:
        num_pairs: Number of pickup-delivery pairs.
        num_locations: Total locations (2*num_pairs + 1 including depot).
        distance_matrix: (num_locations, num_locations) distance matrix.
        coords: Optional (num_locations, 2) coordinates.
        name: Optional instance name.
    """

    num_pairs: int
    num_locations: int
    distance_matrix: np.ndarray
    coords: np.ndarray | None = None
    name: str = ""

    def __post_init__(self):
        self.distance_matrix = np.asarray(self.distance_matrix, dtype=float)
        if self.distance_matrix.shape != (self.num_locations, self.num_locations):
            raise ValueError(
                f"distance_matrix shape != ({self.num_locations}, {self.num_locations})"
            )

    @classmethod
    def random(
        cls,
        num_pairs: int = 5,
        coord_range: tuple[float, float] = (0.0, 100.0),
        seed: int | None = None,
    ) -> PDPInstance:
        """Generate a random PDP instance.

        Args:
            num_pairs: Number of pickup-delivery pairs.
            coord_range: Range for coordinates.
            seed: Random seed.

        Returns:
            A random PDPInstance.
        """
        rng = np.random.default_rng(seed)
        num_locations = 2 * num_pairs + 1
        coords = rng.uniform(coord_range[0], coord_range[1], size=(num_locations, 2))

        dist = np.sqrt(
            np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=2)
        )

        return cls(
            num_pairs=num_pairs,
            num_locations=num_locations,
            distance_matrix=dist,
            coords=coords,
            name=f"random_{num_pairs}",
        )

    def pickup_of(self, pair: int) -> int:
        """Return pickup location index for pair (1-indexed)."""
        return pair

    def delivery_of(self, pair: int) -> int:
        """Return delivery location index for pair (1-indexed)."""
        return pair + self.num_pairs

    def tour_distance(self, tour: list[int]) -> float:
        """Total distance of a tour (cyclic, returning to depot)."""
        dist = 0.0
        for i in range(len(tour)):
            dist += self.distance_matrix[tour[i]][tour[(i + 1) % len(tour)]]
        return dist

    def precedence_feasible(self, tour: list[int]) -> bool:
        """Check if all pickups precede their deliveries in the tour."""
        pos = {loc: idx for idx, loc in enumerate(tour)}
        for p in range(1, self.num_pairs + 1):
            pickup = p
            delivery = p + self.num_pairs
            if pickup not in pos or delivery not in pos:
                return False
            if pos[pickup] >= pos[delivery]:
                return False
        return True


@dataclass
class PDPSolution:
    """Solution to a PDP instance.

    Attributes:
        tour: Location visit order starting from depot.
        distance: Total tour distance.
        feasible: Whether precedence constraints are satisfied.
    """

    tour: list[int]
    distance: float
    feasible: bool = True

    def __repr__(self) -> str:
        return (
            f"PDPSolution(distance={self.distance:.2f}, "
            f"feasible={self.feasible})"
        )


def validate_solution(
    instance: PDPInstance, solution: PDPSolution
) -> tuple[bool, list[str]]:
    """Validate a PDP solution."""
    errors = []

    if len(solution.tour) != instance.num_locations:
        errors.append(
            f"Tour length {len(solution.tour)} != {instance.num_locations}"
        )
        return False, errors

    if set(solution.tour) != set(range(instance.num_locations)):
        errors.append("Tour is not a valid permutation of all locations")
        return False, errors

    # Check precedence
    pos = {loc: idx for idx, loc in enumerate(solution.tour)}
    for p in range(1, instance.num_pairs + 1):
        pickup = p
        delivery = p + instance.num_pairs
        if pos[pickup] >= pos[delivery]:
            errors.append(
                f"Pair {p}: pickup at pos {pos[pickup]} >= "
                f"delivery at pos {pos[delivery]}"
            )

    actual_dist = instance.tour_distance(solution.tour)
    if abs(actual_dist - solution.distance) > 1e-4:
        errors.append(
            f"Reported distance {solution.distance:.2f} != "
            f"actual {actual_dist:.2f}"
        )

    return len(errors) == 0, errors


# ── Benchmark instances ──────────────────────────────────────────────────────


def small_pdp_3() -> PDPInstance:
    """3 pairs (7 locations including depot)."""
    coords = np.array([
        [50, 50],   # depot
        [20, 80],   # pickup 1
        [80, 80],   # pickup 2
        [50, 20],   # pickup 3
        [30, 30],   # delivery 1
        [70, 20],   # delivery 2
        [80, 50],   # delivery 3
    ], dtype=float)

    n = 7
    dist = np.sqrt(
        np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=2)
    )

    return PDPInstance(
        num_pairs=3,
        num_locations=n,
        distance_matrix=dist,
        coords=coords,
        name="small_3",
    )


if __name__ == "__main__":
    inst = small_pdp_3()
    print(f"{inst.name}: {inst.num_pairs} pairs, {inst.num_locations} locations")
    tour = [0, 1, 2, 3, 4, 5, 6]
    print(f"  Tour {tour}: dist={inst.tour_distance(tour):.1f}, "
          f"prec_feasible={inst.precedence_feasible(tour)}")
