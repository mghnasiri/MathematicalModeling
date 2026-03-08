"""
Uncapacitated Facility Location Problem (UFLP) — Instance and Solution.

Problem notation: UFLP

Given a set of m potential facility sites and n customers, each facility
i has a fixed opening cost f_i, and serving customer j from facility i
costs c_ij. Select a subset of facilities to open and assign each
customer to an open facility to minimize total fixed + assignment cost.

Complexity: NP-hard (Cornuéjols, Nemhauser & Wolsey, 1990).
Best known approximation: 1.488 (Li, 2013).

References:
    Cornuéjols, G., Nemhauser, G.L. & Wolsey, L.A. (1990). The
    uncapacitated facility location problem. In: Mirchandani, P.B.
    & Francis, R.L. (eds) Discrete Location Theory, Wiley, 119-171.

    Li, S. (2013). A 1.488 approximation algorithm for the
    uncapacitated facility location problem. Information and
    Computation, 222, 45-58.
    https://doi.org/10.1016/j.ic.2012.01.007

    Krarup, J. & Pruzan, P.M. (1983). The simple plant location
    problem: Survey and synthesis. European Journal of Operational
    Research, 12(1), 36-81.
    https://doi.org/10.1016/0377-2217(83)90181-9
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class FacilityLocationInstance:
    """Uncapacitated Facility Location Problem instance.

    Attributes:
        m: Number of potential facility sites.
        n: Number of customers.
        fixed_costs: Array of facility opening costs, shape (m,).
        assignment_costs: Cost matrix c_ij, shape (m, n).
            c_ij = cost of serving customer j from facility i.
        coords_facilities: Optional (m, 2) facility coordinates.
        coords_customers: Optional (n, 2) customer coordinates.
        name: Optional instance name.
    """

    m: int
    n: int
    fixed_costs: np.ndarray
    assignment_costs: np.ndarray
    coords_facilities: np.ndarray | None = None
    coords_customers: np.ndarray | None = None
    name: str = ""

    def __post_init__(self):
        self.fixed_costs = np.asarray(self.fixed_costs, dtype=float)
        self.assignment_costs = np.asarray(self.assignment_costs, dtype=float)

        if self.fixed_costs.shape != (self.m,):
            raise ValueError(
                f"fixed_costs shape {self.fixed_costs.shape} != ({self.m},)"
            )
        if self.assignment_costs.shape != (self.m, self.n):
            raise ValueError(
                f"assignment_costs shape {self.assignment_costs.shape} "
                f"!= ({self.m}, {self.n})"
            )

    @classmethod
    def random(
        cls,
        m: int,
        n: int,
        fixed_cost_range: tuple[float, float] = (100.0, 500.0),
        coord_range: tuple[float, float] = (0.0, 100.0),
        seed: int | None = None,
    ) -> FacilityLocationInstance:
        """Generate a random UFLP instance with Euclidean distances.

        Args:
            m: Number of facilities.
            n: Number of customers.
            fixed_cost_range: Range for fixed opening costs.
            coord_range: Range for random coordinates.
            seed: Random seed.

        Returns:
            A random FacilityLocationInstance.
        """
        rng = np.random.default_rng(seed)
        fac_coords = rng.uniform(coord_range[0], coord_range[1], size=(m, 2))
        cust_coords = rng.uniform(coord_range[0], coord_range[1], size=(n, 2))

        fixed_costs = np.round(
            rng.uniform(fixed_cost_range[0], fixed_cost_range[1], size=m)
        ).astype(float)

        # Assignment costs = Euclidean distances
        assignment_costs = np.sqrt(
            np.sum(
                (fac_coords[:, None, :] - cust_coords[None, :, :]) ** 2,
                axis=2,
            )
        )

        return cls(
            m=m, n=n,
            fixed_costs=fixed_costs,
            assignment_costs=assignment_costs,
            coords_facilities=fac_coords,
            coords_customers=cust_coords,
            name=f"random_{m}_{n}",
        )

    def total_cost(
        self, open_facilities: list[int], assignments: list[int]
    ) -> float:
        """Compute total cost (fixed + assignment).

        Args:
            open_facilities: List of opened facility indices.
            assignments: List of length n, assignments[j] = facility index for customer j.

        Returns:
            Total cost.
        """
        fixed = sum(self.fixed_costs[i] for i in open_facilities)
        assign = sum(
            self.assignment_costs[assignments[j]][j]
            for j in range(self.n)
        )
        return fixed + assign


@dataclass
class FacilityLocationSolution:
    """Solution to a UFLP instance.

    Attributes:
        open_facilities: List of opened facility indices.
        assignments: List of length n — assignments[j] = facility for customer j.
        cost: Total cost (fixed + assignment).
    """

    open_facilities: list[int]
    assignments: list[int]
    cost: float

    def __repr__(self) -> str:
        return (
            f"FacilityLocationSolution(cost={self.cost:.1f}, "
            f"open={self.open_facilities})"
        )


def validate_solution(
    instance: FacilityLocationInstance,
    solution: FacilityLocationSolution,
) -> tuple[bool, list[str]]:
    """Validate a UFLP solution."""
    errors = []

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
            errors.append(f"Customer {j} assigned to closed facility {fac}")

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


def small_uflp_3_5() -> FacilityLocationInstance:
    """3 facilities, 5 customers.

    Facility costs: [100, 150, 200].
    Assignment costs based on simple grid layout.
    """
    assignment_costs = np.array([
        [10.0, 20.0, 30.0, 40.0, 50.0],   # facility 0
        [25.0, 15.0, 10.0, 20.0, 30.0],   # facility 1
        [40.0, 35.0, 20.0, 10.0, 5.0],    # facility 2
    ])
    return FacilityLocationInstance(
        m=3, n=5,
        fixed_costs=np.array([100.0, 150.0, 200.0]),
        assignment_costs=assignment_costs,
        name="small_3_5",
    )


def medium_uflp_5_10() -> FacilityLocationInstance:
    """5 facilities, 10 customers — generated with seed for reproducibility."""
    return FacilityLocationInstance.random(5, 10, seed=42)


if __name__ == "__main__":
    inst = small_uflp_3_5()
    print(f"{inst.name}: m={inst.m}, n={inst.n}")
    print(f"  fixed costs: {inst.fixed_costs}")
    print(f"  assignment costs:\n{inst.assignment_costs}")
