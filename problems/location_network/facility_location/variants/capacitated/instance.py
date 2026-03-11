"""
Capacitated Facility Location Problem (CFLP) — Instance and Solution.

Problem notation: CFLP

Extends UFLP with capacity constraints: each facility i has a maximum
capacity u_i limiting the total demand it can serve. Select facilities
to open and assign customers to minimize total fixed + assignment cost,
subject to: sum of demands assigned to facility i <= u_i.

Applications: warehouse location, distribution center planning,
server placement, emergency facility siting.

Complexity: NP-hard (harder than UFLP due to capacity constraints).

References:
    Cornuéjols, G., Sridharan, R. & Thizy, J.M. (1991). A comparison
    of heuristics and relaxations for the capacitated plant location
    problem. European Journal of Operational Research, 50(3), 280-297.
    https://doi.org/10.1016/0377-2217(91)90261-S

    Klose, A. & Drexl, A. (2005). Facility location models for
    distribution system design. European Journal of Operational
    Research, 162(1), 4-29.
    https://doi.org/10.1016/j.ejor.2003.10.031

    Shmoys, D.B., Tardos, É. & Aardal, K. (1997). Approximation
    algorithms for facility location problems. Proceedings of the
    29th ACM STOC, 265-274.
    https://doi.org/10.1145/258533.258600
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class CFLPInstance:
    """Capacitated Facility Location Problem instance.

    Attributes:
        m: Number of potential facility sites.
        n: Number of customers.
        fixed_costs: Facility opening costs, shape (m,).
        assignment_costs: Cost matrix c_ij, shape (m, n).
        capacities: Facility capacities u_i, shape (m,).
        demands: Customer demands d_j, shape (n,).
        name: Optional instance name.
    """

    m: int
    n: int
    fixed_costs: np.ndarray
    assignment_costs: np.ndarray
    capacities: np.ndarray
    demands: np.ndarray
    name: str = ""

    def __post_init__(self):
        self.fixed_costs = np.asarray(self.fixed_costs, dtype=float)
        self.assignment_costs = np.asarray(self.assignment_costs, dtype=float)
        self.capacities = np.asarray(self.capacities, dtype=float)
        self.demands = np.asarray(self.demands, dtype=float)

        if self.fixed_costs.shape != (self.m,):
            raise ValueError(f"fixed_costs shape != ({self.m},)")
        if self.assignment_costs.shape != (self.m, self.n):
            raise ValueError(f"assignment_costs shape != ({self.m}, {self.n})")
        if self.capacities.shape != (self.m,):
            raise ValueError(f"capacities shape != ({self.m},)")
        if self.demands.shape != (self.n,):
            raise ValueError(f"demands shape != ({self.n},)")

    @classmethod
    def random(
        cls,
        m: int,
        n: int,
        fixed_cost_range: tuple[float, float] = (100.0, 500.0),
        demand_range: tuple[float, float] = (5.0, 25.0),
        capacity_factor: float = 2.5,
        coord_range: tuple[float, float] = (0.0, 100.0),
        seed: int | None = None,
    ) -> CFLPInstance:
        """Generate a random CFLP instance.

        Args:
            m: Number of facilities.
            n: Number of customers.
            fixed_cost_range: Range for fixed costs.
            demand_range: Range for demands.
            capacity_factor: Capacity = (total demand / m) * factor.
            coord_range: Range for coordinates.
            seed: Random seed.

        Returns:
            A random CFLPInstance.
        """
        rng = np.random.default_rng(seed)
        fac_coords = rng.uniform(coord_range[0], coord_range[1], size=(m, 2))
        cust_coords = rng.uniform(coord_range[0], coord_range[1], size=(n, 2))

        fixed_costs = np.round(
            rng.uniform(fixed_cost_range[0], fixed_cost_range[1], size=m)
        )
        demands = np.round(
            rng.uniform(demand_range[0], demand_range[1], size=n)
        )
        avg_cap = demands.sum() / m * capacity_factor
        capacities = np.round(
            rng.uniform(avg_cap * 0.7, avg_cap * 1.3, size=m)
        )

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
            capacities=capacities,
            demands=demands,
            name=f"random_{m}_{n}",
        )

    def total_cost(
        self, open_facilities: list[int], assignments: list[int]
    ) -> float:
        """Compute total cost (fixed + assignment)."""
        fixed = sum(self.fixed_costs[i] for i in open_facilities)
        assign = sum(
            self.assignment_costs[assignments[j]][j]
            for j in range(self.n)
        )
        return fixed + assign


@dataclass
class CFLPSolution:
    """Solution to a CFLP instance.

    Attributes:
        open_facilities: List of opened facility indices.
        assignments: assignments[j] = facility for customer j.
        cost: Total cost.
    """

    open_facilities: list[int]
    assignments: list[int]
    cost: float

    def __repr__(self) -> str:
        return (
            f"CFLPSolution(cost={self.cost:.1f}, "
            f"open={self.open_facilities})"
        )


def validate_solution(
    instance: CFLPInstance, solution: CFLPSolution
) -> tuple[bool, list[str]]:
    """Validate a CFLP solution."""
    errors = []
    open_set = set(solution.open_facilities)

    for i in solution.open_facilities:
        if i < 0 or i >= instance.m:
            errors.append(f"Invalid facility index: {i}")

    if len(solution.assignments) != instance.n:
        errors.append(f"Assignments length != {instance.n}")
        return False, errors

    for j, fac in enumerate(solution.assignments):
        if fac not in open_set:
            errors.append(f"Customer {j} assigned to closed facility {fac}")

    if errors:
        return False, errors

    # Check capacity constraints
    load = np.zeros(instance.m)
    for j, fac in enumerate(solution.assignments):
        load[fac] += instance.demands[j]

    for i in open_set:
        if load[i] > instance.capacities[i] + 1e-10:
            errors.append(
                f"Facility {i}: load {load[i]:.1f} > "
                f"capacity {instance.capacities[i]:.1f}"
            )

    actual_cost = instance.total_cost(
        solution.open_facilities, solution.assignments
    )
    if abs(actual_cost - solution.cost) > 1e-4:
        errors.append(
            f"Reported cost {solution.cost:.2f} != actual {actual_cost:.2f}"
        )

    return len(errors) == 0, errors


# ── Benchmark instances ──────────────────────────────────────────────────────


def small_cflp_3_5() -> CFLPInstance:
    """3 facilities, 5 customers with capacity constraints."""
    return CFLPInstance(
        m=3, n=5,
        fixed_costs=np.array([100.0, 150.0, 200.0]),
        assignment_costs=np.array([
            [10.0, 20.0, 30.0, 40.0, 50.0],
            [25.0, 15.0, 10.0, 20.0, 30.0],
            [40.0, 35.0, 20.0, 10.0, 5.0],
        ]),
        capacities=np.array([30.0, 25.0, 35.0]),
        demands=np.array([8.0, 6.0, 10.0, 7.0, 5.0]),
        name="small_3_5",
    )


if __name__ == "__main__":
    inst = small_cflp_3_5()
    print(f"{inst.name}: m={inst.m}, n={inst.n}")
    print(f"  demands: {inst.demands} (total={inst.demands.sum():.0f})")
    print(f"  capacities: {inst.capacities} (total={inst.capacities.sum():.0f})")
