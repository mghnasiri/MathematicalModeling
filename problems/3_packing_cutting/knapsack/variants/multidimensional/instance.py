"""
Multi-dimensional Knapsack Problem (MKP) — Instance and Solution.

Problem notation: MKP (d-KP)

Given n items, each with value v_i and d-dimensional weight vector
(w_i1, ..., w_id), and a knapsack with d capacity constraints
(W_1, ..., W_d), select a subset to maximize total value subject
to sum(w_ij * x_i) <= W_j for all dimensions j.

The MKP generalizes the 0-1 Knapsack to multiple resource constraints.
NP-hard even for d=1; practically harder as d grows.

Applications: capital budgeting, cargo loading, resource allocation,
project selection with multiple resource constraints.

Complexity: NP-hard (strongly for d >= 2).

References:
    Chu, P.C. & Beasley, J.E. (1998). A genetic algorithm for the
    multidimensional knapsack problem. Journal of Heuristics,
    4(1), 63-86.
    https://doi.org/10.1023/A:1009642405419

    Pirkul, H. (1987). A heuristic solution procedure for the
    multiconstraint zero-one knapsack problem. Naval Research
    Logistics, 34(2), 161-172.
    https://doi.org/10.1002/1520-6750(198704)34:2<161::AID-NAV3220340203>3.0.CO;2-A

    Fréville, A. (2004). The multidimensional 0-1 knapsack problem:
    An overview. European Journal of Operational Research,
    155(1), 1-21.
    https://doi.org/10.1016/S0377-2217(03)00274-1
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class MKPInstance:
    """Multi-dimensional Knapsack Problem instance.

    Attributes:
        n: Number of items.
        d: Number of dimensions (resource constraints).
        values: Array of item values, shape (n,).
        weights: Weight matrix, shape (d, n). weights[j][i] = weight of
            item i in dimension j.
        capacities: Array of capacity constraints, shape (d,).
        name: Optional instance name.
    """

    n: int
    d: int
    values: np.ndarray
    weights: np.ndarray
    capacities: np.ndarray
    name: str = ""

    def __post_init__(self):
        self.values = np.asarray(self.values, dtype=float)
        self.weights = np.asarray(self.weights, dtype=float)
        self.capacities = np.asarray(self.capacities, dtype=float)

        if self.values.shape != (self.n,):
            raise ValueError(f"values shape {self.values.shape} != ({self.n},)")
        if self.weights.shape != (self.d, self.n):
            raise ValueError(
                f"weights shape {self.weights.shape} != ({self.d}, {self.n})"
            )
        if self.capacities.shape != (self.d,):
            raise ValueError(
                f"capacities shape {self.capacities.shape} != ({self.d},)"
            )

    @classmethod
    def random(
        cls,
        n: int,
        d: int = 3,
        capacity_ratio: float = 0.5,
        weight_range: tuple[float, float] = (1.0, 50.0),
        value_range: tuple[float, float] = (1.0, 100.0),
        seed: int | None = None,
    ) -> MKPInstance:
        """Generate a random MKP instance.

        Args:
            n: Number of items.
            d: Number of dimensions.
            capacity_ratio: Capacity as fraction of total weight per dimension.
            weight_range: Range for random weights.
            value_range: Range for random values.
            seed: Random seed.

        Returns:
            A random MKPInstance.
        """
        rng = np.random.default_rng(seed)
        values = np.round(rng.uniform(value_range[0], value_range[1], size=n))
        weights = np.round(
            rng.uniform(weight_range[0], weight_range[1], size=(d, n))
        )
        capacities = np.round(weights.sum(axis=1) * capacity_ratio)

        return cls(
            n=n, d=d,
            values=values,
            weights=weights,
            capacities=capacities,
            name=f"random_{n}_{d}d",
        )

    def total_value(self, selection: list[int]) -> float:
        """Compute total value of selected items."""
        return float(sum(self.values[i] for i in selection))

    def total_weights(self, selection: list[int]) -> np.ndarray:
        """Compute total weight vector of selected items."""
        if not selection:
            return np.zeros(self.d)
        return self.weights[:, selection].sum(axis=1)

    def is_feasible(self, selection: list[int]) -> bool:
        """Check if selection satisfies all capacity constraints."""
        w = self.total_weights(selection)
        return bool(np.all(w <= self.capacities + 1e-10))


@dataclass
class MKPSolution:
    """Solution to a Multi-dimensional Knapsack Problem.

    Attributes:
        items: List of selected item indices.
        value: Total value.
        weights: Total weight per dimension.
    """

    items: list[int]
    value: float
    weights: np.ndarray

    def __repr__(self) -> str:
        return (
            f"MKPSolution(value={self.value:.1f}, "
            f"items={len(self.items)}, dims={len(self.weights)})"
        )


def validate_solution(
    instance: MKPInstance, solution: MKPSolution
) -> tuple[bool, list[str]]:
    """Validate an MKP solution."""
    errors = []

    for idx in solution.items:
        if idx < 0 or idx >= instance.n:
            errors.append(f"Invalid item index: {idx}")
    if len(solution.items) != len(set(solution.items)):
        errors.append("Duplicate items selected")

    if errors:
        return False, errors

    actual_weights = instance.total_weights(solution.items)
    for j in range(instance.d):
        if actual_weights[j] > instance.capacities[j] + 1e-10:
            errors.append(
                f"Dimension {j}: weight {actual_weights[j]:.1f} > "
                f"capacity {instance.capacities[j]:.1f}"
            )

    actual_value = instance.total_value(solution.items)
    if abs(actual_value - solution.value) > 1e-6:
        errors.append(
            f"Reported value {solution.value:.1f} != actual {actual_value:.1f}"
        )

    return len(errors) == 0, errors


# ── Benchmark instances ──────────────────────────────────────────────────────


def small_mkp_5_2() -> MKPInstance:
    """5 items, 2 dimensions. Known optimal = 150 (items [0, 2, 4])."""
    return MKPInstance(
        n=5, d=2,
        values=np.array([60.0, 50.0, 70.0, 30.0, 20.0]),
        weights=np.array([
            [10.0, 20.0, 15.0, 25.0, 5.0],   # dimension 0
            [15.0, 10.0, 20.0, 10.0, 8.0],    # dimension 1
        ]),
        capacities=np.array([35.0, 45.0]),
        name="small_5_2d",
    )


def medium_mkp_10_3() -> MKPInstance:
    """10 items, 3 dimensions — generated with seed."""
    return MKPInstance.random(n=10, d=3, seed=42)


if __name__ == "__main__":
    inst = small_mkp_5_2()
    print(f"{inst.name}: n={inst.n}, d={inst.d}")
    print(f"  values: {inst.values}")
    print(f"  weights:\n{inst.weights}")
    print(f"  capacities: {inst.capacities}")
    print(f"  items [0,2,4]: value={inst.total_value([0,2,4])}, "
          f"feasible={inst.is_feasible([0,2,4])}")
