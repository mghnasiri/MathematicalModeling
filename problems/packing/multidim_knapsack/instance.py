"""
Multi-dimensional Knapsack Problem (MdKP) — Instance and Solution definitions.

Problem notation: MdKP

Given n items, each consuming d resources (weight_id for item i in
dimension d), with values v_i, and d capacity constraints C_d, select
a subset of items to maximize total value such that for each resource
dimension the total consumption does not exceed capacity.

Complexity: NP-hard (Garey & Johnson, 1979). Harder than 0-1 KP due to
multiple constraints.

References:
    Fréville, A. (2004). The multidimensional 0-1 knapsack problem:
    An overview. European Journal of Operational Research, 155(1), 1-21.
    https://doi.org/10.1016/S0377-2217(03)00274-1

    Chu, P.C. & Beasley, J.E. (1998). A genetic algorithm for the
    multidimensional knapsack problem. Journal of Heuristics, 4(1), 63-86.
    https://doi.org/10.1023/A:1009642405419

    Pirkul, H. (1987). A heuristic solution procedure for the multiconstraint
    zero-one knapsack problem. Naval Research Logistics, 34(2), 161-172.
    https://doi.org/10.1002/1520-6750(198704)34:2<161::AID-NAV3220340203>3.0.CO;2-A
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class MultidimKnapsackInstance:
    """Multi-dimensional Knapsack Problem instance.

    Attributes:
        n: Number of items.
        d: Number of resource dimensions.
        weights: Resource consumption matrix, shape (n, d).
                 weights[i, k] is consumption of item i in dimension k.
        values: Array of item values, shape (n,).
        capacities: Array of resource capacities, shape (d,).
        name: Optional instance name.
    """

    n: int
    d: int
    weights: np.ndarray
    values: np.ndarray
    capacities: np.ndarray
    name: str = ""

    def __post_init__(self):
        self.weights = np.asarray(self.weights, dtype=float)
        self.values = np.asarray(self.values, dtype=float)
        self.capacities = np.asarray(self.capacities, dtype=float)

        if self.weights.shape != (self.n, self.d):
            raise ValueError(
                f"weights shape {self.weights.shape} != ({self.n}, {self.d})"
            )
        if self.values.shape != (self.n,):
            raise ValueError(
                f"values shape {self.values.shape} != ({self.n},)"
            )
        if self.capacities.shape != (self.d,):
            raise ValueError(
                f"capacities shape {self.capacities.shape} != ({self.d},)"
            )
        if np.any(self.weights < 0):
            raise ValueError("All weights must be non-negative")
        if np.any(self.values < 0):
            raise ValueError("All item values must be non-negative")
        if np.any(self.capacities <= 0):
            raise ValueError("All capacities must be positive")

    @classmethod
    def random(
        cls,
        n: int,
        d: int = 3,
        capacity_ratio: float = 0.5,
        weight_range: tuple[float, float] = (1.0, 20.0),
        value_range: tuple[float, float] = (1.0, 50.0),
        seed: int | None = None,
    ) -> MultidimKnapsackInstance:
        """Generate a random Multi-dimensional Knapsack instance.

        Args:
            n: Number of items.
            d: Number of resource dimensions.
            capacity_ratio: Fraction of total weight per dimension for capacity.
            weight_range: Range for random item weights.
            value_range: Range for random item values.
            seed: Random seed for reproducibility.

        Returns:
            A random MultidimKnapsackInstance.
        """
        rng = np.random.default_rng(seed)
        weights = np.round(
            rng.uniform(weight_range[0], weight_range[1], size=(n, d))
        ).astype(float)
        values = np.round(
            rng.uniform(value_range[0], value_range[1], size=n)
        ).astype(float)

        capacities = np.round(
            weights.sum(axis=0) * capacity_ratio
        ).astype(float)
        # Ensure capacities are at least max single-item weight per dim
        for k in range(d):
            capacities[k] = max(capacities[k], float(np.max(weights[:, k])))

        return cls(
            n=n, d=d, weights=weights, values=values,
            capacities=capacities, name=f"random_{n}_{d}",
        )

    def is_feasible(self, selection: list[int]) -> bool:
        """Check if a selection of items is feasible.

        Args:
            selection: List of item indices.

        Returns:
            True if total consumption in every dimension is within capacity.
        """
        if len(selection) == 0:
            return True
        total = self.weights[selection].sum(axis=0)
        return bool(np.all(total <= self.capacities + 1e-10))

    def total_value(self, selection: list[int]) -> float:
        """Compute total value of selected items."""
        if len(selection) == 0:
            return 0.0
        return float(np.sum(self.values[selection]))

    def resource_usage(self, selection: list[int]) -> np.ndarray:
        """Compute resource consumption per dimension.

        Args:
            selection: List of item indices.

        Returns:
            Array of shape (d,) with total consumption per dimension.
        """
        if len(selection) == 0:
            return np.zeros(self.d)
        return self.weights[selection].sum(axis=0)


@dataclass
class MultidimKnapsackSolution:
    """Solution to a Multi-dimensional Knapsack instance.

    Attributes:
        items: List of selected item indices (0-indexed).
        value: Total value of selected items.
    """

    items: list[int]
    value: float

    def __repr__(self) -> str:
        return (
            f"MultidimKnapsackSolution(value={self.value:.1f}, "
            f"n_items={len(self.items)}, items={self.items})"
        )


def validate_solution(
    instance: MultidimKnapsackInstance, solution: MultidimKnapsackSolution
) -> tuple[bool, list[str]]:
    """Validate a multi-dimensional knapsack solution.

    Args:
        instance: The MdKP instance.
        solution: The solution to validate.

    Returns:
        Tuple of (is_valid, list of error messages).
    """
    errors = []

    # Check for invalid indices
    for idx in solution.items:
        if idx < 0 or idx >= instance.n:
            errors.append(f"Invalid item index: {idx}")

    # Check duplicates
    if len(solution.items) != len(set(solution.items)):
        errors.append("Duplicate items selected")

    if errors:
        return False, errors

    # Check capacity constraints per dimension
    usage = instance.resource_usage(solution.items)
    for k in range(instance.d):
        if usage[k] > instance.capacities[k] + 1e-10:
            errors.append(
                f"Dimension {k}: usage {usage[k]:.1f} > "
                f"capacity {instance.capacities[k]:.1f}"
            )

    # Check value matches
    actual_value = instance.total_value(solution.items)
    if abs(actual_value - solution.value) > 1e-6:
        errors.append(
            f"Reported value {solution.value:.1f} != actual {actual_value:.1f}"
        )

    return len(errors) == 0, errors


# ── Benchmark instances ──────────────────────────────────────────────────────


def small_mdkp_5_2() -> MultidimKnapsackInstance:
    """5 items, 2 resource dimensions.

    Items: values=[10, 20, 15, 25, 12]
    Weights dim0: [3, 5, 4, 7, 2]
    Weights dim1: [4, 3, 6, 5, 3]
    Capacities: [12, 10]
    """
    return MultidimKnapsackInstance(
        n=5, d=2,
        weights=np.array([
            [3.0, 4.0],
            [5.0, 3.0],
            [4.0, 6.0],
            [7.0, 5.0],
            [2.0, 3.0],
        ]),
        values=np.array([10.0, 20.0, 15.0, 25.0, 12.0]),
        capacities=np.array([12.0, 10.0]),
        name="small_5_2",
    )


def medium_mdkp_8_3() -> MultidimKnapsackInstance:
    """8 items, 3 resource dimensions."""
    return MultidimKnapsackInstance(
        n=8, d=3,
        weights=np.array([
            [3.0, 5.0, 2.0],
            [6.0, 4.0, 3.0],
            [4.0, 2.0, 5.0],
            [5.0, 6.0, 4.0],
            [2.0, 3.0, 1.0],
            [7.0, 1.0, 6.0],
            [3.0, 4.0, 3.0],
            [4.0, 5.0, 2.0],
        ]),
        values=np.array([12.0, 18.0, 15.0, 22.0, 8.0, 20.0, 10.0, 16.0]),
        capacities=np.array([15.0, 14.0, 12.0]),
        name="medium_8_3",
    )


if __name__ == "__main__":
    for name, fn in [("small_5_2", small_mdkp_5_2),
                      ("medium_8_3", medium_mdkp_8_3)]:
        inst = fn()
        print(f"{name}: n={inst.n}, d={inst.d}, C={inst.capacities}")
