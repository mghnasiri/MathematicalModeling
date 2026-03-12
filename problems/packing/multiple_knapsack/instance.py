"""
Multiple Knapsack Problem (MKP) — Instance and Solution definitions.

Problem notation: MKP

Given n items with weights w_i and values v_i, and m knapsacks with
capacities C_j, assign items to knapsacks to maximize total value such
that the total weight in each knapsack does not exceed its capacity.
Each item is assigned to at most one knapsack.

Complexity: NP-hard (generalizes 0-1 Knapsack).

References:
    Martello, S. & Toth, P. (1990). Knapsack Problems: Algorithms and
    Computer Implementations. John Wiley & Sons, Chichester.

    Kellerer, H., Pferschy, U. & Pisinger, D. (2004). Knapsack Problems.
    Springer-Verlag, Berlin.
    https://doi.org/10.1007/978-3-540-24777-7

    Pisinger, D. (1999). An exact algorithm for large multiple knapsack
    problems. European Journal of Operational Research, 114(3), 528-541.
    https://doi.org/10.1016/S0377-2217(98)00120-9
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class MultipleKnapsackInstance:
    """Multiple Knapsack Problem instance.

    Attributes:
        n: Number of items.
        m: Number of knapsacks.
        weights: Array of item weights, shape (n,).
        values: Array of item values, shape (n,).
        capacities: Array of knapsack capacities, shape (m,).
        name: Optional instance name.
    """

    n: int
    m: int
    weights: np.ndarray
    values: np.ndarray
    capacities: np.ndarray
    name: str = ""

    def __post_init__(self):
        self.weights = np.asarray(self.weights, dtype=float)
        self.values = np.asarray(self.values, dtype=float)
        self.capacities = np.asarray(self.capacities, dtype=float)

        if self.weights.shape != (self.n,):
            raise ValueError(
                f"weights shape {self.weights.shape} != ({self.n},)"
            )
        if self.values.shape != (self.n,):
            raise ValueError(
                f"values shape {self.values.shape} != ({self.n},)"
            )
        if self.capacities.shape != (self.m,):
            raise ValueError(
                f"capacities shape {self.capacities.shape} != ({self.m},)"
            )
        if np.any(self.weights <= 0):
            raise ValueError("All item weights must be positive")
        if np.any(self.values < 0):
            raise ValueError("All item values must be non-negative")
        if np.any(self.capacities <= 0):
            raise ValueError("All knapsack capacities must be positive")

    @classmethod
    def random(
        cls,
        n: int,
        m: int = 3,
        capacity_range: tuple[float, float] = (30.0, 60.0),
        weight_range: tuple[float, float] = (1.0, 20.0),
        value_range: tuple[float, float] = (1.0, 50.0),
        seed: int | None = None,
    ) -> MultipleKnapsackInstance:
        """Generate a random Multiple Knapsack instance.

        Args:
            n: Number of items.
            m: Number of knapsacks.
            capacity_range: Range for random knapsack capacities.
            weight_range: Range for random item weights.
            value_range: Range for random item values.
            seed: Random seed for reproducibility.

        Returns:
            A random MultipleKnapsackInstance.
        """
        rng = np.random.default_rng(seed)
        weights = np.round(
            rng.uniform(weight_range[0], weight_range[1], size=n)
        ).astype(float)
        values = np.round(
            rng.uniform(value_range[0], value_range[1], size=n)
        ).astype(float)
        capacities = np.round(
            rng.uniform(capacity_range[0], capacity_range[1], size=m)
        ).astype(float)

        return cls(
            n=n, m=m, weights=weights, values=values,
            capacities=capacities, name=f"random_{n}_{m}",
        )

    def total_capacity(self) -> float:
        """Total capacity across all knapsacks."""
        return float(np.sum(self.capacities))

    def total_weight(self) -> float:
        """Total weight of all items."""
        return float(np.sum(self.weights))


@dataclass
class MultipleKnapsackSolution:
    """Solution to a Multiple Knapsack instance.

    Attributes:
        assignments: List of m lists, each containing item indices
                     assigned to that knapsack.
        value: Total value of assigned items.
    """

    assignments: list[list[int]]
    value: float

    def __repr__(self) -> str:
        items_per_ks = [len(a) for a in self.assignments]
        return (
            f"MultipleKnapsackSolution(value={self.value:.1f}, "
            f"items_per_knapsack={items_per_ks})"
        )


def validate_solution(
    instance: MultipleKnapsackInstance, solution: MultipleKnapsackSolution
) -> tuple[bool, list[str]]:
    """Validate a multiple knapsack solution.

    Args:
        instance: The MKP instance.
        solution: The solution to validate.

    Returns:
        Tuple of (is_valid, list of error messages).
    """
    errors = []

    if len(solution.assignments) != instance.m:
        errors.append(
            f"Expected {instance.m} knapsacks, got {len(solution.assignments)}"
        )
        return False, errors

    # Check for duplicates and invalid indices
    all_items = []
    for j, ks in enumerate(solution.assignments):
        for idx in ks:
            if idx < 0 or idx >= instance.n:
                errors.append(f"Knapsack {j}: invalid item index {idx}")
            all_items.append(idx)

    if len(all_items) != len(set(all_items)):
        errors.append("Item assigned to multiple knapsacks")

    if errors:
        return False, errors

    # Check capacity constraints
    actual_value = 0.0
    for j, ks in enumerate(solution.assignments):
        total_w = sum(instance.weights[i] for i in ks)
        if total_w > instance.capacities[j] + 1e-10:
            errors.append(
                f"Knapsack {j}: weight {total_w:.1f} > "
                f"capacity {instance.capacities[j]:.1f}"
            )
        actual_value += sum(instance.values[i] for i in ks)

    # Check value matches
    if abs(actual_value - solution.value) > 1e-6:
        errors.append(
            f"Reported value {solution.value:.1f} != actual {actual_value:.1f}"
        )

    return len(errors) == 0, errors


# ── Benchmark instances ──────────────────────────────────────────────────────


def small_mkp_6_2() -> MultipleKnapsackInstance:
    """6 items, 2 knapsacks.

    Items: w=[3,4,5,2,7,1], v=[10,15,20,8,25,3].
    Knapsacks: C=[10, 8].
    Optimal: ks0=[2,0] (w=8,v=30), ks1=[1,3] (w=6,v=23) => total=53.
    """
    return MultipleKnapsackInstance(
        n=6, m=2,
        weights=np.array([3.0, 4.0, 5.0, 2.0, 7.0, 1.0]),
        values=np.array([10.0, 15.0, 20.0, 8.0, 25.0, 3.0]),
        capacities=np.array([10.0, 8.0]),
        name="small_6_2",
    )


def medium_mkp_8_3() -> MultipleKnapsackInstance:
    """8 items, 3 knapsacks."""
    return MultipleKnapsackInstance(
        n=8, m=3,
        weights=np.array([5.0, 8.0, 3.0, 6.0, 4.0, 7.0, 2.0, 9.0]),
        values=np.array([12.0, 20.0, 8.0, 15.0, 10.0, 18.0, 5.0, 22.0]),
        capacities=np.array([15.0, 12.0, 10.0]),
        name="medium_8_3",
    )


if __name__ == "__main__":
    for name, fn in [("small_6_2", small_mkp_6_2),
                      ("medium_8_3", medium_mkp_8_3)]:
        inst = fn()
        print(f"{name}: n={inst.n}, m={inst.m}, "
              f"C={inst.capacities}, total_cap={inst.total_capacity():.0f}")
