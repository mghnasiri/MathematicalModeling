"""
0-1 Knapsack Problem — Instance and Solution definitions.

Problem notation: KP01

Given n items, each with weight w_i and value v_i, and a knapsack with
capacity W, select a subset of items to maximize total value without
exceeding the weight capacity.

Complexity: NP-hard (Karp, 1972), weakly NP-hard (admits pseudo-polynomial DP).

References:
    Karp, R.M. (1972). Reducibility among combinatorial problems.
    In: Miller, R.E. & Thatcher, J.W. (eds) Complexity of Computer
    Computations, Plenum Press, New York, 85-103.
    https://doi.org/10.1007/978-1-4684-2001-2_9

    Kellerer, H., Pferschy, U. & Pisinger, D. (2004). Knapsack Problems.
    Springer-Verlag, Berlin.
    https://doi.org/10.1007/978-3-540-24777-7

    Martello, S. & Toth, P. (1990). Knapsack Problems: Algorithms and
    Computer Implementations. John Wiley & Sons, Chichester.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class KnapsackInstance:
    """0-1 Knapsack problem instance.

    Attributes:
        n: Number of items.
        weights: Array of item weights, shape (n,).
        values: Array of item values, shape (n,).
        capacity: Knapsack capacity W.
        name: Optional instance name.
    """

    n: int
    weights: np.ndarray
    values: np.ndarray
    capacity: float
    name: str = ""

    def __post_init__(self):
        self.weights = np.asarray(self.weights, dtype=float)
        self.values = np.asarray(self.values, dtype=float)

        if self.weights.shape != (self.n,):
            raise ValueError(
                f"weights shape {self.weights.shape} != ({self.n},)"
            )
        if self.values.shape != (self.n,):
            raise ValueError(
                f"values shape {self.values.shape} != ({self.n},)"
            )
        if self.capacity < 0:
            raise ValueError("capacity must be non-negative")

    @classmethod
    def random(
        cls,
        n: int,
        capacity: float | None = None,
        weight_range: tuple[float, float] = (1.0, 50.0),
        value_range: tuple[float, float] = (1.0, 100.0),
        seed: int | None = None,
    ) -> KnapsackInstance:
        """Generate a random 0-1 Knapsack instance.

        Args:
            n: Number of items.
            capacity: Knapsack capacity. If None, set to ~50% of total weight.
            weight_range: Range for random item weights.
            value_range: Range for random item values.
            seed: Random seed for reproducibility.

        Returns:
            A random KnapsackInstance.
        """
        rng = np.random.default_rng(seed)
        weights = np.round(
            rng.uniform(weight_range[0], weight_range[1], size=n)
        ).astype(float)
        values = np.round(
            rng.uniform(value_range[0], value_range[1], size=n)
        ).astype(float)

        if capacity is None:
            capacity = float(np.sum(weights) * 0.5)

        return cls(
            n=n,
            weights=weights,
            values=values,
            capacity=capacity,
            name=f"random_{n}",
        )

    def total_weight(self, selection: list[int]) -> float:
        """Compute total weight of selected items."""
        return float(sum(self.weights[i] for i in selection))

    def total_value(self, selection: list[int]) -> float:
        """Compute total value of selected items."""
        return float(sum(self.values[i] for i in selection))

    def is_feasible(self, selection: list[int]) -> bool:
        """Check if selection is feasible (within capacity)."""
        return self.total_weight(selection) <= self.capacity + 1e-10


@dataclass
class KnapsackSolution:
    """Solution to a 0-1 Knapsack instance.

    Attributes:
        items: List of selected item indices (0-indexed).
        value: Total value of selected items.
        weight: Total weight of selected items.
    """

    items: list[int]
    value: float
    weight: float

    def __repr__(self) -> str:
        return (
            f"KnapsackSolution(value={self.value:.1f}, "
            f"weight={self.weight:.1f}, items={self.items})"
        )


def validate_solution(
    instance: KnapsackInstance, solution: KnapsackSolution
) -> tuple[bool, list[str]]:
    """Validate a knapsack solution.

    Args:
        instance: The Knapsack instance.
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

    # Check capacity
    actual_weight = instance.total_weight(solution.items)
    if actual_weight > instance.capacity + 1e-10:
        errors.append(
            f"Weight {actual_weight:.1f} exceeds capacity {instance.capacity:.1f}"
        )

    # Check value matches
    actual_value = instance.total_value(solution.items)
    if abs(actual_value - solution.value) > 1e-6:
        errors.append(
            f"Reported value {solution.value:.1f} != actual {actual_value:.1f}"
        )

    return len(errors) == 0, errors


# ── Benchmark instances ──────────────────────────────────────────────────────


def small_knapsack_4() -> KnapsackInstance:
    """4-item instance with known optimal = 35.

    Optimal selection: items [0, 2, 3] (weight=7, value=35).
    """
    return KnapsackInstance(
        n=4,
        weights=np.array([2.0, 3.0, 4.0, 1.0]),
        values=np.array([10.0, 15.0, 20.0, 5.0]),
        capacity=7.0,
        name="small4",
    )


def medium_knapsack_8() -> KnapsackInstance:
    """8-item instance for testing.

    Known optimal value: 300 (items [0, 3, 4, 7]).
    """
    return KnapsackInstance(
        n=8,
        weights=np.array([10.0, 20.0, 15.0, 25.0, 5.0, 10.0, 30.0, 8.0]),
        values=np.array([60.0, 100.0, 70.0, 120.0, 50.0, 45.0, 80.0, 70.0]),
        capacity=48.0,
        name="medium8",
    )


def strongly_correlated_10() -> KnapsackInstance:
    """10-item strongly correlated instance (v_i = w_i + 10).

    Strongly correlated instances are hard for greedy heuristics.
    """
    weights = np.array([10.0, 20.0, 30.0, 15.0, 25.0,
                        12.0, 18.0, 22.0, 8.0, 35.0])
    values = weights + 10.0
    return KnapsackInstance(
        n=10,
        weights=weights,
        values=values,
        capacity=80.0,
        name="strongly_correlated_10",
    )


if __name__ == "__main__":
    inst = small_knapsack_4()
    print(f"{inst.name}: n={inst.n}, W={inst.capacity}")
    print(f"  weights: {inst.weights}")
    print(f"  values: {inst.values}")
    print(f"  optimal items [0,2,3]: value={inst.total_value([0,2,3])}, "
          f"weight={inst.total_weight([0,2,3])}")
