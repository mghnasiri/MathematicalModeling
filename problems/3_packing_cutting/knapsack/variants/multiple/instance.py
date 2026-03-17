"""
Multiple Knapsack Problem (mKP) — Instance and Solution.

Problem notation: mKP

Given n items with weights w_j and values v_j, and k knapsacks each with
capacity C_k, assign items to knapsacks to maximize total value subject to
each knapsack's capacity constraint. Each item is assigned to at most one
knapsack (or not selected).

Applications: resource allocation across budgets, cargo loading across
vehicles, task assignment with resource limits, memory allocation.

Complexity: NP-hard (generalizes 0-1 Knapsack).

References:
    Martello, S. & Toth, P. (1990). Knapsack Problems: Algorithms and
    Computer Implementations. Wiley.

    Pisinger, D. (1999). An exact algorithm for large multiple knapsack
    problems. European Journal of Operational Research, 114(3), 528-541.
    https://doi.org/10.1016/S0377-2217(98)00120-9

    Kellerer, H., Pferschy, U. & Pisinger, D. (2004). Knapsack Problems.
    Springer.
    https://doi.org/10.1007/978-3-540-24777-7
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


@dataclass
class MultipleKnapsackInstance:
    """Multiple Knapsack Problem instance.

    Attributes:
        n: Number of items.
        k: Number of knapsacks.
        values: Item values, shape (n,).
        weights: Item weights, shape (n,).
        capacities: Knapsack capacities, shape (k,).
        name: Optional instance name.
    """

    n: int
    k: int
    values: np.ndarray
    weights: np.ndarray
    capacities: np.ndarray
    name: str = ""

    def __post_init__(self):
        self.values = np.asarray(self.values, dtype=float)
        self.weights = np.asarray(self.weights, dtype=float)
        self.capacities = np.asarray(self.capacities, dtype=float)

        if self.values.shape != (self.n,):
            raise ValueError(f"values shape != ({self.n},)")
        if self.weights.shape != (self.n,):
            raise ValueError(f"weights shape != ({self.n},)")
        if self.capacities.shape != (self.k,):
            raise ValueError(f"capacities shape != ({self.k},)")

    @classmethod
    def random(
        cls,
        n: int = 15,
        k: int = 3,
        weight_range: tuple[int, int] = (5, 30),
        value_range: tuple[int, int] = (10, 100),
        capacity_factor: float = 0.4,
        seed: int | None = None,
    ) -> MultipleKnapsackInstance:
        """Generate a random mKP instance.

        Args:
            n: Number of items.
            k: Number of knapsacks.
            weight_range: Range for item weights.
            value_range: Range for item values.
            capacity_factor: Each capacity ≈ factor * sum(weights) / k.
            seed: Random seed.

        Returns:
            A random MultipleKnapsackInstance.
        """
        rng = np.random.default_rng(seed)
        weights = rng.integers(weight_range[0], weight_range[1] + 1, size=n).astype(float)
        values = rng.integers(value_range[0], value_range[1] + 1, size=n).astype(float)
        avg_cap = weights.sum() * capacity_factor / k
        capacities = np.round(
            rng.uniform(avg_cap * 0.8, avg_cap * 1.2, size=k)
        )
        return cls(n=n, k=k, values=values, weights=weights,
                   capacities=capacities, name=f"random_{n}_{k}")

    def total_value(self, assignments: list[int]) -> float:
        """Compute total value. assignments[j] = knapsack index or -1."""
        return sum(self.values[j] for j in range(self.n) if assignments[j] >= 0)

    def is_feasible(self, assignments: list[int]) -> bool:
        """Check if assignments respect all capacity constraints."""
        load = np.zeros(self.k)
        for j in range(self.n):
            if assignments[j] >= 0:
                if assignments[j] >= self.k:
                    return False
                load[assignments[j]] += self.weights[j]
        return all(load[i] <= self.capacities[i] + 1e-10 for i in range(self.k))


@dataclass
class MultipleKnapsackSolution:
    """Solution to a multiple knapsack instance.

    Attributes:
        assignments: assignments[j] = knapsack index for item j, or -1 if unassigned.
        value: Total value of assigned items.
    """

    assignments: list[int]
    value: float

    def __repr__(self) -> str:
        assigned = sum(1 for a in self.assignments if a >= 0)
        return f"MultipleKnapsackSolution(value={self.value:.1f}, assigned={assigned})"


def validate_solution(
    instance: MultipleKnapsackInstance,
    solution: MultipleKnapsackSolution,
) -> tuple[bool, list[str]]:
    """Validate a multiple knapsack solution."""
    errors = []

    if len(solution.assignments) != instance.n:
        errors.append(f"Assignments length {len(solution.assignments)} != {instance.n}")
        return False, errors

    load = np.zeros(instance.k)
    actual_value = 0.0
    for j in range(instance.n):
        a = solution.assignments[j]
        if a >= 0:
            if a >= instance.k:
                errors.append(f"Item {j} assigned to invalid knapsack {a}")
            else:
                load[a] += instance.weights[j]
                actual_value += instance.values[j]

    for i in range(instance.k):
        if load[i] > instance.capacities[i] + 1e-10:
            errors.append(
                f"Knapsack {i}: load {load[i]:.1f} > capacity {instance.capacities[i]:.1f}"
            )

    if abs(actual_value - solution.value) > 1e-4:
        errors.append(f"Reported value {solution.value:.2f} != actual {actual_value:.2f}")

    return len(errors) == 0, errors


# ── Benchmark instances ──────────────────────────────────────────────────────


def small_mkp_6_2() -> MultipleKnapsackInstance:
    """6 items, 2 knapsacks."""
    return MultipleKnapsackInstance(
        n=6, k=2,
        values=np.array([60, 100, 120, 80, 50, 90], dtype=float),
        weights=np.array([10, 20, 30, 15, 10, 25], dtype=float),
        capacities=np.array([40, 35], dtype=float),
        name="small_6_2",
    )


if __name__ == "__main__":
    inst = small_mkp_6_2()
    print(f"{inst.name}: n={inst.n}, k={inst.k}")
    print(f"  weights: {inst.weights} (total={inst.weights.sum():.0f})")
    print(f"  capacities: {inst.capacities} (total={inst.capacities.sum():.0f})")
