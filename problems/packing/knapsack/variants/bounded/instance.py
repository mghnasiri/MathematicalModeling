"""
Bounded Knapsack Problem (BKP) — Instance and Solution.

Problem notation: BKP

Given n item types with weights w_i, values v_i, and upper bounds b_i
on the number of copies, and a knapsack with capacity W, select quantities
x_i ∈ {0,...,b_i} to maximize total value subject to capacity.

Applications: investment allocation, cargo loading with limited stock,
production planning with bounded resources.

Complexity: NP-hard (weakly; admits O(nW) DP).

References:
    Kellerer, H., Pferschy, U. & Pisinger, D. (2004). Knapsack Problems.
    Springer. https://doi.org/10.1007/978-3-540-24777-7
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class BKPInstance:
    """Bounded Knapsack instance.

    Attributes:
        n: Number of item types.
        capacity: Knapsack capacity.
        values: Value per item, shape (n,).
        weights: Weight per item, shape (n,).
        bounds: Max copies of each item, shape (n,).
        name: Optional instance name.
    """

    n: int
    capacity: float
    values: np.ndarray
    weights: np.ndarray
    bounds: np.ndarray
    name: str = ""

    def __post_init__(self):
        self.values = np.asarray(self.values, dtype=float)
        self.weights = np.asarray(self.weights, dtype=float)
        self.bounds = np.asarray(self.bounds, dtype=int)

    @classmethod
    def random(
        cls, n: int = 8, capacity: float = 100.0,
        weight_range: tuple[int, int] = (5, 25),
        value_range: tuple[int, int] = (10, 60),
        bound_range: tuple[int, int] = (1, 5),
        seed: int | None = None,
    ) -> BKPInstance:
        rng = np.random.default_rng(seed)
        weights = rng.integers(weight_range[0], weight_range[1] + 1, size=n).astype(float)
        values = rng.integers(value_range[0], value_range[1] + 1, size=n).astype(float)
        bounds = rng.integers(bound_range[0], bound_range[1] + 1, size=n)
        return cls(n=n, capacity=capacity, values=values, weights=weights,
                   bounds=bounds, name=f"random_{n}")

    def is_feasible(self, quantities: list[int]) -> bool:
        total_weight = sum(self.weights[i] * quantities[i] for i in range(self.n))
        if total_weight > self.capacity + 1e-10:
            return False
        for i in range(self.n):
            if quantities[i] < 0 or quantities[i] > self.bounds[i]:
                return False
        return True

    def total_value(self, quantities: list[int]) -> float:
        return sum(self.values[i] * quantities[i] for i in range(self.n))


@dataclass
class BKPSolution:
    quantities: list[int]
    value: float

    def __repr__(self) -> str:
        return f"BKPSolution(value={self.value:.1f})"


def validate_solution(
    instance: BKPInstance, solution: BKPSolution
) -> tuple[bool, list[str]]:
    errors = []
    if len(solution.quantities) != instance.n:
        errors.append(f"Expected {instance.n} quantities")
        return False, errors
    for i in range(instance.n):
        if solution.quantities[i] < 0:
            errors.append(f"Item {i}: negative quantity")
        if solution.quantities[i] > instance.bounds[i]:
            errors.append(f"Item {i}: qty {solution.quantities[i]} > bound {instance.bounds[i]}")
    total_w = sum(instance.weights[i] * solution.quantities[i] for i in range(instance.n))
    if total_w > instance.capacity + 1e-10:
        errors.append(f"Weight {total_w:.1f} > capacity {instance.capacity:.1f}")
    actual_v = instance.total_value(solution.quantities)
    if abs(actual_v - solution.value) > 1e-4:
        errors.append(f"Reported value {solution.value:.2f} != actual {actual_v:.2f}")
    return len(errors) == 0, errors


def small_bkp_5() -> BKPInstance:
    return BKPInstance(
        n=5, capacity=50,
        values=np.array([60, 100, 120, 80, 50], dtype=float),
        weights=np.array([10, 20, 30, 15, 10], dtype=float),
        bounds=np.array([2, 1, 1, 3, 2]),
        name="small_5",
    )


if __name__ == "__main__":
    inst = small_bkp_5()
    print(f"{inst.name}: n={inst.n}, W={inst.capacity}")
