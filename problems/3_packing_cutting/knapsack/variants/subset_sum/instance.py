"""
Subset Sum Problem (SSP) — Instance and Solution.

Given a set of n positive integers and a target value T, find a subset
whose elements sum to exactly T (decision) or as close to T as possible
without exceeding it (optimization).

Complexity: NP-complete (decision), weakly NP-hard (admits pseudo-polynomial DP).

References:
    Garey, M.R. & Johnson, D.S. (1979). Computers and Intractability.
    W.H. Freeman. ISBN 978-0716710455.

    Kellerer, H., Pferschy, U. & Pisinger, D. (2004). Knapsack Problems.
    Springer. https://doi.org/10.1007/978-3-540-24777-7
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class SubsetSumInstance:
    """Subset Sum instance.

    Attributes:
        n: Number of elements.
        values: Element values, shape (n,).
        target: Target sum.
        name: Optional instance name.
    """

    n: int
    values: np.ndarray
    target: int
    name: str = ""

    def __post_init__(self):
        self.values = np.asarray(self.values, dtype=int)

    @classmethod
    def random(
        cls,
        n: int = 10,
        value_range: tuple[int, int] = (1, 100),
        target_ratio: float = 0.5,
        seed: int | None = None,
    ) -> SubsetSumInstance:
        rng = np.random.default_rng(seed)
        values = rng.integers(value_range[0], value_range[1] + 1, size=n)
        target = int(values.sum() * target_ratio)
        return cls(n=n, values=values, target=target, name=f"random_{n}")

    def subset_value(self, selected: list[int]) -> int:
        """Sum of selected elements."""
        return int(sum(self.values[i] for i in selected))


@dataclass
class SubsetSumSolution:
    """Subset Sum solution.

    Attributes:
        selected: Indices of selected elements.
        total: Sum of selected elements.
    """

    selected: list[int]
    total: int

    def __repr__(self) -> str:
        return f"SubsetSumSolution(total={self.total}, count={len(self.selected)})"


def validate_solution(
    instance: SubsetSumInstance, solution: SubsetSumSolution
) -> tuple[bool, list[str]]:
    errors = []
    if len(set(solution.selected)) != len(solution.selected):
        errors.append("Duplicate indices in selection")
    for i in solution.selected:
        if i < 0 or i >= instance.n:
            errors.append(f"Index {i} out of range")
    actual = instance.subset_value(solution.selected)
    if actual != solution.total:
        errors.append(f"Reported total {solution.total} != actual {actual}")
    if actual > instance.target:
        errors.append(f"Total {actual} exceeds target {instance.target}")
    return len(errors) == 0, errors


def small_ssp_6() -> SubsetSumInstance:
    return SubsetSumInstance(
        n=6,
        values=np.array([3, 7, 1, 8, 5, 2]),
        target=14,
        name="small_6",
    )


if __name__ == "__main__":
    inst = small_ssp_6()
    print(f"{inst.name}: n={inst.n}, target={inst.target}")
    print(f"  Values: {inst.values}")
