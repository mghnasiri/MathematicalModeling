"""
Online Bin Packing — Instance and Solution.

Items arrive one at a time and must be irrevocably placed into a bin
upon arrival (no repacking). The total number of items is unknown
in advance.

Competitive ratio: no online algorithm can achieve better than 1.5
(Yao, 1980). Best Fit Decreasing not available (items unsorted).

References:
    Johnson, D.S. (1973). Near-optimal bin packing algorithms. PhD thesis,
    MIT.

    Yao, A.C. (1980). New algorithms for bin packing. Journal of the ACM,
    27(2), 207-227. https://doi.org/10.1145/322186.322187
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class OnlineBPInstance:
    """Online Bin Packing instance.

    Attributes:
        n: Number of items.
        sizes: Item sizes, shape (n,). Each in (0, 1].
        capacity: Bin capacity (default 1.0).
        name: Optional instance name.
    """

    n: int
    sizes: np.ndarray
    capacity: float = 1.0
    name: str = ""

    def __post_init__(self):
        self.sizes = np.asarray(self.sizes, dtype=float)

    @classmethod
    def random(
        cls,
        n: int = 20,
        size_range: tuple[float, float] = (0.1, 0.7),
        seed: int | None = None,
    ) -> OnlineBPInstance:
        rng = np.random.default_rng(seed)
        sizes = rng.uniform(size_range[0], size_range[1], size=n)
        return cls(n=n, sizes=sizes, name=f"random_{n}")


@dataclass
class OnlineBPSolution:
    """Online Bin Packing solution.

    Attributes:
        assignments: assignment[i] = bin index for item i.
        num_bins: Number of bins used.
    """

    assignments: list[int]
    num_bins: int

    def __repr__(self) -> str:
        return f"OnlineBPSolution(bins={self.num_bins})"


def validate_solution(
    instance: OnlineBPInstance, solution: OnlineBPSolution
) -> tuple[bool, list[str]]:
    errors = []
    if len(solution.assignments) != instance.n:
        errors.append(f"Expected {instance.n} assignments, got {len(solution.assignments)}")
        return False, errors

    bin_loads: dict[int, float] = {}
    for i, b in enumerate(solution.assignments):
        if b < 0:
            errors.append(f"Item {i} not assigned")
            continue
        bin_loads[b] = bin_loads.get(b, 0.0) + instance.sizes[i]

    for b, load in bin_loads.items():
        if load > instance.capacity + 1e-6:
            errors.append(f"Bin {b} overloaded: {load:.4f} > {instance.capacity}")

    actual_bins = len(bin_loads) if bin_loads else 0
    if actual_bins != solution.num_bins:
        errors.append(f"Reported {solution.num_bins} bins but used {actual_bins}")

    return len(errors) == 0, errors


def small_online_8() -> OnlineBPInstance:
    return OnlineBPInstance(
        n=8,
        sizes=np.array([0.3, 0.7, 0.4, 0.5, 0.2, 0.6, 0.1, 0.8]),
        capacity=1.0,
        name="small_8",
    )


if __name__ == "__main__":
    inst = small_online_8()
    print(f"{inst.name}: n={inst.n}")
    print(f"  Sizes: {inst.sizes}")
    print(f"  L1 lower bound: {int(np.ceil(inst.sizes.sum() / inst.capacity))}")
