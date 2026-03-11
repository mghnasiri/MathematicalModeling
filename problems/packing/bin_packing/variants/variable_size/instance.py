"""
Variable-Size Bin Packing Problem (VS-BPP) — Instance and Solution.

Problem notation: VS-BPP

Given n items with sizes s_i, and K bin types each with capacity C_k and
cost c_k, pack all items using bins to minimize total cost (or total bins).

Applications: cloud VM provisioning, container ship loading, memory
allocation with different page sizes, truck fleet selection.

Complexity: NP-hard (generalizes standard BPP).

References:
    Friesen, D.K. & Langston, M.A. (1986). Variable sized bin packing.
    SIAM Journal on Computing, 15(1), 222-230.
    https://doi.org/10.1137/0215016

    Correia, I., Gouveia, L. & Saldanha-da-Gama, F. (2008). Solving
    the variable size bin packing problem with discretized formulations.
    Computers & Operations Research, 35(6), 2103-2113.
    https://doi.org/10.1016/j.cor.2006.10.014
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class VSBPPInstance:
    """Variable-Size Bin Packing instance.

    Attributes:
        n: Number of items.
        num_bin_types: Number of bin types.
        item_sizes: Item sizes, shape (n,).
        bin_capacities: Bin type capacities, shape (num_bin_types,).
        bin_costs: Bin type costs, shape (num_bin_types,).
        name: Optional instance name.
    """

    n: int
    num_bin_types: int
    item_sizes: np.ndarray
    bin_capacities: np.ndarray
    bin_costs: np.ndarray
    name: str = ""

    def __post_init__(self):
        self.item_sizes = np.asarray(self.item_sizes, dtype=float)
        self.bin_capacities = np.asarray(self.bin_capacities, dtype=float)
        self.bin_costs = np.asarray(self.bin_costs, dtype=float)

    @classmethod
    def random(
        cls,
        n: int = 12,
        num_bin_types: int = 3,
        size_range: tuple[float, float] = (5.0, 40.0),
        seed: int | None = None,
    ) -> VSBPPInstance:
        """Generate a random VS-BPP instance."""
        rng = np.random.default_rng(seed)
        item_sizes = np.round(rng.uniform(size_range[0], size_range[1], size=n))
        capacities = np.sort(np.round(rng.uniform(40.0, 120.0, size=num_bin_types)))
        # Cost roughly proportional to capacity with discount for larger bins
        costs = np.round(capacities * rng.uniform(0.8, 1.2, size=num_bin_types) * 0.1, 1)
        return cls(n=n, num_bin_types=num_bin_types, item_sizes=item_sizes,
                   bin_capacities=capacities, bin_costs=costs, name=f"random_{n}")


@dataclass
class VSBPPSolution:
    """Solution to a VS-BPP instance.

    Attributes:
        bins: List of (bin_type, [items]) tuples.
        total_cost: Total cost of bins used.
    """

    bins: list[tuple[int, list[int]]]
    total_cost: float

    def __repr__(self) -> str:
        return f"VSBPPSolution(cost={self.total_cost:.1f}, bins={len(self.bins)})"


def validate_solution(
    instance: VSBPPInstance, solution: VSBPPSolution
) -> tuple[bool, list[str]]:
    """Validate a VS-BPP solution."""
    errors = []
    all_items = []
    actual_cost = 0.0

    for idx, (btype, items) in enumerate(solution.bins):
        if btype < 0 or btype >= instance.num_bin_types:
            errors.append(f"Bin {idx}: invalid type {btype}")
            continue
        load = sum(instance.item_sizes[i] for i in items)
        if load > instance.bin_capacities[btype] + 1e-10:
            errors.append(
                f"Bin {idx} (type {btype}): load {load:.1f} > "
                f"capacity {instance.bin_capacities[btype]:.1f}"
            )
        actual_cost += instance.bin_costs[btype]
        all_items.extend(items)

    if sorted(all_items) != list(range(instance.n)):
        errors.append("Not all items packed exactly once")

    if abs(actual_cost - solution.total_cost) > 1e-4:
        errors.append(f"Reported cost {solution.total_cost:.2f} != actual {actual_cost:.2f}")

    return len(errors) == 0, errors


def small_vsbpp_8() -> VSBPPInstance:
    """8 items, 3 bin types."""
    return VSBPPInstance(
        n=8, num_bin_types=3,
        item_sizes=np.array([15, 25, 10, 30, 20, 35, 5, 40], dtype=float),
        bin_capacities=np.array([50, 80, 120], dtype=float),
        bin_costs=np.array([4, 6, 8], dtype=float),
        name="small_8",
    )


if __name__ == "__main__":
    inst = small_vsbpp_8()
    print(f"{inst.name}: {inst.n} items, {inst.num_bin_types} bin types")
