"""
1D Bin Packing Problem (BPP) — Instance and Solution definitions.

Problem notation: BPP1D

Given n items with sizes s_i and bins of capacity C, pack all items into
the minimum number of bins such that the total size in each bin does not
exceed C.

Complexity: NP-hard (Garey & Johnson, 1979).

References:
    Garey, M.R. & Johnson, D.S. (1979). Computers and Intractability:
    A Guide to the Theory of NP-Completeness. W.H. Freeman, New York.

    Coffman, E.G., Garey, M.R. & Johnson, D.S. (1997). Approximation
    algorithms for bin packing: A survey. In: Hochbaum, D.S. (ed)
    Approximation Algorithms for NP-Hard Problems, PWS Publishing,
    46-93.

    Martello, S. & Toth, P. (1990). Lower bounds and reduction
    procedures for the bin packing problem. Discrete Applied
    Mathematics, 28(1), 59-70.
    https://doi.org/10.1016/0166-218X(90)90094-S
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class BinPackingInstance:
    """1D Bin Packing Problem instance.

    Attributes:
        n: Number of items.
        sizes: Array of item sizes, shape (n,).
        capacity: Bin capacity C.
        name: Optional instance name.
    """

    n: int
    sizes: np.ndarray
    capacity: float
    name: str = ""

    def __post_init__(self):
        self.sizes = np.asarray(self.sizes, dtype=float)

        if self.sizes.shape != (self.n,):
            raise ValueError(
                f"sizes shape {self.sizes.shape} != ({self.n},)"
            )
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        if np.any(self.sizes > self.capacity):
            raise ValueError("Some item sizes exceed bin capacity")
        if np.any(self.sizes <= 0):
            raise ValueError("All item sizes must be positive")

    @classmethod
    def random(
        cls,
        n: int,
        capacity: float = 100.0,
        size_range: tuple[float, float] = (1.0, 50.0),
        seed: int | None = None,
    ) -> BinPackingInstance:
        """Generate a random 1D Bin Packing instance.

        Args:
            n: Number of items.
            capacity: Bin capacity.
            size_range: Range for random item sizes.
            seed: Random seed for reproducibility.

        Returns:
            A random BinPackingInstance.
        """
        rng = np.random.default_rng(seed)
        max_size = min(size_range[1], capacity)
        sizes = np.round(
            rng.uniform(size_range[0], max_size, size=n)
        ).astype(float)

        return cls(
            n=n, sizes=sizes, capacity=capacity,
            name=f"random_{n}",
        )

    def lower_bound_l1(self) -> int:
        """Continuous lower bound L1 = ceil(sum(sizes) / capacity).

        Returns:
            Lower bound on optimal number of bins.
        """
        return int(np.ceil(np.sum(self.sizes) / self.capacity))

    def lower_bound_l2(self) -> int:
        """Martello-Toth L2 lower bound.

        Considers items larger than capacity/2 that must be in separate
        bins, plus residual space for smaller items.

        Returns:
            Lower bound on optimal number of bins (>= L1).
        """
        half_cap = self.capacity / 2.0
        large = self.sizes[self.sizes > half_cap]
        small = self.sizes[self.sizes <= half_cap]

        n_large = len(large)
        if n_large == 0:
            return self.lower_bound_l1()

        residual = n_large * self.capacity - float(np.sum(large))
        small_total = float(np.sum(small))

        if small_total <= residual:
            return n_large
        else:
            extra = int(np.ceil((small_total - residual) / self.capacity))
            return max(n_large + extra, self.lower_bound_l1())


@dataclass
class BinPackingSolution:
    """Solution to a 1D Bin Packing instance.

    Attributes:
        bins: List of bins, each a list of item indices (0-indexed).
        num_bins: Number of bins used.
    """

    bins: list[list[int]]
    num_bins: int

    def __repr__(self) -> str:
        return f"BinPackingSolution(num_bins={self.num_bins}, bins={self.bins})"


def validate_solution(
    instance: BinPackingInstance, solution: BinPackingSolution
) -> tuple[bool, list[str]]:
    """Validate a bin packing solution.

    Args:
        instance: The BPP instance.
        solution: The solution to validate.

    Returns:
        Tuple of (is_valid, list of error messages).
    """
    errors = []

    # Check all items packed
    all_items = []
    for b in solution.bins:
        all_items.extend(b)

    expected = set(range(instance.n))
    packed = set(all_items)

    if len(all_items) != len(set(all_items)):
        errors.append("Duplicate items in bins")
    if packed != expected:
        missing = expected - packed
        extra = packed - expected
        if missing:
            errors.append(f"Unpacked items: {missing}")
        if extra:
            errors.append(f"Invalid item indices: {extra}")

    if errors:
        return False, errors

    # Check capacity constraints
    for i, b in enumerate(solution.bins):
        total = sum(instance.sizes[j] for j in b)
        if total > instance.capacity + 1e-10:
            errors.append(
                f"Bin {i}: total size {total:.1f} > capacity {instance.capacity:.1f}"
            )

    return len(errors) == 0, errors


# ── Benchmark instances ──────────────────────────────────────────────────────


def easy_bpp_6() -> BinPackingInstance:
    """6-item instance with optimal = 2 bins (capacity 10).

    Items: [6, 5, 4, 3, 2, 1], sum=21, L1=ceil(21/10)=3.
    Optimal: {6,4}, {5,3,2} = 2 bins (since 6+4=10, 5+3+2=10).
    Wait: 1 is not packed. Actually: {6,3,1}, {5,4}, and item 2?
    Let me recalculate: items sum=21, capacity=10.
    Bins: {6,4}=10, {5,3,2}=10, {1}=1 => 3 bins.
    Or: {6,3,1}=10, {5,4}=9, {2}=2 => 3 bins.
    L1 = ceil(21/10) = 3, so optimal = 3.
    """
    return BinPackingInstance(
        n=6,
        sizes=np.array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0]),
        capacity=10.0,
        name="easy6",
    )


def tight_bpp_8() -> BinPackingInstance:
    """8-item instance with tight packing.

    Items: [7, 7, 6, 6, 5, 5, 4, 4], capacity=10.
    L1 = ceil(44/10) = 5.
    Optimal: {7,3-no}... Actually: can't pair 7s. {7}, {7}, {6,4}, {6,4}, {5,5}.
    That's 5 bins. Optimal = 5.
    """
    return BinPackingInstance(
        n=8,
        sizes=np.array([7.0, 7.0, 6.0, 6.0, 5.0, 5.0, 4.0, 4.0]),
        capacity=10.0,
        name="tight8",
    )


def uniform_bpp_10() -> BinPackingInstance:
    """10 items all of size 3, capacity 7. Optimal = 5 bins (2 items/bin)."""
    return BinPackingInstance(
        n=10,
        sizes=np.full(10, 3.0),
        capacity=7.0,
        name="uniform10",
    )


if __name__ == "__main__":
    for name, fn in [("easy6", easy_bpp_6), ("tight8", tight_bpp_8),
                      ("uniform10", uniform_bpp_10)]:
        inst = fn()
        print(f"{name}: n={inst.n}, C={inst.capacity}, L1={inst.lower_bound_l1()}, "
              f"L2={inst.lower_bound_l2()}")
