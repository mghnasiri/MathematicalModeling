"""
1D Cutting Stock Problem (CSP) — Instance and Solution definitions.

Problem notation: CSP1D

Given stock material of length L and a set of m item types, each with
length l_i and demand d_i, determine how to cut stock rolls to satisfy
all demands using the minimum number of rolls. Each roll can be cut
according to a cutting pattern.

The CSP is a generalization of the Bin Packing Problem where items of
the same type are interchangeable. The LP relaxation can be solved via
column generation (Gilmore-Gomory), and the integer solution is typically
at most LP_OPT + 1 (IRUP property).

Complexity: NP-hard (reduces from Bin Packing).

References:
    Gilmore, P.C. & Gomory, R.E. (1961). A linear programming approach
    to the cutting-stock problem. Operations Research, 9(6), 849-859.
    https://doi.org/10.1287/opre.9.6.849

    Gilmore, P.C. & Gomory, R.E. (1963). A linear programming approach
    to the cutting stock problem — Part II. Operations Research,
    11(6), 863-888.
    https://doi.org/10.1287/opre.11.6.863

    Wäscher, G., Haußner, H. & Schumann, H. (2007). An improved
    typology of cutting and packing problems. European Journal of
    Operational Research, 183(3), 1109-1130.
    https://doi.org/10.1016/j.ejor.2005.12.047
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class CuttingStockInstance:
    """1D Cutting Stock Problem instance.

    Attributes:
        m: Number of item types.
        stock_length: Length of stock rolls L.
        lengths: Array of item lengths, shape (m,).
        demands: Array of item demands, shape (m,).
        name: Optional instance name.
    """

    m: int
    stock_length: float
    lengths: np.ndarray
    demands: np.ndarray
    name: str = ""

    def __post_init__(self):
        self.lengths = np.asarray(self.lengths, dtype=float)
        self.demands = np.asarray(self.demands, dtype=int)

        if self.lengths.shape != (self.m,):
            raise ValueError(
                f"lengths shape {self.lengths.shape} != ({self.m},)"
            )
        if self.demands.shape != (self.m,):
            raise ValueError(
                f"demands shape {self.demands.shape} != ({self.m},)"
            )
        if self.stock_length <= 0:
            raise ValueError("stock_length must be positive")
        if np.any(self.lengths > self.stock_length):
            raise ValueError("Some item lengths exceed stock length")
        if np.any(self.lengths <= 0):
            raise ValueError("All item lengths must be positive")
        if np.any(self.demands < 0):
            raise ValueError("All demands must be non-negative")

    @classmethod
    def random(
        cls,
        m: int,
        stock_length: float = 100.0,
        length_range: tuple[float, float] = (10.0, 50.0),
        demand_range: tuple[int, int] = (1, 10),
        seed: int | None = None,
    ) -> CuttingStockInstance:
        """Generate a random Cutting Stock instance.

        Args:
            m: Number of item types.
            stock_length: Stock roll length.
            length_range: Range for item lengths.
            demand_range: Range for item demands.
            seed: Random seed.

        Returns:
            A random CuttingStockInstance.
        """
        rng = np.random.default_rng(seed)
        max_len = min(length_range[1], stock_length)
        lengths = np.round(
            rng.uniform(length_range[0], max_len, size=m)
        ).astype(float)
        demands = rng.integers(demand_range[0], demand_range[1] + 1, size=m)

        return cls(
            m=m, stock_length=stock_length,
            lengths=lengths, demands=demands,
            name=f"random_{m}",
        )

    def lower_bound(self) -> int:
        """Continuous lower bound = ceil(sum(l_i * d_i) / L).

        Returns:
            Lower bound on optimal number of rolls.
        """
        total = float(np.sum(self.lengths * self.demands))
        return int(np.ceil(total / self.stock_length))

    def total_items(self) -> int:
        """Total number of individual items to cut."""
        return int(np.sum(self.demands))


@dataclass
class CuttingPattern:
    """A cutting pattern for a single stock roll.

    Attributes:
        counts: Array of how many of each item type is cut.
    """

    counts: np.ndarray

    @property
    def waste(self) -> float:
        """Requires instance to compute — use instance method."""
        return 0.0


@dataclass
class CuttingStockSolution:
    """Solution to a Cutting Stock instance.

    Attributes:
        patterns: List of (pattern_counts, frequency) tuples.
            pattern_counts: np.ndarray of how many of each type.
            frequency: how many rolls cut with this pattern.
        num_rolls: Total number of rolls used.
    """

    patterns: list[tuple[np.ndarray, int]]
    num_rolls: int

    def __repr__(self) -> str:
        return (
            f"CuttingStockSolution(num_rolls={self.num_rolls}, "
            f"patterns={len(self.patterns)})"
        )


def validate_solution(
    instance: CuttingStockInstance, solution: CuttingStockSolution
) -> tuple[bool, list[str]]:
    """Validate a cutting stock solution.

    Args:
        instance: The CSP instance.
        solution: The solution to validate.

    Returns:
        Tuple of (is_valid, list of error messages).
    """
    errors = []

    # Check each pattern fits in stock
    for i, (pattern, freq) in enumerate(solution.patterns):
        if freq < 0:
            errors.append(f"Pattern {i}: negative frequency {freq}")
        total_length = float(np.dot(pattern, instance.lengths))
        if total_length > instance.stock_length + 1e-10:
            errors.append(
                f"Pattern {i}: total length {total_length:.1f} > "
                f"stock length {instance.stock_length:.1f}"
            )
        if np.any(pattern < 0):
            errors.append(f"Pattern {i}: negative counts")

    # Check demands are satisfied
    total_produced = np.zeros(instance.m, dtype=int)
    for pattern, freq in solution.patterns:
        total_produced += np.asarray(pattern, dtype=int) * freq

    for i in range(instance.m):
        if total_produced[i] < instance.demands[i]:
            errors.append(
                f"Item type {i}: produced {total_produced[i]} < "
                f"demand {instance.demands[i]}"
            )

    return len(errors) == 0, errors


# ── Benchmark instances ──────────────────────────────────────────────────────


def simple_csp_3() -> CuttingStockInstance:
    """3-type instance. Stock length=100.

    Types: 45 (x3), 36 (x2), 31 (x2).
    Total material = 3*45 + 2*36 + 2*31 = 135+72+62 = 269.
    LB = ceil(269/100) = 3.
    """
    return CuttingStockInstance(
        m=3,
        stock_length=100.0,
        lengths=np.array([45.0, 36.0, 31.0]),
        demands=np.array([3, 2, 2]),
        name="simple3",
    )


def classic_csp_4() -> CuttingStockInstance:
    """Classic 4-type instance from textbooks. Stock length=10.

    Types: 4 (x5), 3 (x8), 2.5 (x10), 2 (x6).
    """
    return CuttingStockInstance(
        m=4,
        stock_length=10.0,
        lengths=np.array([4.0, 3.0, 2.5, 2.0]),
        demands=np.array([5, 8, 10, 6]),
        name="classic4",
    )


if __name__ == "__main__":
    for name, fn in [("simple3", simple_csp_3), ("classic4", classic_csp_4)]:
        inst = fn()
        print(f"{name}: m={inst.m}, L={inst.stock_length}, LB={inst.lower_bound()}")
        print(f"  lengths: {inst.lengths}")
        print(f"  demands: {inst.demands}")
