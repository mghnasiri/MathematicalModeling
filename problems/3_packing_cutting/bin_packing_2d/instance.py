"""
2D Bin Packing Problem (2D-BPP) — Instance and Solution definitions.

Problem notation: 2D-BPP

Given n rectangular items with widths w_i and heights h_i, and bins of
width W and height H, pack all items into the minimum number of bins such
that items do not overlap and fit within bin boundaries. Items may not
be rotated (oriented variant).

Complexity: NP-hard (Lodi, Martello & Vigo, 2002).

References:
    Lodi, A., Martello, S. & Vigo, D. (2002). Two-dimensional packing
    problems: A survey. European Journal of Operational Research,
    141(2), 241-252.
    https://doi.org/10.1016/S0377-2217(02)00123-6

    Coffman, E.G., Garey, M.R., Johnson, D.S. & Tarjan, R.E. (1980).
    Performance bounds for level-oriented two-dimensional packing
    algorithms. SIAM Journal on Computing, 9(4), 808-826.
    https://doi.org/10.1137/0209062
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


@dataclass
class BinPacking2DInstance:
    """2D Bin Packing Problem instance.

    Attributes:
        n: Number of items.
        widths: Array of item widths, shape (n,).
        heights: Array of item heights, shape (n,).
        bin_width: Bin width W.
        bin_height: Bin height H.
        name: Optional instance name.
    """

    n: int
    widths: np.ndarray
    heights: np.ndarray
    bin_width: float
    bin_height: float
    name: str = ""

    def __post_init__(self):
        self.widths = np.asarray(self.widths, dtype=float)
        self.heights = np.asarray(self.heights, dtype=float)

        if self.widths.shape != (self.n,):
            raise ValueError(
                f"widths shape {self.widths.shape} != ({self.n},)"
            )
        if self.heights.shape != (self.n,):
            raise ValueError(
                f"heights shape {self.heights.shape} != ({self.n},)"
            )
        if self.bin_width <= 0 or self.bin_height <= 0:
            raise ValueError("bin_width and bin_height must be positive")
        if np.any(self.widths <= 0) or np.any(self.heights <= 0):
            raise ValueError("All item dimensions must be positive")
        if np.any(self.widths > self.bin_width):
            raise ValueError("Some item widths exceed bin width")
        if np.any(self.heights > self.bin_height):
            raise ValueError("Some item heights exceed bin height")

    @classmethod
    def random(
        cls,
        n: int,
        bin_width: float = 100.0,
        bin_height: float = 100.0,
        width_range: tuple[float, float] = (5.0, 50.0),
        height_range: tuple[float, float] = (5.0, 50.0),
        seed: int | None = None,
    ) -> BinPacking2DInstance:
        """Generate a random 2D Bin Packing instance.

        Args:
            n: Number of items.
            bin_width: Bin width.
            bin_height: Bin height.
            width_range: Range for random item widths.
            height_range: Range for random item heights.
            seed: Random seed for reproducibility.

        Returns:
            A random BinPacking2DInstance.
        """
        rng = np.random.default_rng(seed)
        max_w = min(width_range[1], bin_width)
        max_h = min(height_range[1], bin_height)
        widths = np.round(
            rng.uniform(width_range[0], max_w, size=n)
        ).astype(float)
        heights = np.round(
            rng.uniform(height_range[0], max_h, size=n)
        ).astype(float)

        return cls(
            n=n, widths=widths, heights=heights,
            bin_width=bin_width, bin_height=bin_height,
            name=f"random_{n}",
        )

    def area_lower_bound(self) -> int:
        """Area-based lower bound: ceil(sum(w_i * h_i) / (W * H)).

        Returns:
            Lower bound on optimal number of bins.
        """
        total_area = float(np.sum(self.widths * self.heights))
        bin_area = self.bin_width * self.bin_height
        return int(np.ceil(total_area / bin_area))


@dataclass
class Placement:
    """Placement of a single item in a bin.

    Attributes:
        item: Item index (0-indexed).
        x: X-coordinate of bottom-left corner.
        y: Y-coordinate of bottom-left corner.
    """

    item: int
    x: float
    y: float


@dataclass
class BinPacking2DSolution:
    """Solution to a 2D Bin Packing instance.

    Attributes:
        bins: List of bins, each a list of Placement objects.
        num_bins: Number of bins used.
    """

    bins: list[list[Placement]]
    num_bins: int

    def __repr__(self) -> str:
        items_per_bin = [len(b) for b in self.bins]
        return (
            f"BinPacking2DSolution(num_bins={self.num_bins}, "
            f"items_per_bin={items_per_bin})"
        )


def validate_solution(
    instance: BinPacking2DInstance, solution: BinPacking2DSolution
) -> tuple[bool, list[str]]:
    """Validate a 2D bin packing solution.

    Checks: all items placed exactly once, items within bin boundaries,
    no overlaps within any bin.

    Args:
        instance: The 2D-BPP instance.
        solution: The solution to validate.

    Returns:
        Tuple of (is_valid, list of error messages).
    """
    errors = []

    # Check all items packed exactly once
    all_items = []
    for b in solution.bins:
        for p in b:
            all_items.append(p.item)

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

    # Check boundary and overlap constraints per bin
    for bi, b in enumerate(solution.bins):
        for p in b:
            w = instance.widths[p.item]
            h = instance.heights[p.item]
            if p.x < -1e-10 or p.y < -1e-10:
                errors.append(
                    f"Bin {bi}, item {p.item}: negative position "
                    f"({p.x:.1f}, {p.y:.1f})"
                )
            if p.x + w > instance.bin_width + 1e-10:
                errors.append(
                    f"Bin {bi}, item {p.item}: exceeds bin width "
                    f"(x={p.x:.1f}, w={w:.1f}, W={instance.bin_width:.1f})"
                )
            if p.y + h > instance.bin_height + 1e-10:
                errors.append(
                    f"Bin {bi}, item {p.item}: exceeds bin height "
                    f"(y={p.y:.1f}, h={h:.1f}, H={instance.bin_height:.1f})"
                )

        # Check overlaps within bin
        for i in range(len(b)):
            for j in range(i + 1, len(b)):
                p1, p2 = b[i], b[j]
                w1 = instance.widths[p1.item]
                h1 = instance.heights[p1.item]
                w2 = instance.widths[p2.item]
                h2 = instance.heights[p2.item]

                # Two rectangles overlap iff they overlap on both axes
                x_overlap = (
                    p1.x < p2.x + w2 - 1e-10 and p2.x < p1.x + w1 - 1e-10
                )
                y_overlap = (
                    p1.y < p2.y + h2 - 1e-10 and p2.y < p1.y + h1 - 1e-10
                )
                if x_overlap and y_overlap:
                    errors.append(
                        f"Bin {bi}: items {p1.item} and {p2.item} overlap"
                    )

    return len(errors) == 0, errors


# ── Benchmark instances ──────────────────────────────────────────────────────


def small_2dbpp_4() -> BinPacking2DInstance:
    """4-item instance, bin 10x10.

    Items: 6x6, 6x4, 4x6, 4x4.
    Area sum = 36+24+24+16 = 100 = 1 bin area, but cannot fit in 1 bin.
    Optimal: 2 bins.
    """
    return BinPacking2DInstance(
        n=4,
        widths=np.array([6.0, 6.0, 4.0, 4.0]),
        heights=np.array([6.0, 4.0, 6.0, 4.0]),
        bin_width=10.0,
        bin_height=10.0,
        name="small4",
    )


def uniform_2dbpp_6() -> BinPacking2DInstance:
    """6 items of size 3x3, bin 10x10.

    Each bin fits floor(10/3)^2 = 9 items, so 1 bin suffices.
    """
    return BinPacking2DInstance(
        n=6,
        widths=np.full(6, 3.0),
        heights=np.full(6, 3.0),
        bin_width=10.0,
        bin_height=10.0,
        name="uniform6",
    )


def tall_items_5() -> BinPacking2DInstance:
    """5 tall items, bin 10x10.

    Items: all 3x8 (tall). Each bin row (width 10) fits 3 items (3*3=9<=10).
    Height 8 means only 1 row per bin. So 2 bins: {3,3,3} and {3,3}.
    """
    return BinPacking2DInstance(
        n=5,
        widths=np.full(5, 3.0),
        heights=np.full(5, 8.0),
        bin_width=10.0,
        bin_height=10.0,
        name="tall5",
    )


if __name__ == "__main__":
    for name, fn in [("small4", small_2dbpp_4), ("uniform6", uniform_2dbpp_6),
                      ("tall5", tall_items_5)]:
        inst = fn()
        print(f"{name}: n={inst.n}, W={inst.bin_width}, H={inst.bin_height}, "
              f"area_LB={inst.area_lower_bound()}")
