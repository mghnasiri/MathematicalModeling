"""
2D Strip Packing Problem (2D-SPP) — Instance and Solution definitions.

Problem notation: 2D-SPP

Given n rectangular items with widths w_i and heights h_i, and a strip of
fixed width W and unlimited height, pack all items (no rotation, no overlap)
to minimize the total strip height used.

Complexity: NP-hard (Baker, Coffman & Rivest, 1980).

References:
    Baker, B.S., Coffman, E.G. & Rivest, R.L. (1980). Orthogonal packings
    in two dimensions. SIAM Journal on Computing, 9(4), 846-855.
    https://doi.org/10.1137/0209064

    Coffman, E.G., Garey, M.R., Johnson, D.S. & Tarjan, R.E. (1980).
    Performance bounds for level-oriented two-dimensional packing
    algorithms. SIAM Journal on Computing, 9(4), 808-826.
    https://doi.org/10.1137/0209062
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class StripPackingPlacement:
    """Placement of a single item on the strip.

    Attributes:
        item: Item index (0-indexed).
        x: X-coordinate of bottom-left corner.
        y: Y-coordinate of bottom-left corner.
    """

    item: int
    x: float
    y: float


@dataclass
class StripPackingInstance:
    """2D Strip Packing Problem instance.

    Attributes:
        n: Number of items.
        widths: Array of item widths, shape (n,).
        heights: Array of item heights, shape (n,).
        strip_width: Fixed strip width W.
        name: Optional instance name.
    """

    n: int
    widths: np.ndarray
    heights: np.ndarray
    strip_width: float
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
        if self.strip_width <= 0:
            raise ValueError("strip_width must be positive")
        if np.any(self.widths <= 0) or np.any(self.heights <= 0):
            raise ValueError("All item dimensions must be positive")
        if np.any(self.widths > self.strip_width):
            raise ValueError("Some item widths exceed strip width")

    @classmethod
    def random(
        cls,
        n: int,
        strip_width: float = 100.0,
        width_range: tuple[float, float] = (5.0, 50.0),
        height_range: tuple[float, float] = (5.0, 50.0),
        seed: int | None = None,
    ) -> StripPackingInstance:
        """Generate a random Strip Packing instance.

        Args:
            n: Number of items.
            strip_width: Fixed strip width.
            width_range: Range for random item widths.
            height_range: Range for random item heights.
            seed: Random seed for reproducibility.

        Returns:
            A random StripPackingInstance.
        """
        rng = np.random.default_rng(seed)
        max_w = min(width_range[1], strip_width)
        widths = np.round(
            rng.uniform(width_range[0], max_w, size=n)
        ).astype(float)
        heights = np.round(
            rng.uniform(height_range[0], height_range[1], size=n)
        ).astype(float)

        return cls(
            n=n, widths=widths, heights=heights,
            strip_width=strip_width,
            name=f"random_{n}",
        )

    def area_lower_bound(self) -> float:
        """Area-based lower bound: sum(w_i * h_i) / W.

        Returns:
            Lower bound on optimal strip height.
        """
        total_area = float(np.sum(self.widths * self.heights))
        return total_area / self.strip_width

    def max_height_lower_bound(self) -> float:
        """Max-height lower bound: max(h_i).

        Any packing must have height >= the tallest item.

        Returns:
            Lower bound on optimal strip height.
        """
        return float(np.max(self.heights))


@dataclass
class StripPackingSolution:
    """Solution to a Strip Packing instance.

    Attributes:
        placements: List of StripPackingPlacement objects.
        height: Total strip height used (max y + h over all items).
    """

    placements: list[StripPackingPlacement]
    height: float

    def __repr__(self) -> str:
        return (
            f"StripPackingSolution(height={self.height:.1f}, "
            f"n_items={len(self.placements)})"
        )


def validate_solution(
    instance: StripPackingInstance, solution: StripPackingSolution
) -> tuple[bool, list[str]]:
    """Validate a strip packing solution.

    Checks: all items placed exactly once, items within strip width,
    no overlaps, reported height matches actual.

    Args:
        instance: The SPP instance.
        solution: The solution to validate.

    Returns:
        Tuple of (is_valid, list of error messages).
    """
    errors = []

    # Check all items placed exactly once
    placed_items = [p.item for p in solution.placements]
    expected = set(range(instance.n))
    packed = set(placed_items)

    if len(placed_items) != len(set(placed_items)):
        errors.append("Duplicate items in placements")
    if packed != expected:
        missing = expected - packed
        extra = packed - expected
        if missing:
            errors.append(f"Unpacked items: {missing}")
        if extra:
            errors.append(f"Invalid item indices: {extra}")

    if errors:
        return False, errors

    # Check boundary constraints
    actual_height = 0.0
    for p in solution.placements:
        w = instance.widths[p.item]
        h = instance.heights[p.item]
        if p.x < -1e-10 or p.y < -1e-10:
            errors.append(
                f"Item {p.item}: negative position ({p.x:.1f}, {p.y:.1f})"
            )
        if p.x + w > instance.strip_width + 1e-10:
            errors.append(
                f"Item {p.item}: exceeds strip width "
                f"(x={p.x:.1f}, w={w:.1f}, W={instance.strip_width:.1f})"
            )
        actual_height = max(actual_height, p.y + h)

    # Check reported height
    if abs(actual_height - solution.height) > 1e-6:
        errors.append(
            f"Reported height {solution.height:.1f} != actual {actual_height:.1f}"
        )

    # Check overlaps
    ps = solution.placements
    for i in range(len(ps)):
        for j in range(i + 1, len(ps)):
            p1, p2 = ps[i], ps[j]
            w1 = instance.widths[p1.item]
            h1 = instance.heights[p1.item]
            w2 = instance.widths[p2.item]
            h2 = instance.heights[p2.item]

            x_overlap = (
                p1.x < p2.x + w2 - 1e-10 and p2.x < p1.x + w1 - 1e-10
            )
            y_overlap = (
                p1.y < p2.y + h2 - 1e-10 and p2.y < p1.y + h1 - 1e-10
            )
            if x_overlap and y_overlap:
                errors.append(f"Items {p1.item} and {p2.item} overlap")

    return len(errors) == 0, errors


# ── Benchmark instances ──────────────────────────────────────────────────────


def small_spp_3() -> StripPackingInstance:
    """3-item instance, strip width 10.

    Items: 6x4, 5x3, 4x5. Sum area = 24+15+20 = 59.
    Area LB = 59/10 = 5.9. Max height LB = 5.
    """
    return StripPackingInstance(
        n=3,
        widths=np.array([6.0, 5.0, 4.0]),
        heights=np.array([4.0, 3.0, 5.0]),
        strip_width=10.0,
        name="small3",
    )


def uniform_spp_6() -> StripPackingInstance:
    """6 items of size 3x3, strip width 10.

    Area = 54. Area LB = 54/10 = 5.4.
    Shelf packing: row1 = 3 items (width 9), row2 = 3 items. Height = 6.
    """
    return StripPackingInstance(
        n=6,
        widths=np.full(6, 3.0),
        heights=np.full(6, 3.0),
        strip_width=10.0,
        name="uniform6",
    )


def wide_items_4() -> StripPackingInstance:
    """4 wide items, strip width 10.

    Items: 8x2, 9x3, 7x4, 10x1. Each takes nearly full width.
    Optimal height ~ 10 (stacked).
    """
    return StripPackingInstance(
        n=4,
        widths=np.array([8.0, 9.0, 7.0, 10.0]),
        heights=np.array([2.0, 3.0, 4.0, 1.0]),
        strip_width=10.0,
        name="wide4",
    )


if __name__ == "__main__":
    for name, fn in [("small3", small_spp_3), ("uniform6", uniform_spp_6),
                      ("wide4", wide_items_4)]:
        inst = fn()
        print(f"{name}: n={inst.n}, W={inst.strip_width}, "
              f"area_LB={inst.area_lower_bound():.1f}, "
              f"max_h_LB={inst.max_height_lower_bound():.1f}")
