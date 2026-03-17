"""
2D Bin Packing Problem (2D-BPP) — Instance and Solution.

Problem notation: 2D-BPP (strip variant: fixed width, minimize height)

Given n rectangular items with widths w_i and heights h_i, pack all items
into a strip of fixed width W to minimize total height used. Items may
not overlap and may not be rotated (unless specified).

Applications: VLSI layout, sheet metal cutting, textile cutting,
newspaper page layout, container loading.

Complexity: NP-hard (generalizes 1D-BPP).

References:
    Lodi, A., Martello, S. & Vigo, D. (2002). Two-dimensional packing
    problems: A survey. European Journal of Operational Research, 141(2),
    241-252. https://doi.org/10.1016/S0377-2217(02)00123-6

    Coffman, E.G., Garey, M.R., Johnson, D.S. & Tarjan, R.E. (1980).
    Performance bounds for level-oriented two-dimensional packing
    algorithms. SIAM Journal on Computing, 9(4), 808-826.
    https://doi.org/10.1137/0209062
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class BPP2DInstance:
    """2D Bin Packing (strip packing) instance.

    Attributes:
        n: Number of items.
        strip_width: Width of the strip.
        widths: Item widths, shape (n,).
        heights: Item heights, shape (n,).
        name: Optional instance name.
    """

    n: int
    strip_width: float
    widths: np.ndarray
    heights: np.ndarray
    name: str = ""

    def __post_init__(self):
        self.widths = np.asarray(self.widths, dtype=float)
        self.heights = np.asarray(self.heights, dtype=float)
        if self.widths.shape != (self.n,):
            raise ValueError(f"widths shape != ({self.n},)")
        if self.heights.shape != (self.n,):
            raise ValueError(f"heights shape != ({self.n},)")

    @classmethod
    def random(
        cls,
        n: int = 10,
        strip_width: float = 100.0,
        width_range: tuple[float, float] = (10.0, 50.0),
        height_range: tuple[float, float] = (10.0, 40.0),
        seed: int | None = None,
    ) -> BPP2DInstance:
        """Generate a random 2D-BPP instance."""
        rng = np.random.default_rng(seed)
        widths = np.round(rng.uniform(width_range[0], width_range[1], size=n))
        heights = np.round(rng.uniform(height_range[0], height_range[1], size=n))
        return cls(n=n, strip_width=strip_width, widths=widths,
                   heights=heights, name=f"random_{n}")

    def area_lower_bound(self) -> float:
        """Lower bound on strip height from total area."""
        total_area = np.sum(self.widths * self.heights)
        return total_area / self.strip_width


@dataclass
class BPP2DSolution:
    """Solution to a 2D-BPP instance.

    Attributes:
        positions: List of (x, y) positions for each item (bottom-left corner).
        height: Total strip height used.
    """

    positions: list[tuple[float, float]]
    height: float

    def __repr__(self) -> str:
        return f"BPP2DSolution(height={self.height:.1f})"


def validate_solution(
    instance: BPP2DInstance, solution: BPP2DSolution
) -> tuple[bool, list[str]]:
    """Validate a 2D-BPP solution."""
    errors = []

    if len(solution.positions) != instance.n:
        errors.append(f"Positions length {len(solution.positions)} != {instance.n}")
        return False, errors

    for i in range(instance.n):
        x, y = solution.positions[i]
        if x < -1e-10 or y < -1e-10:
            errors.append(f"Item {i}: negative position ({x:.1f}, {y:.1f})")
        if x + instance.widths[i] > instance.strip_width + 1e-10:
            errors.append(f"Item {i}: exceeds strip width")

    # Check overlaps
    for i in range(instance.n):
        x1, y1 = solution.positions[i]
        w1, h1 = instance.widths[i], instance.heights[i]
        for j in range(i + 1, instance.n):
            x2, y2 = solution.positions[j]
            w2, h2 = instance.widths[j], instance.heights[j]
            if (x1 < x2 + w2 - 1e-10 and x2 < x1 + w1 - 1e-10 and
                    y1 < y2 + h2 - 1e-10 and y2 < y1 + h1 - 1e-10):
                errors.append(f"Items {i} and {j} overlap")

    max_y = max(
        solution.positions[i][1] + instance.heights[i]
        for i in range(instance.n)
    )
    if abs(max_y - solution.height) > 1e-4:
        errors.append(f"Reported height {solution.height:.2f} != actual {max_y:.2f}")

    return len(errors) == 0, errors


# ── Benchmark instances ──────────────────────────────────────────────────────


def small_2dbpp_5() -> BPP2DInstance:
    """5 items on a strip of width 100."""
    return BPP2DInstance(
        n=5,
        strip_width=100.0,
        widths=np.array([40, 30, 50, 20, 60], dtype=float),
        heights=np.array([20, 30, 15, 25, 10], dtype=float),
        name="small_5",
    )


if __name__ == "__main__":
    inst = small_2dbpp_5()
    print(f"{inst.name}: {inst.n} items, W={inst.strip_width}")
    print(f"  Area LB: {inst.area_lower_bound():.1f}")
