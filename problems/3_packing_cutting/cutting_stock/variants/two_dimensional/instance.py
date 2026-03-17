"""
Two-Dimensional Cutting Stock Problem (2D-CSP) — Instance and Solution.

Cut rectangular items from large stock sheets to satisfy demands,
minimizing the number of stock sheets used.

Complexity: NP-hard (strongly, generalizes 2D bin packing).

References:
    Gilmore, P.C. & Gomory, R.E. (1965). Multistage cutting stock problems
    of two and more dimensions. Operations Research, 13(1), 94-120.
    https://doi.org/10.1287/opre.13.1.94

    Lodi, A., Martello, S. & Monaci, M. (2002). Two-dimensional packing
    problems: A survey. European Journal of Operational Research, 141(2),
    241-252. https://doi.org/10.1016/S0377-2217(02)00123-6
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class TwoDCSPInstance:
    """Two-Dimensional Cutting Stock Problem instance.

    Attributes:
        num_types: Number of item types.
        widths: Width of each item type, shape (num_types,).
        heights: Height of each item type, shape (num_types,).
        demands: Demand for each item type, shape (num_types,).
        sheet_width: Width of stock sheet.
        sheet_height: Height of stock sheet.
        allow_rotation: Whether 90-degree rotation is allowed.
        name: Optional instance name.
    """

    num_types: int
    widths: np.ndarray
    heights: np.ndarray
    demands: np.ndarray
    sheet_width: float
    sheet_height: float
    allow_rotation: bool = True
    name: str = ""

    def __post_init__(self):
        self.widths = np.asarray(self.widths, dtype=float)
        self.heights = np.asarray(self.heights, dtype=float)
        self.demands = np.asarray(self.demands, dtype=int)

    def item_fits(self, w: float, h: float) -> bool:
        """Check if an item with dimensions w x h fits on the sheet."""
        if w <= self.sheet_width + 1e-6 and h <= self.sheet_height + 1e-6:
            return True
        if self.allow_rotation:
            if h <= self.sheet_width + 1e-6 and w <= self.sheet_height + 1e-6:
                return True
        return False

    @classmethod
    def random(
        cls,
        num_types: int = 5,
        sheet_width: float = 100.0,
        sheet_height: float = 100.0,
        seed: int | None = None,
    ) -> TwoDCSPInstance:
        rng = np.random.default_rng(seed)
        widths = rng.integers(10, int(sheet_width * 0.6), size=num_types).astype(float)
        heights = rng.integers(10, int(sheet_height * 0.6), size=num_types).astype(float)
        demands = rng.integers(1, 6, size=num_types)
        return cls(
            num_types=num_types, widths=widths, heights=heights,
            demands=demands, sheet_width=sheet_width, sheet_height=sheet_height,
            name=f"random_2dcsp_{num_types}",
        )


@dataclass
class TwoDCSPSolution:
    """2D-CSP solution.

    Attributes:
        sheets: List of sheets. Each sheet is a list of
            (type_index, x, y, rotated) placement tuples.
        num_sheets: Total sheets used.
    """

    sheets: list[list[tuple[int, float, float, bool]]]
    num_sheets: int

    def __repr__(self) -> str:
        return f"TwoDCSPSolution(sheets={self.num_sheets})"


def validate_solution(
    instance: TwoDCSPInstance, solution: TwoDCSPSolution
) -> tuple[bool, list[str]]:
    errors = []

    # Check demands satisfied
    counts = np.zeros(instance.num_types, dtype=int)
    for sheet_idx, sheet in enumerate(solution.sheets):
        for type_idx, x, y, rotated in sheet:
            if type_idx < 0 or type_idx >= instance.num_types:
                errors.append(f"Sheet {sheet_idx}: invalid type {type_idx}")
                continue
            counts[type_idx] += 1

            w = instance.widths[type_idx]
            h = instance.heights[type_idx]
            if rotated:
                w, h = h, w
                if not instance.allow_rotation:
                    errors.append(f"Sheet {sheet_idx}: rotation not allowed")

            # Check bounds
            if x < -1e-6 or y < -1e-6:
                errors.append(f"Sheet {sheet_idx}: negative position ({x:.1f}, {y:.1f})")
            if x + w > instance.sheet_width + 1e-6:
                errors.append(f"Sheet {sheet_idx}: item exceeds width at x={x:.1f}")
            if y + h > instance.sheet_height + 1e-6:
                errors.append(f"Sheet {sheet_idx}: item exceeds height at y={y:.1f}")

        # Check no overlaps within sheet
        placements = []
        for type_idx, x, y, rotated in sheet:
            w = instance.widths[type_idx]
            h = instance.heights[type_idx]
            if rotated:
                w, h = h, w
            placements.append((x, y, x + w, y + h))

        for i in range(len(placements)):
            for j in range(i + 1, len(placements)):
                x1, y1, x2, y2 = placements[i]
                x3, y3, x4, y4 = placements[j]
                # Check overlap
                if not (x2 <= x3 + 1e-6 or x4 <= x1 + 1e-6 or
                        y2 <= y3 + 1e-6 or y4 <= y1 + 1e-6):
                    errors.append(f"Sheet {sheet_idx}: overlap between items {i} and {j}")

    for t in range(instance.num_types):
        if counts[t] < instance.demands[t]:
            errors.append(f"Type {t}: placed {counts[t]} < demand {instance.demands[t]}")

    if solution.num_sheets != len(solution.sheets):
        errors.append(f"Sheet count: {solution.num_sheets} != {len(solution.sheets)}")

    return len(errors) == 0, errors


def small_2dcsp_4() -> TwoDCSPInstance:
    """Small 2D-CSP with 4 item types."""
    return TwoDCSPInstance(
        num_types=4,
        widths=np.array([30, 40, 20, 50], dtype=float),
        heights=np.array([20, 30, 40, 25], dtype=float),
        demands=np.array([3, 2, 2, 1], dtype=int),
        sheet_width=100.0,
        sheet_height=100.0,
        name="small_2dcsp_4",
    )


if __name__ == "__main__":
    inst = small_2dcsp_4()
    print(f"{inst.name}: {inst.num_types} types, sheet={inst.sheet_width}x{inst.sheet_height}")
