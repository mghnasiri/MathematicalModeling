"""
Level Algorithms — Heuristics for the 2D Strip Packing Problem.

Problem: 2D Strip Packing (2D-SPP), oriented (no rotation)
Complexity: O(n^2) for Bottom-Left, O(n log n) for NFDH

Bottom-Left (BL): For each item (sorted by decreasing height), place it
at the lowest possible y, then leftmost possible x. Simple but effective.

Next Fit Decreasing Height (NFDH): Sort items by height (decreasing).
Place items left-to-right on the current level. When an item doesn't
fit, start a new level at the top of the current shelf.

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

import os
import importlib.util
import sys

import numpy as np

_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_mod(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_inst = _load_mod("spp_instance_level", os.path.join(_parent_dir, "instance.py"))
StripPackingInstance = _inst.StripPackingInstance
StripPackingSolution = _inst.StripPackingSolution
StripPackingPlacement = _inst.StripPackingPlacement


def bottom_left(instance: StripPackingInstance) -> StripPackingSolution:
    """Pack items using Bottom-Left heuristic.

    Sort items by decreasing height. For each item, find the lowest
    position where it fits without overlapping any placed item, then
    leftmost at that height.

    Args:
        instance: A StripPackingInstance.

    Returns:
        StripPackingSolution with BL packing.
    """
    n = instance.n
    W = instance.strip_width

    # Sort by height descending
    order = sorted(range(n), key=lambda i: instance.heights[i], reverse=True)

    placements: list[StripPackingPlacement] = []
    # Track placed rectangles as (x, y, w, h)
    placed_rects: list[tuple[float, float, float, float]] = []

    for idx in order:
        w = instance.widths[idx]
        h = instance.heights[idx]

        best_x = 0.0
        best_y = float("inf")

        # Candidate y positions: 0 and top of each placed rectangle
        candidate_ys = [0.0]
        for (rx, ry, rw, rh) in placed_rects:
            candidate_ys.append(ry + rh)

        for cy in sorted(set(candidate_ys)):
            # Try all x positions: 0 and right edge of each placed rect
            candidate_xs = [0.0]
            for (rx, ry, rw, rh) in placed_rects:
                candidate_xs.append(rx + rw)

            for cx in sorted(set(candidate_xs)):
                if cx + w > W + 1e-10:
                    continue

                # Check for overlaps
                overlaps = False
                for (rx, ry, rw, rh) in placed_rects:
                    x_over = cx < rx + rw - 1e-10 and rx < cx + w - 1e-10
                    y_over = cy < ry + rh - 1e-10 and ry < cy + h - 1e-10
                    if x_over and y_over:
                        overlaps = True
                        break

                if not overlaps:
                    if cy < best_y - 1e-10 or (
                        abs(cy - best_y) < 1e-10 and cx < best_x - 1e-10
                    ):
                        best_y = cy
                        best_x = cx

        placements.append(StripPackingPlacement(item=idx, x=best_x, y=best_y))
        placed_rects.append((best_x, best_y, w, h))

    # Compute total height
    total_height = 0.0
    for p in placements:
        total_height = max(total_height, p.y + instance.heights[p.item])

    return StripPackingSolution(placements=placements, height=total_height)


def nfdh_strip(instance: StripPackingInstance) -> StripPackingSolution:
    """Pack items using Next Fit Decreasing Height on a strip.

    Sort items by height (decreasing). Place items left-to-right on the
    current level. When an item doesn't fit horizontally, create a new
    level.

    Args:
        instance: A StripPackingInstance.

    Returns:
        StripPackingSolution with NFDH packing.
    """
    n = instance.n
    W = instance.strip_width

    order = sorted(range(n), key=lambda i: instance.heights[i], reverse=True)

    placements: list[StripPackingPlacement] = []
    current_x = 0.0
    current_y = 0.0
    shelf_height = 0.0

    for idx in order:
        w = instance.widths[idx]
        h = instance.heights[idx]

        if current_x + w > W + 1e-10:
            # Start new level
            current_y += shelf_height
            current_x = 0.0
            shelf_height = 0.0

        placements.append(StripPackingPlacement(item=idx, x=current_x, y=current_y))
        shelf_height = max(shelf_height, h)
        current_x += w

    # Compute total height
    total_height = current_y + shelf_height

    return StripPackingSolution(placements=placements, height=total_height)


if __name__ == "__main__":
    _inst_mod = _load_mod("spp_inst_main", os.path.join(_parent_dir, "instance.py"))
    small_spp_3 = _inst_mod.small_spp_3
    uniform_spp_6 = _inst_mod.uniform_spp_6
    validate_solution = _inst_mod.validate_solution

    print("=== Strip Packing Level Algorithms ===\n")

    for name, inst_fn in [("small3", small_spp_3), ("uniform6", uniform_spp_6)]:
        inst = inst_fn()
        for algo_name, algo in [("BL", bottom_left), ("NFDH", nfdh_strip)]:
            sol = algo(inst)
            valid, errors = validate_solution(inst, sol)
            print(f"{name} {algo_name}: height={sol.height:.1f}, valid={valid}")
