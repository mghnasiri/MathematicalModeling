"""
Constructive Heuristics for 2D Bin Packing (Strip Packing).

Problem: 2D-BPP
Complexity: O(n^2) for shelf heuristics

1. Bottom-Left Decreasing Height (BLDH): sort by height descending,
   place each item at the lowest available position, leftmost first.
2. Next-Fit Decreasing Height (NFDH): shelf-based, place items on
   current shelf; open new shelf when item doesn't fit.

References:
    Coffman, E.G., Garey, M.R., Johnson, D.S. & Tarjan, R.E. (1980).
    Performance bounds for level-oriented two-dimensional packing
    algorithms. SIAM Journal on Computing, 9(4), 808-826.
    https://doi.org/10.1137/0209062

    Lodi, A., Martello, S. & Vigo, D. (2002). Two-dimensional packing
    problems: A survey. European Journal of Operational Research, 141(2),
    241-252. https://doi.org/10.1016/S0377-2217(02)00123-6
"""

from __future__ import annotations

import sys
import os
import importlib.util

import numpy as np

_this_dir = os.path.dirname(os.path.abspath(__file__))


def _load_mod(name, filepath):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_inst = _load_mod("bpp2d_instance_h", os.path.join(_this_dir, "instance.py"))
BPP2DInstance = _inst.BPP2DInstance
BPP2DSolution = _inst.BPP2DSolution


def bottom_left_dh(instance: BPP2DInstance) -> BPP2DSolution:
    """Bottom-Left Decreasing Height heuristic.

    Sort items by height (decreasing), then place each at the
    bottom-left-most available position.

    Args:
        instance: A BPP2DInstance.

    Returns:
        BPP2DSolution.
    """
    n = instance.n
    W = instance.strip_width
    order = sorted(range(n), key=lambda i: instance.heights[i], reverse=True)

    positions = [(0.0, 0.0)] * n
    placed = []

    for idx in order:
        w, h = instance.widths[idx], instance.heights[idx]
        best_x, best_y = 0.0, float("inf")

        # Try candidate positions: bottom-left corners formed by placed items
        candidates = [(0.0, 0.0)]
        for p_idx in placed:
            px, py = positions[p_idx]
            pw, ph = instance.widths[p_idx], instance.heights[p_idx]
            candidates.append((px + pw, py))
            candidates.append((px, py + ph))
            candidates.append((0.0, py + ph))

        for cx, cy in candidates:
            if cx + w > W + 1e-10:
                continue

            # Check overlap with all placed items
            overlap = False
            for p_idx in placed:
                px, py = positions[p_idx]
                pw, ph = instance.widths[p_idx], instance.heights[p_idx]
                if (cx < px + pw - 1e-10 and px < cx + w - 1e-10 and
                        cy < py + ph - 1e-10 and py < cy + h - 1e-10):
                    overlap = True
                    break

            if not overlap:
                if cy < best_y - 1e-10 or (abs(cy - best_y) < 1e-10 and cx < best_x):
                    best_y = cy
                    best_x = cx

        if best_y == float("inf"):
            # Fallback: stack on top
            max_h = max(
                (positions[p][1] + instance.heights[p] for p in placed),
                default=0.0,
            )
            best_x, best_y = 0.0, max_h

        positions[idx] = (best_x, best_y)
        placed.append(idx)

    height = max(positions[i][1] + instance.heights[i] for i in range(n))
    return BPP2DSolution(positions=positions, height=height)


def nfdh(instance: BPP2DInstance) -> BPP2DSolution:
    """Next-Fit Decreasing Height shelf heuristic.

    Sort items by height (decreasing). Place on current shelf if it fits;
    otherwise open a new shelf at the height of the tallest item on the
    current shelf.

    Args:
        instance: A BPP2DInstance.

    Returns:
        BPP2DSolution.
    """
    n = instance.n
    W = instance.strip_width
    order = sorted(range(n), key=lambda i: instance.heights[i], reverse=True)

    positions = [(0.0, 0.0)] * n
    shelf_y = 0.0
    shelf_height = 0.0
    shelf_x = 0.0

    for idx in order:
        w, h = instance.widths[idx], instance.heights[idx]

        if shelf_x + w > W + 1e-10:
            # Open new shelf
            shelf_y += shelf_height
            shelf_x = 0.0
            shelf_height = 0.0

        positions[idx] = (shelf_x, shelf_y)
        shelf_x += w
        shelf_height = max(shelf_height, h)

    height = max(positions[i][1] + instance.heights[i] for i in range(n))
    return BPP2DSolution(positions=positions, height=height)


if __name__ == "__main__":
    inst = _inst.small_2dbpp_5()
    print(f"2D-BPP: {inst.n} items, W={inst.strip_width}")

    sol1 = bottom_left_dh(inst)
    print(f"BLDH: height={sol1.height:.1f}")

    sol2 = nfdh(inst)
    print(f"NFDH: height={sol2.height:.1f}")
