"""
Shelf Algorithms — Heuristics for the 2D Bin Packing Problem.

Problem: 2D Bin Packing (2D-BPP), oriented (no rotation)
Complexity: O(n log n) for sorting + O(n * B) for placement

Next Fit Decreasing Height (NFDH): Sort items by height (decreasing).
Place items left-to-right on the current shelf. When an item doesn't
fit horizontally, start a new shelf. When a shelf doesn't fit vertically,
start a new bin.

First Fit Decreasing Height (FFDH): Sort items by height (decreasing).
For each item, try to place it on the first shelf (across all bins) where
it fits. If no shelf fits, create a new shelf.

References:
    Coffman, E.G., Garey, M.R., Johnson, D.S. & Tarjan, R.E. (1980).
    Performance bounds for level-oriented two-dimensional packing
    algorithms. SIAM Journal on Computing, 9(4), 808-826.
    https://doi.org/10.1137/0209062

    Lodi, A., Martello, S. & Vigo, D. (2002). Two-dimensional packing
    problems: A survey. European Journal of Operational Research,
    141(2), 241-252.
    https://doi.org/10.1016/S0377-2217(02)00123-6
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


_inst = _load_mod("bp2d_instance_shelf", os.path.join(_parent_dir, "instance.py"))
BinPacking2DInstance = _inst.BinPacking2DInstance
BinPacking2DSolution = _inst.BinPacking2DSolution
Placement = _inst.Placement


def nfdh(instance: BinPacking2DInstance) -> BinPacking2DSolution:
    """Pack items using Next Fit Decreasing Height (NFDH).

    Sort items by height (decreasing). Place left-to-right on the current
    shelf. When an item doesn't fit horizontally, close the shelf. When
    the new shelf doesn't fit vertically, open a new bin.

    Args:
        instance: A BinPacking2DInstance.

    Returns:
        BinPacking2DSolution with NFDH packing.
    """
    n = instance.n
    W = instance.bin_width
    H = instance.bin_height

    # Sort items by height descending
    order = sorted(range(n), key=lambda i: instance.heights[i], reverse=True)

    bins: list[list[Placement]] = [[]]
    # Track current bin's shelf state
    current_x = 0.0
    current_y = 0.0
    shelf_height = 0.0

    for idx in order:
        w = instance.widths[idx]
        h = instance.heights[idx]

        # Try to place on current shelf
        if current_x + w <= W + 1e-10:
            # Fits on current shelf
            if current_y + h > H + 1e-10:
                # Doesn't fit vertically even as new shelf - need new bin
                bins.append([])
                current_x = 0.0
                current_y = 0.0
                shelf_height = 0.0

            bins[-1].append(Placement(item=idx, x=current_x, y=current_y))
            shelf_height = max(shelf_height, h)
            current_x += w
        else:
            # Start a new shelf
            new_y = current_y + shelf_height
            if new_y + h > H + 1e-10:
                # New shelf doesn't fit - need new bin
                bins.append([])
                new_y = 0.0

            bins[-1].append(Placement(item=idx, x=0.0, y=new_y))
            current_x = w
            current_y = new_y
            shelf_height = h

    # Remove empty bins
    bins = [b for b in bins if len(b) > 0]

    return BinPacking2DSolution(bins=bins, num_bins=len(bins))


def ffdh(instance: BinPacking2DInstance) -> BinPacking2DSolution:
    """Pack items using First Fit Decreasing Height (FFDH).

    Sort items by height (decreasing). For each item, try to place it on
    the first existing shelf where it fits (horizontally and vertically).
    If no shelf fits, create a new shelf in the first bin with room, or
    open a new bin.

    Args:
        instance: A BinPacking2DInstance.

    Returns:
        BinPacking2DSolution with FFDH packing.
    """
    n = instance.n
    W = instance.bin_width
    H = instance.bin_height

    # Sort items by height descending
    order = sorted(range(n), key=lambda i: instance.heights[i], reverse=True)

    # Each bin has a list of shelves: (y_offset, shelf_height, current_x)
    bin_shelves: list[list[list[float]]] = []  # [bin_idx][shelf_idx] = [y, h, x]
    bins: list[list[Placement]] = []

    for idx in order:
        w = instance.widths[idx]
        h = instance.heights[idx]

        placed = False

        # Try each existing shelf in each bin
        for bi in range(len(bins)):
            for si in range(len(bin_shelves[bi])):
                shelf = bin_shelves[bi][si]
                sy, sh, sx = shelf[0], shelf[1], shelf[2]
                if sx + w <= W + 1e-10 and h <= sh + 1e-10:
                    bins[bi].append(Placement(item=idx, x=sx, y=sy))
                    shelf[2] = sx + w
                    placed = True
                    break
            if placed:
                break

        if not placed:
            # Try to create a new shelf in an existing bin
            for bi in range(len(bins)):
                shelves = bin_shelves[bi]
                if len(shelves) == 0:
                    next_y = 0.0
                else:
                    last = shelves[-1]
                    next_y = last[0] + last[1]
                if next_y + h <= H + 1e-10:
                    bin_shelves[bi].append([next_y, h, w])
                    bins[bi].append(Placement(item=idx, x=0.0, y=next_y))
                    placed = True
                    break

        if not placed:
            # Open a new bin
            bins.append([Placement(item=idx, x=0.0, y=0.0)])
            bin_shelves.append([[0.0, h, w]])

    return BinPacking2DSolution(bins=bins, num_bins=len(bins))


if __name__ == "__main__":
    _inst_mod = _load_mod("bp2d_inst_main", os.path.join(_parent_dir, "instance.py"))
    small_2dbpp_4 = _inst_mod.small_2dbpp_4
    uniform_2dbpp_6 = _inst_mod.uniform_2dbpp_6
    validate_solution = _inst_mod.validate_solution

    print("=== 2D Bin Packing Shelf Algorithms ===\n")

    for name, inst_fn in [("small4", small_2dbpp_4), ("uniform6", uniform_2dbpp_6)]:
        inst = inst_fn()
        for algo_name, algo in [("NFDH", nfdh), ("FFDH", ffdh)]:
            sol = algo(inst)
            valid, errors = validate_solution(inst, sol)
            print(f"{name} {algo_name}: {sol.num_bins} bins, valid={valid}")
