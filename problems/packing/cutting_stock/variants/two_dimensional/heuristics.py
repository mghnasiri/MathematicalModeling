"""
Two-Dimensional Cutting Stock Problem — Heuristics.

Algorithms:
    - Bottom-Left (BL) placement with First Fit Decreasing.
    - Shelf-based heuristic (Next Fit Decreasing Height).

References:
    Lodi, A., Martello, S. & Vigo, D. (1999). Heuristic and metaheuristic
    approaches for a class of two-dimensional bin packing problems. INFORMS
    Journal on Computing, 11(4), 345-357.
    https://doi.org/10.1287/ijoc.11.4.345

    Baker, B.S., Coffman, E.G. & Rivest, R.L. (1980). Orthogonal packings
    in two dimensions. SIAM Journal on Computing, 9(4), 846-855.
    https://doi.org/10.1137/0209064
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


_inst = _load_mod("twod_csp_instance_h", os.path.join(_this_dir, "instance.py"))
TwoDCSPInstance = _inst.TwoDCSPInstance
TwoDCSPSolution = _inst.TwoDCSPSolution


def _expand_demands(instance: TwoDCSPInstance) -> list[int]:
    """Expand item types by demand into individual items."""
    items = []
    for t in range(instance.num_types):
        items.extend([t] * instance.demands[t])
    return items


def bottom_left_ffd(instance: TwoDCSPInstance) -> TwoDCSPSolution:
    """Bottom-Left placement with First Fit Decreasing area.

    Items sorted by decreasing area. Each item placed at the lowest
    available position, leftmost if tied.

    Args:
        instance: 2D-CSP instance.

    Returns:
        TwoDCSPSolution.
    """
    items = _expand_demands(instance)
    # Sort by decreasing area
    items.sort(key=lambda t: instance.widths[t] * instance.heights[t],
               reverse=True)

    sheets = []  # Each sheet: list of (type, x, y, rotated)
    sheet_placements = []  # Parallel list of (x1, y1, x2, y2) rects

    for item_type in items:
        w = instance.widths[item_type]
        h = instance.heights[item_type]
        placed = False

        orientations = [(w, h, False)]
        if instance.allow_rotation and abs(w - h) > 1e-6:
            orientations.append((h, w, True))

        for sheet_idx in range(len(sheets)):
            for iw, ih, rotated in orientations:
                # Try bottom-left placement
                best_pos = None
                best_y = float("inf")
                best_x = float("inf")

                # Generate candidate positions
                candidates = [(0.0, 0.0)]
                for _, _, x2, y2 in sheet_placements[sheet_idx]:
                    candidates.append((x2, 0.0))
                    candidates.append((0.0, y2))
                    for _, _, x2b, y2b in sheet_placements[sheet_idx]:
                        candidates.append((x2, y2b))
                        candidates.append((x2b, y2))

                for cx, cy in candidates:
                    if cx + iw > instance.sheet_width + 1e-6:
                        continue
                    if cy + ih > instance.sheet_height + 1e-6:
                        continue

                    # Check no overlap
                    overlap = False
                    for px1, py1, px2, py2 in sheet_placements[sheet_idx]:
                        if not (cx + iw <= px1 + 1e-6 or px2 <= cx + 1e-6 or
                                cy + ih <= py1 + 1e-6 or py2 <= cy + 1e-6):
                            overlap = True
                            break

                    if not overlap:
                        if (cy < best_y - 1e-6 or
                                (abs(cy - best_y) < 1e-6 and cx < best_x - 1e-6)):
                            best_y = cy
                            best_x = cx
                            best_pos = (cx, cy, rotated)

                if best_pos:
                    cx, cy, rot = best_pos
                    sheets[sheet_idx].append((item_type, cx, cy, rot))
                    rw = h if rot else w
                    rh = w if rot else h
                    sheet_placements[sheet_idx].append((cx, cy, cx + rw, cy + rh))
                    placed = True
                    break
            if placed:
                break

        if not placed:
            # New sheet
            rotated = False
            iw, ih = w, h
            if (instance.allow_rotation and
                    w > instance.sheet_width + 1e-6 and h <= instance.sheet_width + 1e-6):
                iw, ih = h, w
                rotated = True
            sheets.append([(item_type, 0.0, 0.0, rotated)])
            sheet_placements.append([(0.0, 0.0, iw, ih)])

    return TwoDCSPSolution(sheets=sheets, num_sheets=len(sheets))


def shelf_nfdh(instance: TwoDCSPInstance) -> TwoDCSPSolution:
    """Next Fit Decreasing Height shelf algorithm.

    Items sorted by decreasing height. Pack into shelves on sheets;
    start a new shelf when item doesn't fit, new sheet when no shelf fits.

    Args:
        instance: 2D-CSP instance.

    Returns:
        TwoDCSPSolution.
    """
    items = _expand_demands(instance)
    # Sort by decreasing height
    items.sort(key=lambda t: instance.heights[t], reverse=True)

    sheets = []
    # Track shelves per sheet: (y_offset, shelf_height, x_used)
    sheet_shelves = []

    for item_type in items:
        w = instance.widths[item_type]
        h = instance.heights[item_type]

        # Try rotation for better fit
        use_w, use_h, rotated = w, h, False
        if instance.allow_rotation and h > w:
            use_w, use_h, rotated = h, w, True

        placed = False
        for si in range(len(sheets)):
            for shelf_idx, (y_off, shelf_h, x_used) in enumerate(sheet_shelves[si]):
                if use_h <= shelf_h + 1e-6 and x_used + use_w <= instance.sheet_width + 1e-6:
                    sheets[si].append((item_type, x_used, y_off, rotated))
                    sheet_shelves[si][shelf_idx] = (y_off, shelf_h, x_used + use_w)
                    placed = True
                    break
            if placed:
                break

            # Try new shelf on this sheet
            if sheet_shelves[si]:
                last_y = sheet_shelves[si][-1][0] + sheet_shelves[si][-1][1]
            else:
                last_y = 0.0
            if last_y + use_h <= instance.sheet_height + 1e-6:
                sheets[si].append((item_type, 0.0, last_y, rotated))
                sheet_shelves[si].append((last_y, use_h, use_w))
                placed = True
                break

        if not placed:
            sheets.append([(item_type, 0.0, 0.0, rotated)])
            sheet_shelves.append([(0.0, use_h, use_w)])

    return TwoDCSPSolution(sheets=sheets, num_sheets=len(sheets))


if __name__ == "__main__":
    from instance import small_2dcsp_4

    inst = small_2dcsp_4()
    sol1 = bottom_left_ffd(inst)
    print(f"BL-FFD: {sol1}")
    sol2 = shelf_nfdh(inst)
    print(f"Shelf NFDH: {sol2}")
