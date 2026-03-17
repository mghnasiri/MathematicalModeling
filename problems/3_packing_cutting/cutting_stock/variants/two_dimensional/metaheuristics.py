"""
Two-Dimensional Cutting Stock Problem — Metaheuristics.

Algorithms:
    - Simulated Annealing with item reordering and rotation toggling.

References:
    Lodi, A., Martello, S. & Vigo, D. (1999). Heuristic and metaheuristic
    approaches for a class of two-dimensional bin packing problems. INFORMS
    Journal on Computing, 11(4), 345-357.
    https://doi.org/10.1287/ijoc.11.4.345
"""

from __future__ import annotations

import math
import sys
import os
import importlib.util
import time

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


_inst = _load_mod("twod_csp_instance_m", os.path.join(_this_dir, "instance.py"))
TwoDCSPInstance = _inst.TwoDCSPInstance
TwoDCSPSolution = _inst.TwoDCSPSolution

_heur = _load_mod("twod_csp_heuristics_m", os.path.join(_this_dir, "heuristics.py"))
bottom_left_ffd = _heur.bottom_left_ffd


def _decode_sequence(
    instance: TwoDCSPInstance,
    sequence: list[int],
    rotations: list[bool],
) -> TwoDCSPSolution:
    """Decode an item sequence + rotation flags into a solution via shelf packing."""
    sheets = []
    sheet_shelves = []  # [(y_offset, shelf_height, x_used)]

    for idx in sequence:
        item_type = idx  # idx is the expanded item index, but we use type
        w = instance.widths[item_type]
        h = instance.heights[item_type]
        rotated = rotations[idx] if instance.allow_rotation else False
        if rotated:
            w, h = h, w

        placed = False
        for si in range(len(sheets)):
            # Try existing shelves
            for shelf_idx, (y_off, shelf_h, x_used) in enumerate(sheet_shelves[si]):
                if h <= shelf_h + 1e-6 and x_used + w <= instance.sheet_width + 1e-6:
                    sheets[si].append((item_type, x_used, y_off, rotated))
                    sheet_shelves[si][shelf_idx] = (y_off, shelf_h, x_used + w)
                    placed = True
                    break
            if placed:
                break

            # New shelf on this sheet
            if sheet_shelves[si]:
                last_y = sheet_shelves[si][-1][0] + sheet_shelves[si][-1][1]
            else:
                last_y = 0.0
            if last_y + h <= instance.sheet_height + 1e-6:
                sheets[si].append((item_type, 0.0, last_y, rotated))
                sheet_shelves[si].append((last_y, h, w))
                placed = True
                break

        if not placed:
            sheets.append([(item_type, 0.0, 0.0, rotated)])
            sheet_shelves.append([(0.0, h, w)])

    return TwoDCSPSolution(sheets=sheets, num_sheets=len(sheets))


def simulated_annealing(
    instance: TwoDCSPInstance,
    max_iterations: int = 50000,
    cooling_rate: float = 0.9995,
    seed: int | None = None,
    time_limit: float | None = None,
) -> TwoDCSPSolution:
    """SA for 2D-CSP with sequence reordering and rotation toggling.

    Args:
        instance: 2D-CSP instance.
        max_iterations: Maximum iterations.
        cooling_rate: Temperature decay factor.
        seed: Random seed.
        time_limit: Time limit in seconds.

    Returns:
        TwoDCSPSolution.
    """
    rng = np.random.default_rng(seed)

    # Expand demands into item list
    items = []
    for t in range(instance.num_types):
        items.extend([t] * instance.demands[t])
    n_items = len(items)

    # Initial sequence: sorted by decreasing area
    sequence = list(range(instance.num_types))
    sequence.sort(key=lambda t: instance.widths[t] * instance.heights[t],
                  reverse=True)
    # Expand
    expanded_seq = []
    for t in sequence:
        expanded_seq.extend([t] * instance.demands[t])

    rotations = [False] * instance.num_types

    sol = _decode_sequence(instance, expanded_seq, rotations)
    cost = sol.num_sheets

    best_seq = list(expanded_seq)
    best_rot = list(rotations)
    best_cost = cost

    temp = max(1.0, best_cost * 0.3)
    start = time.time()

    for it in range(max_iterations):
        if time_limit and time.time() - start > time_limit:
            break

        new_seq = list(expanded_seq)
        new_rot = list(rotations)
        move = rng.integers(0, 2)

        if move == 0:
            # Swap two items in sequence
            i = int(rng.integers(0, n_items))
            j = int(rng.integers(0, n_items - 1))
            if j >= i:
                j += 1
            new_seq[i], new_seq[j] = new_seq[j], new_seq[i]

        elif move == 1 and instance.allow_rotation:
            # Toggle rotation of a random item type
            t = int(rng.integers(0, instance.num_types))
            new_rot[t] = not new_rot[t]

        new_sol = _decode_sequence(instance, new_seq, new_rot)
        new_cost = new_sol.num_sheets
        delta = new_cost - cost

        if delta < 0 or rng.random() < math.exp(-delta / max(temp, 1e-15)):
            expanded_seq = new_seq
            rotations = new_rot
            cost = new_cost
            if cost < best_cost:
                best_cost = cost
                best_seq = list(expanded_seq)
                best_rot = list(rotations)

        temp *= cooling_rate

    return _decode_sequence(instance, best_seq, best_rot)


if __name__ == "__main__":
    from instance import small_2dcsp_4

    inst = small_2dcsp_4()
    sol = simulated_annealing(inst, max_iterations=10000, seed=42)
    print(f"SA: {sol}")
