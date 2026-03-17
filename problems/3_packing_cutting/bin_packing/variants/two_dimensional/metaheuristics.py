"""
Simulated Annealing for 2D Strip Packing.

Problem: 2D-BPP (strip packing)

Permutation encoding: order in which items are placed using the
bottom-left heuristic as a decoder. SA explores permutation neighborhood
(swap, insert moves).

Warm-started with BLDH heuristic.

Complexity: O(iterations * n^2) per run.

References:
    Jakobs, S. (1996). On genetic algorithms for the packing of polygons.
    European Journal of Operational Research, 88(1), 165-181.
    https://doi.org/10.1016/0377-2217(94)00166-9

    Hopper, E. & Turton, B.C.H. (2001). An empirical investigation of
    meta-heuristic and heuristic algorithms for a 2D packing problem.
    European Journal of Operational Research, 128(1), 34-57.
    https://doi.org/10.1016/S0377-2217(99)00357-4
"""

from __future__ import annotations

import sys
import os
import math
import time
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


_inst = _load_mod("bpp2d_instance_meta", os.path.join(_this_dir, "instance.py"))
BPP2DInstance = _inst.BPP2DInstance
BPP2DSolution = _inst.BPP2DSolution


def _decode_bl(instance: BPP2DInstance, perm: list[int]) -> BPP2DSolution:
    """Decode a permutation into a 2D packing using bottom-left placement."""
    n = instance.n
    W = instance.strip_width
    positions = [(0.0, 0.0)] * n
    placed: list[int] = []

    for idx in perm:
        w, h = instance.widths[idx], instance.heights[idx]
        best_x, best_y = 0.0, float("inf")

        candidates = [(0.0, 0.0)]
        for p in placed:
            px, py = positions[p]
            pw, ph = instance.widths[p], instance.heights[p]
            candidates.append((px + pw, py))
            candidates.append((px, py + ph))
            candidates.append((0.0, py + ph))

        for cx, cy in candidates:
            if cx + w > W + 1e-10:
                continue
            overlap = False
            for p in placed:
                px, py = positions[p]
                pw, ph = instance.widths[p], instance.heights[p]
                if (cx < px + pw - 1e-10 and px < cx + w - 1e-10 and
                        cy < py + ph - 1e-10 and py < cy + h - 1e-10):
                    overlap = True
                    break
            if not overlap:
                if cy < best_y - 1e-10 or (abs(cy - best_y) < 1e-10 and cx < best_x):
                    best_y = cy
                    best_x = cx

        if best_y == float("inf"):
            max_h = max(
                (positions[p][1] + instance.heights[p] for p in placed),
                default=0.0,
            )
            best_x, best_y = 0.0, max_h

        positions[idx] = (best_x, best_y)
        placed.append(idx)

    height = max(positions[i][1] + instance.heights[i] for i in range(n))
    return BPP2DSolution(positions=positions, height=height)


def simulated_annealing(
    instance: BPP2DInstance,
    max_iterations: int = 30000,
    initial_temp: float | None = None,
    cooling_rate: float = 0.9995,
    time_limit: float | None = None,
    seed: int | None = None,
) -> BPP2DSolution:
    """Solve 2D strip packing using Simulated Annealing.

    Args:
        instance: A BPP2DInstance.
        max_iterations: Maximum iterations.
        initial_temp: Initial temperature.
        cooling_rate: Geometric cooling factor.
        time_limit: Time limit in seconds.
        seed: Random seed.

    Returns:
        BPP2DSolution.
    """
    rng = np.random.default_rng(seed)
    n = instance.n
    start_time = time.time()

    # Initial permutation: sorted by height descending
    perm = sorted(range(n), key=lambda i: instance.heights[i], reverse=True)
    current_sol = _decode_bl(instance, perm)
    current_h = current_sol.height

    best_perm = perm[:]
    best_sol = current_sol
    best_h = current_h

    if initial_temp is None:
        initial_temp = current_h * 0.2

    temp = initial_temp

    for iteration in range(max_iterations):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        new_perm = perm[:]
        move = rng.integers(0, 2)

        if move == 0:
            # Swap
            i, j = rng.choice(n, 2, replace=False)
            new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
        else:
            # Insert
            i = rng.integers(0, n)
            item = new_perm.pop(i)
            j = rng.integers(0, len(new_perm) + 1)
            new_perm.insert(j, item)

        new_sol = _decode_bl(instance, new_perm)
        new_h = new_sol.height

        delta = new_h - current_h
        if delta < 0 or (temp > 0 and rng.random() < math.exp(-delta / max(temp, 1e-10))):
            perm = new_perm
            current_h = new_h

            if current_h < best_h - 1e-10:
                best_h = current_h
                best_perm = perm[:]
                best_sol = new_sol

        temp *= cooling_rate

    return best_sol


if __name__ == "__main__":
    inst = BPP2DInstance.random(n=10, seed=42)
    print(f"2D-BPP: {inst.n} items, W={inst.strip_width}")
    print(f"  Area LB: {inst.area_lower_bound():.1f}")

    _heur = _load_mod("bpp2d_heur_meta", os.path.join(_this_dir, "heuristics.py"))
    bldh_sol = _heur.bottom_left_dh(inst)
    print(f"BLDH: height={bldh_sol.height:.1f}")

    sa_sol = simulated_annealing(inst, seed=42)
    print(f"SA: height={sa_sol.height:.1f}")
