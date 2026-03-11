"""
Simulated Annealing for 1D Bin Packing Problem (BPP).

Problem: BPP1D (1D Bin Packing)

Representation: A permutation of items. The permutation is decoded into
bins using First Fit Decreasing (FFD): process items in permutation order,
placing each into the first bin where it fits.

Neighborhoods:
- Swap: swap two items in the permutation
- Insert: remove an item and re-insert it at a different position

Warm-started with FFD ordering (descending size).

Complexity: O(iterations * n) per iteration.

References:
    Kirkpatrick, S., Gelatt, C.D. & Vecchi, M.P. (1983). Optimization
    by simulated annealing. Science, 220(4598), 671-680.
    https://doi.org/10.1126/science.220.4598.671

    Loh, K.H., Golden, B. & Wasil, E. (2008). Solving the one-dimensional
    bin packing problem with a weight annealing heuristic. Computers &
    Operations Research, 35(7), 2283-2291.
    https://doi.org/10.1016/j.cor.2006.10.021
"""

from __future__ import annotations

import os
import sys
import math
import time
import importlib.util

import numpy as np

_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_mod(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_inst = _load_mod("bpp_instance_sa", os.path.join(_parent_dir, "instance.py"))
BinPackingInstance = _inst.BinPackingInstance
BinPackingSolution = _inst.BinPackingSolution


def _decode_ff(
    instance: BinPackingInstance,
    perm: list[int],
) -> list[list[int]]:
    """Decode a permutation into bins using First Fit."""
    bins: list[list[int]] = []
    bin_remaining: list[float] = []

    for item in perm:
        size = instance.sizes[item]
        placed = False
        for b in range(len(bins)):
            if bin_remaining[b] >= size - 1e-10:
                bins[b].append(item)
                bin_remaining[b] -= size
                placed = True
                break
        if not placed:
            bins.append([item])
            bin_remaining.append(instance.capacity - size)

    return bins


def simulated_annealing(
    instance: BinPackingInstance,
    max_iterations: int = 10000,
    initial_temp: float | None = None,
    cooling_rate: float = 0.999,
    time_limit: float | None = None,
    seed: int | None = None,
) -> BinPackingSolution:
    """Solve 1D Bin Packing using Simulated Annealing.

    Args:
        instance: Bin packing instance.
        max_iterations: Maximum iterations.
        initial_temp: Initial temperature. If None, auto-calibrated.
        cooling_rate: Geometric cooling factor.
        time_limit: Time limit in seconds.
        seed: Random seed for reproducibility.

    Returns:
        Best BinPackingSolution found.
    """
    rng = np.random.default_rng(seed)
    n = instance.n
    start_time = time.time()

    # ── Initialize with FFD ordering ─────────────────────────────────────
    perm = sorted(range(n), key=lambda i: -instance.sizes[i])
    bins = _decode_ff(instance, perm)
    current_num_bins = len(bins)

    best_perm = list(perm)
    best_bins = [list(b) for b in bins]
    best_num_bins = current_num_bins

    if initial_temp is None:
        initial_temp = max(1.0, n * 0.5)
    temp = initial_temp

    for iteration in range(max_iterations):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        if n < 2:
            break  # Nothing to permute

        # Choose neighborhood
        new_perm = list(perm)

        if rng.random() < 0.5 or n < 3:
            # Swap two items
            i, j = rng.choice(n, size=2, replace=False)
            new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
        else:
            # Insert: remove and re-insert
            i = rng.integers(0, n)
            item = new_perm.pop(i)
            j = rng.integers(0, n)
            if j >= i:
                j = max(0, j - 1)
            new_perm.insert(j, item)

        new_bins = _decode_ff(instance, new_perm)
        new_num_bins = len(new_bins)

        delta = new_num_bins - current_num_bins

        if delta <= 0 or rng.random() < math.exp(-delta / max(temp, 1e-10)):
            perm = new_perm
            current_num_bins = new_num_bins

            if current_num_bins < best_num_bins:
                best_num_bins = current_num_bins
                best_perm = list(perm)
                best_bins = [list(b) for b in new_bins]

        temp *= cooling_rate

    # Re-decode best permutation to get final bins
    if not best_bins:
        best_bins = _decode_ff(instance, best_perm)
        best_num_bins = len(best_bins)

    return BinPackingSolution(bins=best_bins, num_bins=best_num_bins)


if __name__ == "__main__":
    from instance import easy_bpp_6, tight_bpp_8

    print("=== SA on easy6 ===")
    inst = easy_bpp_6()
    sol = simulated_annealing(inst, seed=42)
    print(f"SA: {sol.num_bins} bins, bins={sol.bins}")

    print("\n=== SA on tight8 ===")
    inst = tight_bpp_8()
    sol = simulated_annealing(inst, seed=42)
    print(f"SA: {sol.num_bins} bins")
