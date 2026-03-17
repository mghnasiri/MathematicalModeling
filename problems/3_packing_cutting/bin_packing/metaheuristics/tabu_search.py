"""
Tabu Search for 1D Bin Packing Problem (BPP).

Problem: BPP1D (1D Bin Packing)

Representation: A permutation of items, decoded into bins using First Fit.

Neighborhoods:
- Swap: exchange two items in the permutation
- Insert: remove an item and re-insert at a different position

Uses short-term memory preventing recently moved items from being moved
again. Aspiration criterion overrides tabu when a move yields a new
global best.

Warm-started with FFD ordering (descending size).

Complexity: O(iterations * n) per iteration.

References:
    Loh, K.H., Golden, B. & Wasil, E. (2008). Solving the
    one-dimensional bin packing problem with a weight annealing
    heuristic. Computers & Operations Research, 35(7), 2283-2291.
    https://doi.org/10.1016/j.cor.2006.10.021

    Alvim, A.C.F., Ribeiro, C.C., Glover, F. & Aloise, D.J. (2004).
    A hybrid improvement heuristic for the one-dimensional bin packing
    problem. Journal of Heuristics, 10(2), 205-229.
    https://doi.org/10.1023/B:HEUR.0000026267.44673.ed
"""

from __future__ import annotations

import os
import sys
import time
import importlib.util

import numpy as np

_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_mod(name, filepath):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_inst = _load_mod("bpp_instance_ts", os.path.join(_parent_dir, "instance.py"))
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


def tabu_search(
    instance: BinPackingInstance,
    max_iterations: int = 3000,
    tabu_tenure: int | None = None,
    time_limit: float | None = None,
    seed: int | None = None,
) -> BinPackingSolution:
    """Solve 1D Bin Packing using Tabu Search.

    Args:
        instance: Bin packing instance.
        max_iterations: Maximum iterations.
        tabu_tenure: Iterations a move stays tabu. Default: sqrt(n).
        time_limit: Time limit in seconds.
        seed: Random seed for reproducibility.

    Returns:
        Best BinPackingSolution found.
    """
    rng = np.random.default_rng(seed)
    n = instance.n
    start_time = time.time()

    if tabu_tenure is None:
        tabu_tenure = max(3, int(n ** 0.5))

    # Initialize with FFD ordering
    perm = sorted(range(n), key=lambda i: -instance.sizes[i])
    bins = _decode_ff(instance, perm)
    current_num_bins = len(bins)

    best_perm = list(perm)
    best_bins = [list(b) for b in bins]
    best_num_bins = current_num_bins

    # Tabu list: item -> iteration when tabu expires
    tabu_dict: dict[int, int] = {}

    for iteration in range(max_iterations):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        if n < 2:
            break

        best_delta = float("inf")
        best_move = None

        # Sample moves instead of full enumeration for efficiency
        n_candidates = min(n * 3, n * n)
        for _ in range(n_candidates):
            if rng.random() < 0.5 or n < 3:
                # Swap move
                i, j = rng.choice(n, size=2, replace=False)
                if perm[i] == perm[j]:
                    continue

                is_tabu = (
                    (perm[i] in tabu_dict
                     and tabu_dict[perm[i]] > iteration)
                    or (perm[j] in tabu_dict
                        and tabu_dict[perm[j]] > iteration)
                )

                new_perm = list(perm)
                new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
                new_bins = _decode_ff(instance, new_perm)
                delta = len(new_bins) - current_num_bins

                if is_tabu and current_num_bins + delta >= best_num_bins:
                    continue

                if delta < best_delta:
                    best_delta = delta
                    best_move = ("swap", i, j, new_perm, new_bins)
            else:
                # Insert move
                i = rng.integers(0, n)
                j = rng.integers(0, n)
                if i == j:
                    continue

                item = perm[i]
                is_tabu = (
                    item in tabu_dict and tabu_dict[item] > iteration
                )

                new_perm = list(perm)
                new_perm.pop(i)
                insert_pos = j if j < i else j - 1
                insert_pos = max(0, min(insert_pos, len(new_perm)))
                new_perm.insert(insert_pos, item)

                new_bins = _decode_ff(instance, new_perm)
                delta = len(new_bins) - current_num_bins

                if is_tabu and current_num_bins + delta >= best_num_bins:
                    continue

                if delta < best_delta:
                    best_delta = delta
                    best_move = ("insert", i, item, new_perm, new_bins)

        if best_move is None:
            tabu_dict.clear()
            continue

        # Apply move
        if best_move[0] == "swap":
            _, i, j, new_perm, new_bins = best_move
            tabu_dict[perm[i]] = iteration + tabu_tenure
            tabu_dict[perm[j]] = iteration + tabu_tenure
        else:  # insert
            _, i, item, new_perm, new_bins = best_move
            tabu_dict[item] = iteration + tabu_tenure

        perm = new_perm
        current_num_bins = len(new_bins)

        if current_num_bins < best_num_bins:
            best_num_bins = current_num_bins
            best_perm = list(perm)
            best_bins = [list(b) for b in new_bins]

    if not best_bins:
        best_bins = _decode_ff(instance, best_perm)
        best_num_bins = len(best_bins)

    return BinPackingSolution(bins=best_bins, num_bins=best_num_bins)


if __name__ == "__main__":
    from instance import easy_bpp_6, tight_bpp_8

    print("=== TS on easy6 ===")
    inst = easy_bpp_6()
    sol = tabu_search(inst, seed=42)
    print(f"TS: {sol.num_bins} bins, bins={sol.bins}")

    print("\n=== TS on tight8 ===")
    inst = tight_bpp_8()
    sol = tabu_search(inst, seed=42)
    print(f"TS: {sol.num_bins} bins")
