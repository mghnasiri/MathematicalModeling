"""
Variable Neighborhood Search for 1D Bin Packing.

Problem notation: BPP1D

VNS uses multiple neighborhood structures:
    N1: Move — move an item from one bin to another
    N2: Swap — exchange items between two bins
    N3: Multi-move — move k items simultaneously

Local search uses best-improvement move on the most-loaded bin.
Warm-started with FFD heuristic.

Complexity: O(iterations * k_max * n * B) per run.

References:
    Mladenović, N. & Hansen, P. (1997). Variable neighborhood search.
    Computers & Operations Research, 24(11), 1097-1100.
    https://doi.org/10.1016/S0305-0548(97)00031-2

    Fleszar, K. & Hindi, K.S. (2002). New heuristics for
    one-dimensional bin-packing. Computers & Operations Research,
    29(7), 821-839.
    https://doi.org/10.1016/S0305-0548(01)00065-1
"""

from __future__ import annotations

import sys
import os
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


_inst = _load_mod("bpp_instance_vns", os.path.join(_parent_dir, "instance.py"))
BinPackingInstance = _inst.BinPackingInstance
BinPackingSolution = _inst.BinPackingSolution

_ff = _load_mod(
    "bpp_ff_vns",
    os.path.join(_parent_dir, "heuristics", "first_fit.py"),
)
first_fit_decreasing = _ff.first_fit_decreasing


def vns(
    instance: BinPackingInstance,
    max_iterations: int = 500,
    k_max: int = 3,
    time_limit: float | None = None,
    seed: int | None = None,
) -> BinPackingSolution:
    """Solve 1D Bin Packing using Variable Neighborhood Search.

    Args:
        instance: A BinPackingInstance.
        max_iterations: Maximum number of iterations.
        k_max: Maximum neighborhood size.
        time_limit: Maximum wall-clock time in seconds.
        seed: Random seed for reproducibility.

    Returns:
        BinPackingSolution with the best packing found.
    """
    rng = np.random.default_rng(seed)
    start_time = time.time()

    # Warm-start with FFD
    init_sol = first_fit_decreasing(instance)
    bins = [b[:] for b in init_sol.bins]
    bins = [b for b in bins if b]
    remaining = _compute_remaining(instance, bins)

    best_bins = [b[:] for b in bins]
    best_num = len(bins)

    for iteration in range(max_iterations):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        k = 1
        while k <= k_max:
            if time_limit is not None and time.time() - start_time >= time_limit:
                break

            # Shaking
            shaken = [b[:] for b in bins]
            _shake(instance, shaken, k, rng)
            shaken = [b for b in shaken if b]

            # Local search: try to eliminate bins
            _local_search(instance, shaken)
            shaken = [b for b in shaken if b]

            if len(shaken) < len(bins):
                bins = shaken
                remaining = _compute_remaining(instance, bins)
                k = 1

                if len(bins) < best_num:
                    best_num = len(bins)
                    best_bins = [b[:] for b in bins]
            else:
                k += 1

    best_bins = [b for b in best_bins if b]
    return BinPackingSolution(
        bins=best_bins,
        num_bins=len(best_bins),
    )


def _compute_remaining(
    instance: BinPackingInstance, bins: list[list[int]]
) -> list[float]:
    """Compute remaining capacity for each bin."""
    result = []
    for b in bins:
        used = sum(instance.sizes[i] for i in b)
        result.append(instance.capacity - used)
    return result


def _shake(
    instance: BinPackingInstance,
    bins: list[list[int]],
    k: int,
    rng: np.random.Generator,
) -> None:
    """Shake: perform k random moves between bins."""
    for _ in range(k):
        non_empty = [i for i in range(len(bins)) if bins[i]]
        if len(non_empty) < 2:
            break
        src_idx = non_empty[rng.integers(len(non_empty))]
        if not bins[src_idx]:
            continue
        item_idx = rng.integers(len(bins[src_idx]))
        item = bins[src_idx][item_idx]

        # Try to move to a random different bin
        others = [i for i in range(len(bins)) if i != src_idx]
        if not others:
            continue
        dst_idx = others[rng.integers(len(others))]

        used = sum(instance.sizes[i] for i in bins[dst_idx])
        if used + instance.sizes[item] <= instance.capacity + 1e-10:
            bins[src_idx].pop(item_idx)
            bins[dst_idx].append(item)


def _local_search(
    instance: BinPackingInstance,
    bins: list[list[int]],
) -> None:
    """Local search: try to empty bins by redistributing their items."""
    improved = True
    while improved:
        improved = False
        bins_sorted = sorted(
            range(len(bins)),
            key=lambda i: sum(instance.sizes[j] for j in bins[i]),
        )

        for bi in bins_sorted:
            if not bins[bi]:
                continue
            # Try to redistribute all items from this bin to others
            items = bins[bi][:]
            remaining = []
            for j in range(len(bins)):
                if j == bi:
                    remaining.append(0.0)
                else:
                    used = sum(instance.sizes[i] for i in bins[j])
                    remaining.append(instance.capacity - used)

            # Sort items largest first for greedy placement
            items_sorted = sorted(items, key=lambda i: instance.sizes[i], reverse=True)
            placement = {}
            rem_copy = remaining[:]

            success = True
            for item in items_sorted:
                placed = False
                # Find bin with smallest remaining capacity that fits
                best_bin = -1
                best_rem = float("inf")
                for j in range(len(bins)):
                    if j == bi:
                        continue
                    if rem_copy[j] >= instance.sizes[item] - 1e-10:
                        if rem_copy[j] < best_rem:
                            best_rem = rem_copy[j]
                            best_bin = j
                if best_bin >= 0:
                    placement[item] = best_bin
                    rem_copy[best_bin] -= instance.sizes[item]
                else:
                    success = False
                    break

            if success:
                # Apply redistribution
                bins[bi].clear()
                for item, target in placement.items():
                    bins[target].append(item)
                improved = True
                break

        # Clean up empty bins
        bins[:] = [b for b in bins if b]


if __name__ == "__main__":
    inst = BinPackingInstance.random(n=20, seed=42)
    print(f"Bin Packing: {inst.n} items, capacity={inst.capacity}")
    print(f"L1 lower bound: {inst.lower_bound_l1()}")

    ffd_sol = first_fit_decreasing(inst)
    print(f"FFD: {ffd_sol.num_bins} bins")

    vns_sol = vns(inst, seed=42)
    print(f"VNS: {vns_sol.num_bins} bins")
