"""
Iterated Greedy for 1D Bin Packing.

Problem notation: BPP1D

Iterated Greedy repeatedly destroys and reconstructs solutions:
    1. Destroy: remove d random items from bins
    2. Repair: reinsert using First Fit Decreasing on the removed items
    3. Accept: keep if number of bins is reduced or equal

Warm-started with FFD heuristic.

Complexity: O(iterations * d * B) per run.

References:
    Ruiz, R. & Stuetzle, T. (2007). A simple and effective iterated
    greedy algorithm for the permutation flowshop scheduling problem.
    European Journal of Operational Research, 177(3), 2033-2049.
    https://doi.org/10.1016/j.ejor.2005.12.009

    Johnson, D.S. (1973). Near-optimal bin packing algorithms.
    Ph.D. thesis, MIT, Cambridge, MA.
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


_inst = _load_mod("bpp_instance_ig", os.path.join(_parent_dir, "instance.py"))
BinPackingInstance = _inst.BinPackingInstance
BinPackingSolution = _inst.BinPackingSolution

_ff = _load_mod(
    "bpp_ff_ig",
    os.path.join(_parent_dir, "heuristics", "first_fit.py"),
)
first_fit_decreasing = _ff.first_fit_decreasing


def iterated_greedy(
    instance: BinPackingInstance,
    max_iterations: int = 3000,
    d: int | None = None,
    time_limit: float | None = None,
    seed: int | None = None,
) -> BinPackingSolution:
    """Solve 1D Bin Packing using Iterated Greedy.

    Args:
        instance: A BinPackingInstance.
        max_iterations: Maximum number of iterations.
        d: Number of items to remove per iteration. Defaults to max(1, n//4).
        time_limit: Maximum wall-clock time in seconds.
        seed: Random seed for reproducibility.

    Returns:
        BinPackingSolution with the best packing found.
    """
    rng = np.random.default_rng(seed)
    n = instance.n
    start_time = time.time()

    if d is None:
        d = max(1, n // 4)

    # Warm-start with FFD
    init_sol = first_fit_decreasing(instance)
    bins = [b[:] for b in init_sol.bins]
    bins = [b for b in bins if b]

    best_bins = [b[:] for b in bins]
    best_num = len(bins)

    for iteration in range(max_iterations):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        # Destroy: remove d random items
        all_items = [(bi, ci, item)
                     for bi, b in enumerate(bins)
                     for ci, item in enumerate(b)]
        if len(all_items) == 0:
            break

        d_actual = min(d, len(all_items))
        chosen = rng.choice(len(all_items), size=d_actual, replace=False)
        removed = []
        # Sort in reverse to safely remove
        remove_info = sorted(
            [all_items[i] for i in chosen],
            key=lambda x: (x[0], x[1]),
            reverse=True,
        )
        for bi, ci, item in remove_info:
            bins[bi].pop(ci)
            removed.append(item)

        bins = [b for b in bins if b]

        # Repair: reinsert removed items using FFD strategy
        removed.sort(key=lambda i: instance.sizes[i], reverse=True)
        remaining = [
            instance.capacity - sum(instance.sizes[i] for i in b)
            for b in bins
        ]

        for item in removed:
            # Best fit: find bin with smallest remaining capacity that fits
            best_bin = -1
            best_rem = float("inf")
            for bi in range(len(bins)):
                if remaining[bi] >= instance.sizes[item] - 1e-10:
                    if remaining[bi] < best_rem:
                        best_rem = remaining[bi]
                        best_bin = bi

            if best_bin >= 0:
                bins[best_bin].append(item)
                remaining[best_bin] -= instance.sizes[item]
            else:
                bins.append([item])
                remaining.append(instance.capacity - instance.sizes[item])

        bins = [b for b in bins if b]

        if len(bins) <= best_num:
            best_num = len(bins)
            best_bins = [b[:] for b in bins]

    best_bins = [b for b in best_bins if b]
    return BinPackingSolution(
        bins=best_bins,
        num_bins=len(best_bins),
    )


if __name__ == "__main__":
    inst = BinPackingInstance.random(n=20, seed=42)
    print(f"Bin Packing: {inst.n} items, capacity={inst.capacity}")
    print(f"L1 lower bound: {inst.lower_bound_l1()}")

    ffd_sol = first_fit_decreasing(inst)
    print(f"FFD: {ffd_sol.num_bins} bins")

    ig_sol = iterated_greedy(inst, seed=42)
    print(f"IG: {ig_sol.num_bins} bins")
