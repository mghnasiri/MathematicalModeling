"""
First Fit Decreasing (FFD) and Best Fit Decreasing (BFD) — Bin Packing heuristics.

Problem: 1D Bin Packing (BPP)
Complexity: O(n log n) for sorting + O(n * B) for placement

FFD: Sort items by size (decreasing). For each item, place it in the
first bin that has enough remaining capacity. If no bin fits, open a
new bin.

BFD: Sort items by size (decreasing). For each item, place it in the
bin with the least remaining capacity that still fits the item.

FFD approximation: FFD(I) <= (11/9) * OPT(I) + 6/9 (Dósa, 2007).

References:
    Johnson, D.S. (1973). Near-optimal bin packing algorithms.
    Ph.D. thesis, MIT, Cambridge, MA.

    Johnson, D.S., Demers, A., Ullman, J.D., Garey, M.R. &
    Graham, R.L. (1974). Worst-case performance bounds for simple
    one-dimensional packing algorithms. SIAM Journal on Computing,
    3(4), 299-325.
    https://doi.org/10.1137/0203025

    Dósa, G. (2007). The tight bound of first fit decreasing
    bin-packing algorithm is FFD(I) ≤ (11/9)OPT(I) + 6/9.
    Proceedings of ESCAPE, LNCS 4614, 1-11.
    https://doi.org/10.1007/978-3-540-74450-4_1
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


_inst = _load_mod("bpp_instance_ff", os.path.join(_parent_dir, "instance.py"))
BinPackingInstance = _inst.BinPackingInstance
BinPackingSolution = _inst.BinPackingSolution


def first_fit_decreasing(instance: BinPackingInstance) -> BinPackingSolution:
    """Pack items using First Fit Decreasing.

    Sort items by size (largest first), place each in the first bin
    with sufficient remaining capacity.

    Args:
        instance: A BinPackingInstance.

    Returns:
        BinPackingSolution with FFD packing.
    """
    n = instance.n
    order = sorted(range(n), key=lambda i: instance.sizes[i], reverse=True)

    bins: list[list[int]] = []
    remaining: list[float] = []

    for idx in order:
        size = instance.sizes[idx]
        placed = False
        for b in range(len(bins)):
            if remaining[b] >= size - 1e-10:
                bins[b].append(idx)
                remaining[b] -= size
                placed = True
                break
        if not placed:
            bins.append([idx])
            remaining.append(instance.capacity - size)

    return BinPackingSolution(bins=bins, num_bins=len(bins))


def best_fit_decreasing(instance: BinPackingInstance) -> BinPackingSolution:
    """Pack items using Best Fit Decreasing.

    Sort items by size (largest first), place each in the bin with
    the least remaining capacity that still fits.

    Args:
        instance: A BinPackingInstance.

    Returns:
        BinPackingSolution with BFD packing.
    """
    n = instance.n
    order = sorted(range(n), key=lambda i: instance.sizes[i], reverse=True)

    bins: list[list[int]] = []
    remaining: list[float] = []

    for idx in order:
        size = instance.sizes[idx]
        best_bin = -1
        best_rem = float("inf")

        for b in range(len(bins)):
            if remaining[b] >= size - 1e-10 and remaining[b] < best_rem:
                best_bin = b
                best_rem = remaining[b]

        if best_bin >= 0:
            bins[best_bin].append(idx)
            remaining[best_bin] -= size
        else:
            bins.append([idx])
            remaining.append(instance.capacity - size)

    return BinPackingSolution(bins=bins, num_bins=len(bins))


def first_fit(instance: BinPackingInstance) -> BinPackingSolution:
    """Pack items using First Fit (original order, no sorting).

    Args:
        instance: A BinPackingInstance.

    Returns:
        BinPackingSolution with FF packing.
    """
    n = instance.n
    bins: list[list[int]] = []
    remaining: list[float] = []

    for idx in range(n):
        size = instance.sizes[idx]
        placed = False
        for b in range(len(bins)):
            if remaining[b] >= size - 1e-10:
                bins[b].append(idx)
                remaining[b] -= size
                placed = True
                break
        if not placed:
            bins.append([idx])
            remaining.append(instance.capacity - size)

    return BinPackingSolution(bins=bins, num_bins=len(bins))


if __name__ == "__main__":
    from instance import easy_bpp_6, tight_bpp_8, uniform_bpp_10

    print("=== Bin Packing Heuristics ===\n")

    for name, inst_fn in [
        ("easy6", easy_bpp_6),
        ("tight8", tight_bpp_8),
        ("uniform10", uniform_bpp_10),
    ]:
        inst = inst_fn()
        ffd = first_fit_decreasing(inst)
        bfd = best_fit_decreasing(inst)
        ff = first_fit(inst)
        print(f"{name} (L1={inst.lower_bound_l1()}): "
              f"FF={ff.num_bins}, FFD={ffd.num_bins}, BFD={bfd.num_bins}")
