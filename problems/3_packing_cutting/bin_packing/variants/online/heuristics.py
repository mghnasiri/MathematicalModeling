"""
Online Bin Packing — Online Heuristics.

All algorithms process items in arrival order (no sorting).

Algorithms:
    - Next Fit (NF): O(n), pack in current bin or open new.
    - First Fit (FF): O(n^2), first bin that fits.
    - Best Fit (BF): O(n^2), tightest fitting bin.

References:
    Johnson, D.S. (1973). Near-optimal bin packing algorithms. PhD thesis.
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


_inst = _load_mod("onlinebp_instance_h", os.path.join(_this_dir, "instance.py"))
OnlineBPInstance = _inst.OnlineBPInstance
OnlineBPSolution = _inst.OnlineBPSolution


def next_fit(instance: OnlineBPInstance) -> OnlineBPSolution:
    """Next Fit: use current bin; open new if item doesn't fit. O(n)."""
    assignments = []
    current_bin = 0
    remaining = instance.capacity

    for size in instance.sizes:
        if size > remaining + 1e-10:
            current_bin += 1
            remaining = instance.capacity
        assignments.append(current_bin)
        remaining -= size

    return OnlineBPSolution(assignments=assignments, num_bins=current_bin + 1)


def first_fit(instance: OnlineBPInstance) -> OnlineBPSolution:
    """First Fit: place in first bin with room. O(n^2)."""
    bin_remaining: list[float] = []
    assignments = []

    for size in instance.sizes:
        placed = False
        for b in range(len(bin_remaining)):
            if bin_remaining[b] >= size - 1e-10:
                assignments.append(b)
                bin_remaining[b] -= size
                placed = True
                break
        if not placed:
            assignments.append(len(bin_remaining))
            bin_remaining.append(instance.capacity - size)

    return OnlineBPSolution(assignments=assignments, num_bins=len(bin_remaining))


def best_fit(instance: OnlineBPInstance) -> OnlineBPSolution:
    """Best Fit: place in tightest fitting bin. O(n^2)."""
    bin_remaining: list[float] = []
    assignments = []

    for size in instance.sizes:
        best_b = -1
        best_space = float("inf")
        for b in range(len(bin_remaining)):
            if bin_remaining[b] >= size - 1e-10:
                space = bin_remaining[b] - size
                if space < best_space:
                    best_space = space
                    best_b = b
        if best_b >= 0:
            assignments.append(best_b)
            bin_remaining[best_b] -= size
        else:
            assignments.append(len(bin_remaining))
            bin_remaining.append(instance.capacity - size)

    return OnlineBPSolution(assignments=assignments, num_bins=len(bin_remaining))


if __name__ == "__main__":
    from instance import small_online_8

    inst = small_online_8()
    for name, algo in [("NF", next_fit), ("FF", first_fit), ("BF", best_fit)]:
        sol = algo(inst)
        print(f"{name}: {sol}")
