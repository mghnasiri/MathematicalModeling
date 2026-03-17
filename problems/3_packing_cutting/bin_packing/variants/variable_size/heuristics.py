"""
Constructive Heuristics for Variable-Size Bin Packing.

Problem: VS-BPP
Complexity: O(n^2 * K) for FFD variant

1. FFD with best bin type: sort items decreasing, for each item pick
   the smallest (cheapest) bin type that fits, using first-fit.
2. Cost-ratio greedy: prefer bin types with best capacity/cost ratio.

References:
    Friesen, D.K. & Langston, M.A. (1986). Variable sized bin packing.
    SIAM Journal on Computing, 15(1), 222-230.
    https://doi.org/10.1137/0215016
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


_inst = _load_mod("vsbpp_instance_h", os.path.join(_this_dir, "instance.py"))
VSBPPInstance = _inst.VSBPPInstance
VSBPPSolution = _inst.VSBPPSolution


def ffd_best_type(instance: VSBPPInstance) -> VSBPPSolution:
    """FFD with smallest sufficient bin type.

    Sort items decreasing. For each item, try to fit in existing open
    bins. If none fit, open a new bin of the cheapest type that fits.

    Args:
        instance: A VSBPPInstance.

    Returns:
        VSBPPSolution.
    """
    n = instance.n
    order = sorted(range(n), key=lambda i: instance.item_sizes[i], reverse=True)

    # Sort bin types by cost (prefer cheapest)
    type_order = sorted(range(instance.num_bin_types),
                        key=lambda t: instance.bin_costs[t])

    bins: list[tuple[int, list[int], float]] = []  # (type, items, remaining)

    for i in order:
        size = instance.item_sizes[i]
        placed = False

        # Try existing bins (best-fit: tightest remaining)
        best_bin = -1
        best_remaining = float("inf")
        for b_idx, (btype, items, remaining) in enumerate(bins):
            if remaining >= size - 1e-10 and remaining - size < best_remaining:
                best_remaining = remaining - size
                best_bin = b_idx

        if best_bin >= 0:
            btype, items, remaining = bins[best_bin]
            items.append(i)
            bins[best_bin] = (btype, items, remaining - size)
            placed = True

        if not placed:
            # Open new bin of cheapest sufficient type
            for t in type_order:
                if instance.bin_capacities[t] >= size - 1e-10:
                    bins.append((t, [i], instance.bin_capacities[t] - size))
                    placed = True
                    break

            if not placed:
                # Use largest bin type as fallback
                t = int(np.argmax(instance.bin_capacities))
                bins.append((t, [i], instance.bin_capacities[t] - size))

    result_bins = [(btype, items) for btype, items, _ in bins]
    total_cost = sum(instance.bin_costs[btype] for btype, _ in result_bins)
    return VSBPPSolution(bins=result_bins, total_cost=total_cost)


def cost_ratio_greedy(instance: VSBPPInstance) -> VSBPPSolution:
    """Prefer bin types with best capacity/cost ratio.

    Same as FFD but when opening new bins, prefer best ratio.

    Args:
        instance: A VSBPPInstance.

    Returns:
        VSBPPSolution.
    """
    n = instance.n
    order = sorted(range(n), key=lambda i: instance.item_sizes[i], reverse=True)

    # Sort by capacity/cost ratio (best value)
    ratio_order = sorted(
        range(instance.num_bin_types),
        key=lambda t: instance.bin_capacities[t] / max(instance.bin_costs[t], 1e-10),
        reverse=True,
    )

    bins: list[tuple[int, list[int], float]] = []

    for i in order:
        size = instance.item_sizes[i]
        placed = False

        best_bin = -1
        best_remaining = float("inf")
        for b_idx, (btype, items, remaining) in enumerate(bins):
            if remaining >= size - 1e-10 and remaining - size < best_remaining:
                best_remaining = remaining - size
                best_bin = b_idx

        if best_bin >= 0:
            btype, items, remaining = bins[best_bin]
            items.append(i)
            bins[best_bin] = (btype, items, remaining - size)
            placed = True

        if not placed:
            for t in ratio_order:
                if instance.bin_capacities[t] >= size - 1e-10:
                    bins.append((t, [i], instance.bin_capacities[t] - size))
                    placed = True
                    break

            if not placed:
                t = int(np.argmax(instance.bin_capacities))
                bins.append((t, [i], instance.bin_capacities[t] - size))

    result_bins = [(btype, items) for btype, items, _ in bins]
    total_cost = sum(instance.bin_costs[btype] for btype, _ in result_bins)
    return VSBPPSolution(bins=result_bins, total_cost=total_cost)


if __name__ == "__main__":
    inst = _inst.small_vsbpp_8()
    sol1 = ffd_best_type(inst)
    print(f"FFD best-type: cost={sol1.total_cost:.1f}, bins={len(sol1.bins)}")
    sol2 = cost_ratio_greedy(inst)
    print(f"Cost-ratio: cost={sol2.total_cost:.1f}, bins={len(sol2.bins)}")
