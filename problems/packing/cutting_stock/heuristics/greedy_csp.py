"""
Greedy Heuristics for the 1D Cutting Stock Problem.

Problem: 1D Cutting Stock (CSP)
Complexity: O(m * n_rolls * m) per roll

Two greedy approaches:
- FFD-based: Expand demands into individual items, apply FFD bin packing,
  then aggregate back into patterns.
- Greedy pattern generation: For each roll, greedily fill with the largest
  items that still have remaining demand.

References:
    Gilmore, P.C. & Gomory, R.E. (1961). A linear programming approach
    to the cutting-stock problem. Operations Research, 9(6), 849-859.
    https://doi.org/10.1287/opre.9.6.849

    Wäscher, G., Haußner, H. & Schumann, H. (2007). An improved
    typology of cutting and packing problems. European Journal of
    Operational Research, 183(3), 1109-1130.
    https://doi.org/10.1016/j.ejor.2005.12.047
"""

from __future__ import annotations

import os
import sys

import importlib.util

import numpy as np

_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_mod(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_inst = _load_mod("csp_instance_gr", os.path.join(_parent_dir, "instance.py"))
CuttingStockInstance = _inst.CuttingStockInstance
CuttingStockSolution = _inst.CuttingStockSolution


def greedy_largest_first(
    instance: CuttingStockInstance,
) -> CuttingStockSolution:
    """Solve CSP with greedy pattern generation.

    For each roll, pack items largest-first until no more fit.
    Continue until all demands are met.

    Args:
        instance: A CuttingStockInstance.

    Returns:
        CuttingStockSolution with greedy patterns.
    """
    m = instance.m
    remaining = instance.demands.copy()
    # Sort item types by length (descending) for greedy filling
    order = sorted(range(m), key=lambda i: instance.lengths[i], reverse=True)

    patterns: list[tuple[np.ndarray, int]] = []

    while np.any(remaining > 0):
        pattern = np.zeros(m, dtype=int)
        space = instance.stock_length

        for i in order:
            if remaining[i] > 0 and instance.lengths[i] <= space + 1e-10:
                count = min(
                    remaining[i],
                    int(space / instance.lengths[i])
                )
                if count > 0:
                    pattern[i] = count
                    space -= count * instance.lengths[i]

        if np.sum(pattern) == 0:
            # Shouldn't happen with valid instances
            break

        # Apply pattern once
        patterns.append((pattern, 1))
        remaining -= pattern

    num_rolls = sum(freq for _, freq in patterns)
    return CuttingStockSolution(patterns=patterns, num_rolls=num_rolls)


def ffd_based(instance: CuttingStockInstance) -> CuttingStockSolution:
    """Solve CSP by expanding to individual items and applying FFD.

    Expand all demands into individual items, sort by length (decreasing),
    and apply First Fit Decreasing. Then aggregate back into patterns.

    Args:
        instance: A CuttingStockInstance.

    Returns:
        CuttingStockSolution with FFD-derived patterns.
    """
    m = instance.m

    # Expand items
    items: list[int] = []  # type index for each individual item
    for i in range(m):
        items.extend([i] * instance.demands[i])

    # Sort by length (descending)
    items.sort(key=lambda i: instance.lengths[i], reverse=True)

    # FFD packing
    bins: list[np.ndarray] = []
    remaining: list[float] = []

    for item_type in items:
        size = instance.lengths[item_type]
        placed = False
        for b in range(len(bins)):
            if remaining[b] >= size - 1e-10:
                bins[b][item_type] += 1
                remaining[b] -= size
                placed = True
                break
        if not placed:
            new_pattern = np.zeros(m, dtype=int)
            new_pattern[item_type] = 1
            bins.append(new_pattern)
            remaining.append(instance.stock_length - size)

    # Aggregate identical patterns
    pattern_dict: dict[tuple, int] = {}
    for pattern in bins:
        key = tuple(pattern)
        pattern_dict[key] = pattern_dict.get(key, 0) + 1

    patterns = [
        (np.array(key, dtype=int), freq)
        for key, freq in pattern_dict.items()
    ]

    num_rolls = sum(freq for _, freq in patterns)
    return CuttingStockSolution(patterns=patterns, num_rolls=num_rolls)


if __name__ == "__main__":
    from instance import simple_csp_3, classic_csp_4

    print("=== Greedy Heuristics for Cutting Stock ===\n")

    for name, inst_fn in [
        ("simple3", simple_csp_3),
        ("classic4", classic_csp_4),
    ]:
        inst = inst_fn()
        gl = greedy_largest_first(inst)
        ffd = ffd_based(inst)
        print(f"{name} (LB={inst.lower_bound()}): "
              f"greedy={gl.num_rolls}, ffd={ffd.num_rolls}")
