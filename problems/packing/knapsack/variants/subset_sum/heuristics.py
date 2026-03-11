"""
Subset Sum Problem — Heuristics.

Algorithms:
    - Greedy largest first: pick largest elements that fit.
    - Dynamic Programming: exact O(n * T) pseudo-polynomial.

References:
    Kellerer, H., Pferschy, U. & Pisinger, D. (2004). Knapsack Problems.
    Springer. https://doi.org/10.1007/978-3-540-24777-7
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


_inst = _load_mod("ssp_instance_h", os.path.join(_this_dir, "instance.py"))
SubsetSumInstance = _inst.SubsetSumInstance
SubsetSumSolution = _inst.SubsetSumSolution


def greedy_largest(instance: SubsetSumInstance) -> SubsetSumSolution:
    """Greedy: sort descending, add elements that fit under target.

    Args:
        instance: Subset Sum instance.

    Returns:
        SubsetSumSolution.
    """
    order = np.argsort(-instance.values)
    selected = []
    total = 0
    for i in order:
        v = int(instance.values[i])
        if total + v <= instance.target:
            selected.append(int(i))
            total += v
    return SubsetSumSolution(selected=selected, total=total)


def dynamic_programming(instance: SubsetSumInstance) -> SubsetSumSolution:
    """Exact DP for Subset Sum, O(n * T).

    Args:
        instance: Subset Sum instance.

    Returns:
        Optimal SubsetSumSolution.
    """
    n = instance.n
    T = instance.target
    values = instance.values

    # dp[t] = True if sum t is achievable
    dp = np.zeros(T + 1, dtype=bool)
    dp[0] = True
    # Track which items were used
    parent = [(-1, -1)] * (T + 1)  # (previous sum, item index)

    for i in range(n):
        v = int(values[i])
        # Iterate backwards to avoid using same item twice
        for t in range(T, v - 1, -1):
            if dp[t - v] and not dp[t]:
                dp[t] = True
                parent[t] = (t - v, i)

    # Find best achievable sum <= T
    best = 0
    for t in range(T, -1, -1):
        if dp[t]:
            best = t
            break

    # Backtrack to find selection
    selected = []
    t = best
    while t > 0 and parent[t][0] >= 0:
        prev_t, item = parent[t]
        selected.append(item)
        t = prev_t

    return SubsetSumSolution(selected=selected, total=best)


if __name__ == "__main__":
    from instance import small_ssp_6

    inst = small_ssp_6()
    gr = greedy_largest(inst)
    print(f"Greedy: {gr}")
    dp = dynamic_programming(inst)
    print(f"DP: {dp}")
