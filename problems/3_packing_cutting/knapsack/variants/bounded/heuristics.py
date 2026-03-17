"""
Heuristics for Bounded Knapsack.

Problem: BKP
Complexity: O(n log n) for greedy, O(n * W) for DP

1. Greedy value-density: sort by v/w, take as many as bound allows.
2. Dynamic Programming: exact O(n * W) pseudo-polynomial.

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


_inst = _load_mod("bkp_instance_h", os.path.join(_this_dir, "instance.py"))
BKPInstance = _inst.BKPInstance
BKPSolution = _inst.BKPSolution


def greedy_density(instance: BKPInstance) -> BKPSolution:
    """Greedy: sort by v/w, take max copies that fit."""
    n = instance.n
    density = instance.values / np.maximum(instance.weights, 1e-10)
    order = sorted(range(n), key=lambda i: density[i], reverse=True)

    remaining = instance.capacity
    quantities = [0] * n

    for i in order:
        max_copies = min(
            int(instance.bounds[i]),
            int(remaining / max(instance.weights[i], 1e-10)),
        )
        quantities[i] = max_copies
        remaining -= instance.weights[i] * max_copies

    value = instance.total_value(quantities)
    return BKPSolution(quantities=quantities, value=value)


def dynamic_programming(instance: BKPInstance) -> BKPSolution:
    """Exact DP for BKP. O(n * W * max_bound)."""
    n = instance.n
    W = int(instance.capacity)

    dp = np.zeros(W + 1)
    choice = [[0] * n for _ in range(W + 1)]

    for i in range(n):
        w_i = int(instance.weights[i])
        v_i = instance.values[i]
        b_i = int(instance.bounds[i])

        # Process in reverse to handle bounded copies
        new_dp = dp.copy()
        new_choice = [c[:] for c in choice]

        for w in range(W + 1):
            for k in range(1, b_i + 1):
                if w >= k * w_i:
                    val = dp[w - k * w_i] + k * v_i
                    if val > new_dp[w] + 1e-10:
                        new_dp[w] = val
                        new_choice[w] = choice[w - k * w_i][:]
                        new_choice[w][i] = k

        dp = new_dp
        choice = new_choice

    quantities = choice[W]
    value = instance.total_value(quantities)
    return BKPSolution(quantities=quantities, value=value)


if __name__ == "__main__":
    inst = _inst.small_bkp_5()
    sol1 = greedy_density(inst)
    print(f"Greedy: value={sol1.value:.0f}, qty={sol1.quantities}")
    sol2 = dynamic_programming(inst)
    print(f"DP: value={sol2.value:.0f}, qty={sol2.quantities}")
