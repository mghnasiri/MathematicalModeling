"""
Dynamic Programming — Exact solver for the 0-1 Knapsack Problem.

Problem: 0-1 Knapsack (KP01)
Complexity: O(n * W) time, O(n * W) space (pseudo-polynomial)
Practical limit: n * W <= ~10^8

Standard bottom-up DP with backtracking to recover the optimal item set.
dp[i][w] = maximum value achievable using items 0..i-1 with capacity w.

References:
    Bellman, R. (1957). Dynamic Programming. Princeton University Press.

    Martello, S. & Toth, P. (1990). Knapsack Problems: Algorithms and
    Computer Implementations. John Wiley & Sons, Chichester.

    Kellerer, H., Pferschy, U. & Pisinger, D. (2004). Knapsack Problems.
    Springer-Verlag, Berlin.
    https://doi.org/10.1007/978-3-540-24777-7
"""

from __future__ import annotations

import os
import importlib.util
import sys

_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_mod(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_inst = _load_mod("kp_instance_dp", os.path.join(_parent_dir, "instance.py"))
KnapsackInstance = _inst.KnapsackInstance
KnapsackSolution = _inst.KnapsackSolution


def dynamic_programming(instance: KnapsackInstance) -> KnapsackSolution:
    """Solve 0-1 Knapsack exactly using dynamic programming.

    Uses bottom-up tabulation with integer capacity discretization.
    Works correctly for integer weights; for fractional weights,
    values are approximate within rounding tolerance.

    Args:
        instance: A KnapsackInstance with integer weights.

    Returns:
        KnapsackSolution with optimal item selection.
    """
    n = instance.n
    W = int(instance.capacity)
    weights = [int(round(w)) for w in instance.weights]
    values = instance.values

    # DP table: dp[w] = max value with capacity w
    dp = [0.0] * (W + 1)
    # Track which items were selected
    keep = [[False] * (W + 1) for _ in range(n)]

    for i in range(n):
        # Iterate in reverse to avoid using item i more than once
        for w in range(W, weights[i] - 1, -1):
            if dp[w - weights[i]] + values[i] > dp[w]:
                dp[w] = dp[w - weights[i]] + values[i]
                keep[i][w] = True

    # Backtrack to find selected items
    items = []
    w = W
    for i in range(n - 1, -1, -1):
        if keep[i][w]:
            items.append(i)
            w -= weights[i]

    items.reverse()
    total_value = instance.total_value(items)
    total_weight = instance.total_weight(items)

    return KnapsackSolution(
        items=items,
        value=total_value,
        weight=total_weight,
    )


if __name__ == "__main__":
    from instance import small_knapsack_4, medium_knapsack_8

    print("=== Dynamic Programming for 0-1 Knapsack ===\n")

    for name, inst_fn in [
        ("small4", small_knapsack_4),
        ("medium8", medium_knapsack_8),
    ]:
        inst = inst_fn()
        sol = dynamic_programming(inst)
        print(f"{name}: {sol}")
