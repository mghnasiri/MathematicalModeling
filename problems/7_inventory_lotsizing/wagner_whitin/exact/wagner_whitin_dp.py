"""
Wagner-Whitin Dynamic Programming — Exact solver for uncapacitated lot sizing.

Problem: Uncapacitated Lot Sizing (ULS)
Complexity: O(T^2) time, O(T) space.

Key insight (Wagner-Whitin property): In an optimal solution, either the
order quantity is zero or the entering inventory is zero in each period.
Orders cover contiguous blocks of demand (Zero Inventory Ordering).

References:
    Wagner, H.M. & Whitin, T.M. (1958). Dynamic version of the economic
    lot size model. Management Science, 5(1), 89-96.
    https://doi.org/10.1287/mnsc.5.1.89
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


_inst = _load_mod("ww_instance_dp", os.path.join(_parent_dir, "instance.py"))
WagnerWhitinInstance = _inst.WagnerWhitinInstance
WagnerWhitinSolution = _inst.WagnerWhitinSolution


def wagner_whitin_dp(instance: WagnerWhitinInstance) -> WagnerWhitinSolution:
    """Solve uncapacitated lot sizing via Wagner-Whitin DP.

    Uses the ZIO property: orders occur only when inventory is zero.
    f(t) = min cost to satisfy demands in periods 0..t-1.

    Args:
        instance: WagnerWhitinInstance to solve.

    Returns:
        WagnerWhitinSolution with optimal order quantities and total cost.
    """
    T = instance.T
    d = instance.demands
    K = instance.ordering_costs
    h = instance.holding_costs

    # f[t] = min cost to satisfy demands in periods 0..t-1
    f = np.full(T + 1, float("inf"))
    f[0] = 0.0
    predecessor = np.zeros(T + 1, dtype=int)

    for t in range(1, T + 1):
        # Try ordering in period j (1-indexed) to cover demands j..t
        for j in range(1, t + 1):
            cost_j = f[j - 1] + K[j - 1]
            # Holding cost: demand d[k-1] (period k) ordered in period j,
            # held through periods j-1, j, ..., k-2 (0-indexed)
            holding_cost = 0.0
            for k in range(j + 1, t + 1):
                # d[k-1] is held for (k - j) periods: from period j to period k
                # holding cost = d[k-1] * sum of h[p] for p = j-1..k-2
                for p in range(j - 1, k - 1):
                    holding_cost += h[p] * d[k - 1]

            total = cost_j + holding_cost
            if total < f[t]:
                f[t] = total
                predecessor[t] = j

    # Backtrack to find order periods
    order_quantities = np.zeros(T, dtype=float)
    t = T
    order_periods = []
    while t > 0:
        j = predecessor[t]
        order_periods.append(j - 1)  # convert to 0-indexed
        order_quantities[j - 1] = float(np.sum(d[j - 1:t]))
        t = j - 1

    order_periods.sort()

    return WagnerWhitinSolution(
        order_quantities=order_quantities,
        total_cost=float(f[T]),
        order_periods=order_periods,
    )


if __name__ == "__main__":
    inst = _inst.textbook_4()
    sol = wagner_whitin_dp(inst)
    print(f"Wagner-Whitin DP on {inst.name}:")
    print(f"  {sol}")
    print(f"  Order quantities: {sol.order_quantities}")
    print(f"  Verified cost: {inst.compute_cost(sol.order_quantities):.2f}")
