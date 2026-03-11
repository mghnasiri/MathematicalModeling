"""
Greedy Heuristics for the Multi-dimensional Knapsack Problem (MKP).

Problem: MKP (d-KP)
Complexity: O(n log n + n * d)

Pseudo-utility ratio greedy: sort items by v_i / (sum of normalized weights)
and greedily pack feasible items. The normalization accounts for multiple
resource constraints.

References:
    Pirkul, H. (1987). A heuristic solution procedure for the
    multiconstraint zero-one knapsack problem. Naval Research
    Logistics, 34(2), 161-172.
    https://doi.org/10.1002/1520-6750(198704)34:2<161::AID-NAV3220340203>3.0.CO;2-A

    Loulou, R. & Michaelides, E. (1979). New greedy-like heuristics for
    the multidimensional 0-1 knapsack problem. Operations Research,
    27(6), 1101-1114.
    https://doi.org/10.1287/opre.27.6.1101
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


_inst = _load_mod("mkp_instance", os.path.join(_this_dir, "instance.py"))
MKPInstance = _inst.MKPInstance
MKPSolution = _inst.MKPSolution


def greedy_pseudo_utility(instance: MKPInstance) -> MKPSolution:
    """Solve MKP using pseudo-utility ratio greedy.

    Pseudo-utility for item i = v_i / sum_j(w_ij / W_j).
    This normalizes weights by capacity so tighter dimensions weigh more.

    Args:
        instance: An MKPInstance.

    Returns:
        MKPSolution with greedy selection.
    """
    n, d = instance.n, instance.d

    # Compute pseudo-utility ratios
    ratios = np.zeros(n)
    for i in range(n):
        norm_weight = 0.0
        for j in range(d):
            if instance.capacities[j] > 0:
                norm_weight += instance.weights[j][i] / instance.capacities[j]
        ratios[i] = instance.values[i] / max(norm_weight, 1e-10)

    order = np.argsort(-ratios)

    selected = []
    remaining_cap = instance.capacities.copy()

    for i in order:
        feasible = True
        for j in range(d):
            if instance.weights[j][i] > remaining_cap[j] + 1e-10:
                feasible = False
                break
        if feasible:
            selected.append(int(i))
            for j in range(d):
                remaining_cap[j] -= instance.weights[j][i]

    selected.sort()
    return MKPSolution(
        items=selected,
        value=instance.total_value(selected),
        weights=instance.total_weights(selected),
    )


def greedy_max_value(instance: MKPInstance) -> MKPSolution:
    """Sort by value descending, pack feasible items."""
    order = np.argsort(-instance.values)

    selected = []
    remaining_cap = instance.capacities.copy()

    for i in order:
        feasible = True
        for j in range(instance.d):
            if instance.weights[j][i] > remaining_cap[j] + 1e-10:
                feasible = False
                break
        if feasible:
            selected.append(int(i))
            for j in range(instance.d):
                remaining_cap[j] -= instance.weights[j][i]

    selected.sort()
    return MKPSolution(
        items=selected,
        value=instance.total_value(selected),
        weights=instance.total_weights(selected),
    )


if __name__ == "__main__":
    inst = _inst.small_mkp_5_2()
    print(f"MKP: {inst.n} items, {inst.d} dims")

    sol = greedy_pseudo_utility(inst)
    print(f"Pseudo-utility greedy: value={sol.value}, items={sol.items}")

    sol2 = greedy_max_value(inst)
    print(f"Max-value greedy: value={sol2.value}, items={sol2.items}")
