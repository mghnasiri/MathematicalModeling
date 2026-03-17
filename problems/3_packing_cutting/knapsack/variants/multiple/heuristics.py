"""
Greedy Heuristics for Multiple Knapsack Problem (mKP).

Problem: mKP
Complexity: O(n log n + n * k)

1. Greedy value-density: sort items by v/w, assign to first fitting knapsack.
2. Greedy best-fit: sort items by v/w, assign to knapsack with tightest fit.

References:
    Martello, S. & Toth, P. (1990). Knapsack Problems: Algorithms and
    Computer Implementations. Wiley.
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


_inst = _load_mod("mkp_multi_instance_h", os.path.join(_this_dir, "instance.py"))
MultipleKnapsackInstance = _inst.MultipleKnapsackInstance
MultipleKnapsackSolution = _inst.MultipleKnapsackSolution


def greedy_value_density(
    instance: MultipleKnapsackInstance,
) -> MultipleKnapsackSolution:
    """Assign items by value-density to first fitting knapsack.

    Args:
        instance: A MultipleKnapsackInstance.

    Returns:
        MultipleKnapsackSolution.
    """
    n, k = instance.n, instance.k
    density = instance.values / np.maximum(instance.weights, 1e-10)
    order = sorted(range(n), key=lambda j: density[j], reverse=True)

    remaining = instance.capacities.copy()
    assignments = [-1] * n

    for j in order:
        for i in range(k):
            if remaining[i] >= instance.weights[j] - 1e-10:
                assignments[j] = i
                remaining[i] -= instance.weights[j]
                break

    value = sum(instance.values[j] for j in range(n) if assignments[j] >= 0)
    return MultipleKnapsackSolution(assignments=assignments, value=value)


def greedy_best_fit(
    instance: MultipleKnapsackInstance,
) -> MultipleKnapsackSolution:
    """Assign items by value-density to best-fitting knapsack.

    Best-fit: knapsack with smallest remaining capacity that still fits.

    Args:
        instance: A MultipleKnapsackInstance.

    Returns:
        MultipleKnapsackSolution.
    """
    n, k = instance.n, instance.k
    density = instance.values / np.maximum(instance.weights, 1e-10)
    order = sorted(range(n), key=lambda j: density[j], reverse=True)

    remaining = instance.capacities.copy()
    assignments = [-1] * n

    for j in order:
        best_knapsack = -1
        best_remaining = float("inf")
        for i in range(k):
            if remaining[i] >= instance.weights[j] - 1e-10:
                if remaining[i] - instance.weights[j] < best_remaining:
                    best_remaining = remaining[i] - instance.weights[j]
                    best_knapsack = i
        if best_knapsack >= 0:
            assignments[j] = best_knapsack
            remaining[best_knapsack] -= instance.weights[j]

    value = sum(instance.values[j] for j in range(n) if assignments[j] >= 0)
    return MultipleKnapsackSolution(assignments=assignments, value=value)


if __name__ == "__main__":
    inst = _inst.small_mkp_6_2()
    sol1 = greedy_value_density(inst)
    print(f"Value-density: value={sol1.value:.0f}, assignments={sol1.assignments}")
    sol2 = greedy_best_fit(inst)
    print(f"Best-fit: value={sol2.value:.0f}, assignments={sol2.assignments}")
