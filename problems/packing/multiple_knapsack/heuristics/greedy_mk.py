"""
Greedy Heuristic — Constructive heuristic for the Multiple Knapsack Problem.

Problem: Multiple Knapsack (MKP)
Complexity: O(n log n + n * m) — sorting + greedy assignment

Sort items by value-to-weight ratio (descending). For each item, assign
it to the knapsack with the most remaining capacity that can fit it.

References:
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

import numpy as np

_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_mod(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_inst = _load_mod("mkp_instance_gr", os.path.join(_parent_dir, "instance.py"))
MultipleKnapsackInstance = _inst.MultipleKnapsackInstance
MultipleKnapsackSolution = _inst.MultipleKnapsackSolution


def greedy_value_density(
    instance: MultipleKnapsackInstance,
) -> MultipleKnapsackSolution:
    """Solve MKP using greedy value-density assignment.

    Sort items by value/weight ratio (descending). Assign each item to
    the knapsack with the largest remaining capacity that can fit it.

    Args:
        instance: A MultipleKnapsackInstance.

    Returns:
        MultipleKnapsackSolution with greedy assignment.
    """
    n = instance.n
    m = instance.m

    # Sort items by value density descending
    density = instance.values / instance.weights
    order = sorted(range(n), key=lambda i: density[i], reverse=True)

    assignments: list[list[int]] = [[] for _ in range(m)]
    remaining = instance.capacities.copy().astype(float)

    for idx in order:
        w = instance.weights[idx]

        # Find knapsack with most remaining capacity that can fit this item
        best_ks = -1
        best_rem = -1.0
        for j in range(m):
            if remaining[j] >= w - 1e-10 and remaining[j] > best_rem:
                best_ks = j
                best_rem = remaining[j]

        if best_ks >= 0:
            assignments[best_ks].append(idx)
            remaining[best_ks] -= w

    total_value = sum(
        instance.values[i] for ks in assignments for i in ks
    )

    return MultipleKnapsackSolution(
        assignments=assignments, value=float(total_value)
    )


def greedy_best_fit(
    instance: MultipleKnapsackInstance,
) -> MultipleKnapsackSolution:
    """Solve MKP using greedy best-fit assignment.

    Sort items by value/weight ratio (descending). Assign each item to
    the knapsack with the smallest remaining capacity that still fits
    (tightest fit), to preserve large capacity for future items.

    Args:
        instance: A MultipleKnapsackInstance.

    Returns:
        MultipleKnapsackSolution with best-fit assignment.
    """
    n = instance.n
    m = instance.m

    density = instance.values / instance.weights
    order = sorted(range(n), key=lambda i: density[i], reverse=True)

    assignments: list[list[int]] = [[] for _ in range(m)]
    remaining = instance.capacities.copy().astype(float)

    for idx in order:
        w = instance.weights[idx]

        # Find knapsack with smallest remaining capacity that fits
        best_ks = -1
        best_rem = float("inf")
        for j in range(m):
            if remaining[j] >= w - 1e-10 and remaining[j] < best_rem:
                best_ks = j
                best_rem = remaining[j]

        if best_ks >= 0:
            assignments[best_ks].append(idx)
            remaining[best_ks] -= w

    total_value = sum(
        instance.values[i] for ks in assignments for i in ks
    )

    return MultipleKnapsackSolution(
        assignments=assignments, value=float(total_value)
    )


if __name__ == "__main__":
    _inst_mod = _load_mod("mkp_inst_main", os.path.join(_parent_dir, "instance.py"))
    small_mkp_6_2 = _inst_mod.small_mkp_6_2
    medium_mkp_8_3 = _inst_mod.medium_mkp_8_3
    validate_solution = _inst_mod.validate_solution

    print("=== Multiple Knapsack Greedy Heuristics ===\n")

    for name, inst_fn in [("small_6_2", small_mkp_6_2),
                           ("medium_8_3", medium_mkp_8_3)]:
        inst = inst_fn()
        for algo_name, algo in [("density", greedy_value_density),
                                 ("best_fit", greedy_best_fit)]:
            sol = algo(inst)
            valid, errors = validate_solution(inst, sol)
            print(f"{name} {algo_name}: value={sol.value:.0f}, valid={valid}")
