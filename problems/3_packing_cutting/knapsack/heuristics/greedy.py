"""
Greedy Heuristic — Constructive heuristic for the 0-1 Knapsack Problem.

Problem: 0-1 Knapsack (KP01)
Complexity: O(n log n) — dominated by sorting

Sort items by value-to-weight ratio (descending), then greedily pack
items that fit. Simple and fast but not optimal. For the fractional
knapsack, this is optimal; for 0-1, it provides a lower bound.

Approximation: The greedy solution is at least max(v_max, v_greedy),
which is a 1/2-approximation for the 0-1 knapsack.

References:
    Dantzig, G.B. (1957). Discrete-variable extremum problems.
    Operations Research, 5(2), 266-288.
    https://doi.org/10.1287/opre.5.2.266

    Martello, S. & Toth, P. (1990). Knapsack Problems: Algorithms and
    Computer Implementations. John Wiley & Sons, Chichester.
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


_inst = _load_mod("kp_instance_gr", os.path.join(_parent_dir, "instance.py"))
KnapsackInstance = _inst.KnapsackInstance
KnapsackSolution = _inst.KnapsackSolution


def greedy_value_density(instance: KnapsackInstance) -> KnapsackSolution:
    """Solve 0-1 Knapsack using greedy value-density heuristic.

    Sort items by value/weight ratio (descending), greedily pack
    items that fit.

    Args:
        instance: A KnapsackInstance.

    Returns:
        KnapsackSolution with greedy selection.
    """
    n = instance.n
    ratios = [
        (instance.values[i] / instance.weights[i] if instance.weights[i] > 0
         else float("inf"), i)
        for i in range(n)
    ]
    ratios.sort(reverse=True)

    items = []
    remaining = instance.capacity

    for _, i in ratios:
        if instance.weights[i] <= remaining + 1e-10:
            items.append(i)
            remaining -= instance.weights[i]

    items.sort()
    return KnapsackSolution(
        items=items,
        value=instance.total_value(items),
        weight=instance.total_weight(items),
    )


def greedy_max_value(instance: KnapsackInstance) -> KnapsackSolution:
    """Greedy by highest value first (ignoring efficiency).

    Args:
        instance: A KnapsackInstance.

    Returns:
        KnapsackSolution with greedy selection.
    """
    n = instance.n
    order = sorted(range(n), key=lambda i: instance.values[i], reverse=True)

    items = []
    remaining = instance.capacity

    for i in order:
        if instance.weights[i] <= remaining + 1e-10:
            items.append(i)
            remaining -= instance.weights[i]

    items.sort()
    return KnapsackSolution(
        items=items,
        value=instance.total_value(items),
        weight=instance.total_weight(items),
    )


def greedy_combined(instance: KnapsackInstance) -> KnapsackSolution:
    """Return the better of value-density and max-value greedy solutions.

    This combined strategy provides a 1/2-approximation guarantee
    for the 0-1 Knapsack problem.

    Args:
        instance: A KnapsackInstance.

    Returns:
        KnapsackSolution — best of both greedy strategies.
    """
    sol1 = greedy_value_density(instance)
    sol2 = greedy_max_value(instance)
    return sol1 if sol1.value >= sol2.value else sol2


if __name__ == "__main__":
    from instance import small_knapsack_4, medium_knapsack_8

    print("=== Greedy Heuristics for 0-1 Knapsack ===\n")

    for name, inst_fn in [
        ("small4", small_knapsack_4),
        ("medium8", medium_knapsack_8),
    ]:
        inst = inst_fn()
        sol_vd = greedy_value_density(inst)
        sol_mv = greedy_max_value(inst)
        sol_c = greedy_combined(inst)
        print(f"{name}: density={sol_vd.value:.0f}, maxval={sol_mv.value:.0f}, "
              f"combined={sol_c.value:.0f}")
