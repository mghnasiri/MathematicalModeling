"""
Branch and Bound — Exact solver for the 0-1 Knapsack Problem.

Problem: 0-1 Knapsack (KP01)
Complexity: O(2^n) worst case, fast in practice with LP relaxation bound.
Practical limit: n <= ~40 with good bounds.

Uses depth-first search with the LP relaxation (fractional knapsack)
upper bound. Items are sorted by value-to-weight ratio for the bound
computation. Warm-started with the greedy heuristic.

References:
    Horowitz, E. & Sahni, S. (1974). Computing partitions with
    applications to the knapsack problem. Journal of the ACM,
    21(2), 277-292.
    https://doi.org/10.1145/321812.321823

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


_inst = _load_mod("kp_instance_bb", os.path.join(_parent_dir, "instance.py"))
KnapsackInstance = _inst.KnapsackInstance
KnapsackSolution = _inst.KnapsackSolution


def _lp_bound(
    values: list[float],
    weights: list[float],
    capacity: float,
    items_sorted: list[int],
    start: int,
    current_value: float,
    current_weight: float,
) -> float:
    """Compute LP relaxation upper bound for remaining items.

    Args:
        values: Item values.
        weights: Item weights.
        capacity: Total knapsack capacity.
        items_sorted: Items sorted by value/weight ratio (descending).
        start: Index in items_sorted to start from.
        current_value: Value accumulated so far.
        current_weight: Weight accumulated so far.

    Returns:
        Upper bound on achievable value.
    """
    remaining = capacity - current_weight
    bound = current_value

    for idx in range(start, len(items_sorted)):
        i = items_sorted[idx]
        if weights[i] <= remaining:
            bound += values[i]
            remaining -= weights[i]
        else:
            bound += values[i] * (remaining / weights[i])
            break

    return bound


def branch_and_bound(instance: KnapsackInstance) -> KnapsackSolution:
    """Solve 0-1 Knapsack exactly using branch and bound.

    Args:
        instance: A KnapsackInstance.

    Returns:
        KnapsackSolution with optimal item selection.
    """
    n = instance.n
    weights = list(instance.weights)
    values = list(instance.values)
    capacity = instance.capacity

    # Sort items by value/weight ratio (descending)
    ratios = [(values[i] / weights[i] if weights[i] > 0 else float("inf"), i)
              for i in range(n)]
    ratios.sort(reverse=True)
    sorted_items = [i for _, i in ratios]

    # Greedy warm-start
    best_value = 0.0
    best_items: list[int] = []
    w_sum = 0.0
    greedy_items: list[int] = []
    for i in sorted_items:
        if w_sum + weights[i] <= capacity:
            greedy_items.append(i)
            w_sum += weights[i]
    if greedy_items:
        best_value = sum(values[i] for i in greedy_items)
        best_items = greedy_items[:]

    # DFS
    stack: list[tuple[int, float, float, list[int]]] = [
        (0, 0.0, 0.0, [])
    ]

    while stack:
        level, current_value, current_weight, selected = stack.pop()

        if level >= n:
            if current_value > best_value:
                best_value = current_value
                best_items = selected[:]
            continue

        item = sorted_items[level]

        # Branch: exclude item
        bound_excl = _lp_bound(
            values, weights, capacity,
            sorted_items, level + 1,
            current_value, current_weight,
        )
        if bound_excl > best_value + 1e-10:
            stack.append((level + 1, current_value, current_weight, selected))

        # Branch: include item (if feasible)
        if current_weight + weights[item] <= capacity + 1e-10:
            new_value = current_value + values[item]
            new_weight = current_weight + weights[item]

            bound_incl = _lp_bound(
                values, weights, capacity,
                sorted_items, level + 1,
                new_value, new_weight,
            )
            if bound_incl > best_value + 1e-10:
                stack.append((
                    level + 1, new_value, new_weight,
                    selected + [item],
                ))

    best_items.sort()
    return KnapsackSolution(
        items=best_items,
        value=instance.total_value(best_items),
        weight=instance.total_weight(best_items),
    )


if __name__ == "__main__":
    from instance import small_knapsack_4, medium_knapsack_8

    print("=== Branch and Bound for 0-1 Knapsack ===\n")

    for name, inst_fn in [
        ("small4", small_knapsack_4),
        ("medium8", medium_knapsack_8),
    ]:
        inst = inst_fn()
        sol = branch_and_bound(inst)
        print(f"{name}: {sol}")
