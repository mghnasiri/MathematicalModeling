"""
Variable Neighborhood Search for 0-1 Knapsack.

Problem: 0-1 Knapsack Problem (KP01)

VNS uses multiple neighborhood structures to escape local optima:
    N1: Flip a single bit (add or remove one item)
    N2: Swap — remove one item, add another
    N3: Double-flip — flip two bits simultaneously

Local search uses best-improvement single-flip.
Warm-started with greedy value-density heuristic.

Complexity: O(iterations * n^2) per run.

References:
    Mladenović, N. & Hansen, P. (1997). Variable neighborhood search.
    Computers & Operations Research, 24(11), 1097-1100.
    https://doi.org/10.1016/S0305-0548(97)00031-2

    Hansen, P. & Mladenović, N. (2001). Variable neighborhood search:
    Principles and applications. European Journal of Operational Research,
    130(3), 449-467.
    https://doi.org/10.1016/S0377-2217(00)00100-4
"""

from __future__ import annotations

import os
import time
import numpy as np

_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_module(name, filepath):
    import importlib.util
    import sys as _sys
    if name in _sys.modules:
        return _sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    _sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_inst = _load_module("kp_instance_vns", os.path.join(_parent_dir, "instance.py"))
KnapsackInstance = _inst.KnapsackInstance
KnapsackSolution = _inst.KnapsackSolution


def vns(
    instance: KnapsackInstance,
    max_iterations: int = 1000,
    k_max: int = 3,
    time_limit: float | None = None,
    seed: int | None = None,
) -> KnapsackSolution:
    """Solve 0-1 Knapsack using Variable Neighborhood Search.

    Args:
        instance: A KnapsackInstance.
        max_iterations: Maximum number of VNS iterations.
        k_max: Number of neighborhood structures (1-3).
        time_limit: Maximum wall-clock time in seconds.
        seed: Random seed for reproducibility.

    Returns:
        KnapsackSolution with the best selection found.
    """
    rng = np.random.default_rng(seed)
    n = instance.n
    W = instance.capacity
    start_time = time.time()

    # Warm-start with greedy
    _greedy_mod = _load_module(
        "kp_greedy_vns", os.path.join(_parent_dir, "heuristics", "greedy.py")
    )
    init_sol = _greedy_mod.greedy_value_density(instance)
    selected = set(init_sol.items)

    current_value = init_sol.value
    current_weight = init_sol.weight

    best_selected = set(selected)
    best_value = current_value

    for iteration in range(max_iterations):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        k = 1
        while k <= k_max:
            if time_limit is not None and time.time() - start_time >= time_limit:
                break

            # Shaking
            shaken, s_value, s_weight = _shake(
                instance, selected, current_value, current_weight, k, rng
            )

            # Local search (best-improvement single-flip)
            ls_sel, ls_val, ls_wt = _local_search(instance, shaken, s_value, s_weight)

            if ls_val > current_value + 1e-10:
                selected = ls_sel
                current_value = ls_val
                current_weight = ls_wt
                k = 1  # restart

                if current_value > best_value + 1e-10:
                    best_value = current_value
                    best_selected = set(selected)
            else:
                k += 1

    items = sorted(best_selected)
    return KnapsackSolution(
        items=items,
        value=instance.total_value(items),
        weight=instance.total_weight(items),
    )


def _shake(
    instance: KnapsackInstance,
    selected: set[int],
    value: float,
    weight: float,
    k: int,
    rng: np.random.Generator,
) -> tuple[set[int], float, float]:
    """Random perturbation in neighborhood k."""
    n = instance.n
    W = instance.capacity
    new_sel = set(selected)
    new_val = value
    new_wt = weight

    if k == 1:
        # Flip a random bit
        i = rng.integers(0, n)
        if i in new_sel:
            new_sel.remove(i)
            new_val -= instance.values[i]
            new_wt -= instance.weights[i]
        else:
            new_wt += instance.weights[i]
            if new_wt <= W + 1e-10:
                new_sel.add(i)
                new_val += instance.values[i]
            else:
                new_wt -= instance.weights[i]

    elif k == 2:
        # Swap: remove one selected, add one unselected
        in_items = list(new_sel)
        out_items = [i for i in range(n) if i not in new_sel]
        if in_items and out_items:
            rem = rng.choice(in_items)
            add = rng.choice(out_items)
            new_sel.remove(rem)
            new_val -= instance.values[rem]
            new_wt -= instance.weights[rem]
            new_wt += instance.weights[add]
            if new_wt <= W + 1e-10:
                new_sel.add(add)
                new_val += instance.values[add]
            else:
                new_wt -= instance.weights[add]

    elif k == 3:
        # Double flip: flip two random bits
        indices = rng.choice(n, size=min(2, n), replace=False)
        for i in indices:
            if i in new_sel:
                new_sel.remove(i)
                new_val -= instance.values[i]
                new_wt -= instance.weights[i]
            else:
                new_wt += instance.weights[i]
                if new_wt <= W + 1e-10:
                    new_sel.add(i)
                    new_val += instance.values[i]
                else:
                    new_wt -= instance.weights[i]

    return new_sel, new_val, new_wt


def _local_search(
    instance: KnapsackInstance,
    selected: set[int],
    value: float,
    weight: float,
) -> tuple[set[int], float, float]:
    """Best-improvement local search with single-flip neighborhood."""
    n = instance.n
    W = instance.capacity
    improved = True

    while improved:
        improved = False
        best_delta = 0.0
        best_item = -1
        best_action = None  # "add" or "remove"

        for i in range(n):
            if i in selected:
                # Try removing
                delta = -instance.values[i]
                # Removing never makes objective better alone,
                # but check if we can add a better item
            else:
                # Try adding
                if weight + instance.weights[i] <= W + 1e-10:
                    delta = instance.values[i]
                    if delta > best_delta + 1e-10:
                        best_delta = delta
                        best_item = i
                        best_action = "add"

        # Also try swap: remove one, add a better one
        in_items = list(selected)
        for rem in in_items:
            freed_weight = weight - instance.weights[rem]
            freed_value = value - instance.values[rem]
            for add in range(n):
                if add in selected:
                    continue
                if freed_weight + instance.weights[add] <= W + 1e-10:
                    delta = instance.values[add] - instance.values[rem]
                    if delta > best_delta + 1e-10:
                        best_delta = delta
                        best_item = (rem, add)
                        best_action = "swap"

        if best_action == "add":
            selected.add(best_item)
            value += instance.values[best_item]
            weight += instance.weights[best_item]
            improved = True
        elif best_action == "swap":
            rem, add = best_item
            selected.remove(rem)
            selected.add(add)
            value += instance.values[add] - instance.values[rem]
            weight += instance.weights[add] - instance.weights[rem]
            improved = True

    return selected, value, weight


if __name__ == "__main__":
    from instance import small_knapsack_4, medium_knapsack_8

    print("=== VNS for 0-1 Knapsack ===\n")

    for name, inst_fn in [
        ("small4", small_knapsack_4),
        ("medium8", medium_knapsack_8),
    ]:
        inst = inst_fn()
        sol = vns(inst, seed=42)
        print(f"{name}: {sol}")
