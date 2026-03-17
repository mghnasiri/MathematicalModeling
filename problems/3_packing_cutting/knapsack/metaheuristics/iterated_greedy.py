"""
Iterated Greedy for the 0-1 Knapsack Problem.

Problem: 0-1 Knapsack (KP01)

Iterated Greedy repeatedly destroys and reconstructs solutions:
    1. Destroy: remove d random items from the current selection
    2. Repair: greedily add items by value-density ratio until capacity reached
    3. Accept: Boltzmann-based acceptance criterion

Warm-started with greedy value-density heuristic.

Complexity: O(iterations * n log n) per run.

References:
    Ruiz, R. & Stuetzle, T. (2007). A simple and effective iterated
    greedy algorithm for the permutation flowshop scheduling problem.
    European Journal of Operational Research, 177(3), 2033-2049.
    https://doi.org/10.1016/j.ejor.2005.12.009

    Martello, S. & Toth, P. (1990). Knapsack Problems: Algorithms and
    Computer Implementations. John Wiley & Sons, Chichester.
"""

from __future__ import annotations

import sys
import os
import math
import time
import importlib.util

import numpy as np

_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_mod(name, filepath):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_inst = _load_mod("kp_instance_ig", os.path.join(_parent_dir, "instance.py"))
KnapsackInstance = _inst.KnapsackInstance
KnapsackSolution = _inst.KnapsackSolution


def iterated_greedy(
    instance: KnapsackInstance,
    max_iterations: int = 5000,
    d: int | None = None,
    temperature_factor: float = 0.1,
    time_limit: float | None = None,
    seed: int | None = None,
) -> KnapsackSolution:
    """Solve 0-1 Knapsack using Iterated Greedy.

    Args:
        instance: A KnapsackInstance.
        max_iterations: Maximum number of iterations.
        d: Number of items to remove per iteration. Defaults to max(1, n//4).
        temperature_factor: Temperature as fraction of initial value.
        time_limit: Maximum wall-clock time in seconds.
        seed: Random seed for reproducibility.

    Returns:
        KnapsackSolution with the best selection found.
    """
    rng = np.random.default_rng(seed)
    n = instance.n
    start_time = time.time()

    if d is None:
        d = max(1, n // 4)

    # Precompute value-density ratios for greedy repair
    ratios = np.zeros(n)
    for i in range(n):
        ratios[i] = (instance.values[i] / instance.weights[i]
                     if instance.weights[i] > 0 else float("inf"))
    density_order = np.argsort(-ratios)

    # Warm-start with greedy
    _gr = _load_mod(
        "kp_greedy_ig",
        os.path.join(_parent_dir, "heuristics", "greedy.py"),
    )
    init_sol = _gr.greedy_value_density(instance)
    selected = set(init_sol.items)
    current_value = init_sol.value
    current_weight = init_sol.weight

    best_selected = set(selected)
    best_value = current_value

    temperature = temperature_factor * max(current_value, 1.0)

    for iteration in range(max_iterations):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        # Destroy: remove d random items
        items_list = list(selected)
        if len(items_list) == 0:
            # Try to add items from scratch
            new_selected = set()
            new_weight = 0.0
        else:
            d_actual = min(d, len(items_list))
            to_remove = rng.choice(items_list, size=d_actual, replace=False)
            new_selected = selected - set(to_remove)
            new_weight = sum(instance.weights[i] for i in new_selected)

        # Repair: greedily add items by value-density
        for i in density_order:
            if i in new_selected:
                continue
            if new_weight + instance.weights[i] <= instance.capacity + 1e-10:
                new_selected.add(i)
                new_weight += instance.weights[i]

        new_value = sum(instance.values[i] for i in new_selected)

        # Acceptance
        delta = new_value - current_value  # Positive = better for maximization
        if delta > 0 or (temperature > 0 and rng.random() < math.exp(delta / temperature)):
            selected = new_selected
            current_value = new_value
            current_weight = new_weight

            if current_value > best_value + 1e-10:
                best_value = current_value
                best_selected = set(selected)

    items = sorted(best_selected)
    return KnapsackSolution(
        items=items,
        value=instance.total_value(items),
        weight=instance.total_weight(items),
    )


if __name__ == "__main__":
    inst = KnapsackInstance.random(n=20, seed=42)
    print(f"Knapsack: {inst.n} items, capacity={inst.capacity}")

    _gr = _load_mod(
        "kp_greedy_ig_main",
        os.path.join(_parent_dir, "heuristics", "greedy.py"),
    )
    gr_sol = _gr.greedy_value_density(inst)
    print(f"Greedy: value={gr_sol.value:.1f}, weight={gr_sol.weight:.1f}")

    ig_sol = iterated_greedy(inst, seed=42)
    print(f"IG: value={ig_sol.value:.1f}, weight={ig_sol.weight:.1f}")
