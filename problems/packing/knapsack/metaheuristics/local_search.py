"""
Local Search for 0-1 Knapsack.

Problem: 0-1 Knapsack Problem (KP01)

Iterative improvement using add/drop/swap neighborhoods.

Neighborhoods:
    - Add: include an unselected item if it fits
    - Drop: remove a selected item
    - Swap: replace a selected item with an unselected item

Warm-started with greedy value-density heuristic.
Best-improvement strategy: scan all moves, apply the best.

Complexity: O(iterations * n^2) per run.

References:
    Kellerer, H., Pferschy, U. & Pisinger, D. (2004). Knapsack Problems.
    Springer-Verlag, Berlin.
    https://doi.org/10.1007/978-3-540-24777-7

    Ghosh, J.B. (2003). Computational aspects of the maximum diversity
    problem. Operations Research Letters, 31(4), 316-320.
    https://doi.org/10.1016/S0167-6377(02)00213-0
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


_inst = _load_module("kp_instance_ls", os.path.join(_parent_dir, "instance.py"))
KnapsackInstance = _inst.KnapsackInstance
KnapsackSolution = _inst.KnapsackSolution


def local_search(
    instance: KnapsackInstance,
    neighborhood: str = "both",
    max_iterations: int = 1000,
    time_limit: float | None = None,
    seed: int | None = None,
) -> KnapsackSolution:
    """Solve 0-1 Knapsack using iterative local search.

    Args:
        instance: A KnapsackInstance.
        neighborhood: "add_drop", "swap", or "both".
        max_iterations: Maximum number of iterations.
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
        "kp_greedy_ls", os.path.join(_parent_dir, "heuristics", "greedy.py")
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

        best_delta = 0.0
        best_move = None

        if neighborhood in ("add_drop", "both"):
            # Try adding unselected items
            for i in range(n):
                if i in selected:
                    continue
                if current_weight + instance.weights[i] <= W + 1e-10:
                    delta = instance.values[i]
                    if delta > best_delta + 1e-10:
                        best_delta = delta
                        best_move = ("add", i)

        if neighborhood in ("swap", "both"):
            # Try swaps: remove one, add another
            in_items = list(selected)
            for rem in in_items:
                for add in range(n):
                    if add in selected:
                        continue
                    new_weight = current_weight - instance.weights[rem] + instance.weights[add]
                    if new_weight > W + 1e-10:
                        continue
                    delta = instance.values[add] - instance.values[rem]
                    if delta > best_delta + 1e-10:
                        best_delta = delta
                        best_move = ("swap", rem, add)

        if best_move is None:
            # No improving move found — try perturbation
            _perturb(instance, selected, current_value, current_weight, rng, W)
            current_value = instance.total_value(list(selected))
            current_weight = instance.total_weight(list(selected))

            if current_value > best_value + 1e-10:
                best_value = current_value
                best_selected = set(selected)
            continue

        # Apply best move
        if best_move[0] == "add":
            item = best_move[1]
            selected.add(item)
            current_value += instance.values[item]
            current_weight += instance.weights[item]
        elif best_move[0] == "swap":
            rem, add = best_move[1], best_move[2]
            selected.remove(rem)
            selected.add(add)
            current_value += instance.values[add] - instance.values[rem]
            current_weight += instance.weights[add] - instance.weights[rem]

        if current_value > best_value + 1e-10:
            best_value = current_value
            best_selected = set(selected)

    items = sorted(best_selected)
    return KnapsackSolution(
        items=items,
        value=instance.total_value(items),
        weight=instance.total_weight(items),
    )


def _perturb(
    instance: KnapsackInstance,
    selected: set[int],
    value: float,
    weight: float,
    rng: np.random.Generator,
    W: float,
) -> None:
    """Random perturbation: drop a random item, try to add others."""
    if not selected:
        return

    # Drop a random item
    to_drop = rng.choice(list(selected))
    selected.remove(to_drop)

    # Try to add a random unselected item
    current_weight = sum(instance.weights[i] for i in selected)
    unselected = [i for i in range(instance.n) if i not in selected]
    if unselected:
        rng.shuffle(unselected)
        for item in unselected:
            if current_weight + instance.weights[item] <= W + 1e-10:
                selected.add(item)
                current_weight += instance.weights[item]
                break


if __name__ == "__main__":
    from instance import small_knapsack_4, medium_knapsack_8

    print("=== Local Search for 0-1 Knapsack ===\n")

    for name, inst_fn in [
        ("small4", small_knapsack_4),
        ("medium8", medium_knapsack_8),
    ]:
        inst = inst_fn()
        sol = local_search(inst, seed=42)
        print(f"{name}: {sol}")
