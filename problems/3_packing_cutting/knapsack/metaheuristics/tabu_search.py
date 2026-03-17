"""
Tabu Search for 0-1 Knapsack Problem.

Problem: KP01 (0-1 Knapsack)

Representation: Binary vector x of length n, x[i] = 1 if item i selected.

Neighborhoods:
- Add: include an unselected item (if capacity allows)
- Drop: remove a selected item
- Swap: add one item and remove another

Uses short-term memory preventing recently toggled items from being
toggled again. Aspiration criterion overrides tabu when a move yields
a new global best.

Warm-started with greedy value-density heuristic.

Complexity: O(iterations * n) per run.

References:
    Glover, F. & Kochenberger, G.A. (1996). Critical event tabu
    search for multidimensional knapsack problems. In: Osman, I.H.
    & Kelly, J.P. (eds) Meta-Heuristics, Springer, 407-427.
    https://doi.org/10.1007/978-1-4613-1361-8_25

    Hanafi, S. & Fréville, A. (1998). An efficient tabu search
    approach for the 0-1 multidimensional knapsack problem. European
    Journal of Operational Research, 106(2-3), 659-675.
    https://doi.org/10.1016/S0377-2217(97)00296-8
"""

from __future__ import annotations

import os
import sys
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


_inst = _load_mod("kp_instance_ts", os.path.join(_parent_dir, "instance.py"))
KnapsackInstance = _inst.KnapsackInstance
KnapsackSolution = _inst.KnapsackSolution

_greedy = _load_mod(
    "kp_greedy_ts",
    os.path.join(_parent_dir, "heuristics", "greedy.py"),
)
greedy_value_density = _greedy.greedy_value_density


def tabu_search(
    instance: KnapsackInstance,
    max_iterations: int = 2000,
    tabu_tenure: int | None = None,
    time_limit: float | None = None,
    seed: int | None = None,
) -> KnapsackSolution:
    """Solve 0-1 Knapsack using Tabu Search.

    Args:
        instance: Knapsack instance.
        max_iterations: Maximum iterations.
        tabu_tenure: Iterations a move stays tabu. Default: sqrt(n).
        time_limit: Time limit in seconds.
        seed: Random seed for reproducibility.

    Returns:
        Best KnapsackSolution found.
    """
    rng = np.random.default_rng(seed)
    n = instance.n
    start_time = time.time()

    if tabu_tenure is None:
        tabu_tenure = max(3, int(n ** 0.5))

    # Initialize with greedy
    init_sol = greedy_value_density(instance)
    current = np.zeros(n, dtype=int)
    for i in init_sol.items:
        current[i] = 1

    current_value = instance.total_value(list(np.where(current)[0]))
    current_weight = instance.total_weight(list(np.where(current)[0]))

    best = current.copy()
    best_value = current_value

    # Tabu list: item -> iteration when tabu expires
    tabu_dict: dict[int, int] = {}

    for iteration in range(max_iterations):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        best_move_value = -float("inf")
        best_move_item = -1
        best_move_type = ""  # "add", "drop", or "swap"
        best_swap_drop = -1

        for i in range(n):
            is_tabu_i = (
                i in tabu_dict and tabu_dict[i] > iteration
            )

            if current[i] == 0:
                # Try adding item i
                new_weight = current_weight + instance.weights[i]
                if new_weight <= instance.capacity + 1e-10:
                    new_value = current_value + instance.values[i]

                    if is_tabu_i and new_value <= best_value:
                        continue

                    if new_value > best_move_value:
                        best_move_value = new_value
                        best_move_item = i
                        best_move_type = "add"
                        best_swap_drop = -1

                # Try swap: add i, drop j
                for j in range(n):
                    if current[j] == 0 or j == i:
                        continue
                    swap_weight = (
                        current_weight + instance.weights[i]
                        - instance.weights[j]
                    )
                    if swap_weight > instance.capacity + 1e-10:
                        continue
                    swap_value = (
                        current_value + instance.values[i]
                        - instance.values[j]
                    )

                    is_tabu_j = (
                        j in tabu_dict and tabu_dict[j] > iteration
                    )
                    if (is_tabu_i or is_tabu_j) and swap_value <= best_value:
                        continue

                    if swap_value > best_move_value:
                        best_move_value = swap_value
                        best_move_item = i
                        best_move_type = "swap"
                        best_swap_drop = j

            else:
                # Try dropping item i
                new_value = current_value - instance.values[i]

                if is_tabu_i and new_value <= best_value:
                    continue

                if new_value > best_move_value:
                    best_move_value = new_value
                    best_move_item = i
                    best_move_type = "drop"
                    best_swap_drop = -1

        if best_move_item < 0:
            tabu_dict.clear()
            continue

        # Apply move
        if best_move_type == "add":
            current[best_move_item] = 1
            current_value += instance.values[best_move_item]
            current_weight += instance.weights[best_move_item]
            tabu_dict[best_move_item] = iteration + tabu_tenure

        elif best_move_type == "drop":
            current[best_move_item] = 0
            current_value -= instance.values[best_move_item]
            current_weight -= instance.weights[best_move_item]
            tabu_dict[best_move_item] = iteration + tabu_tenure

        elif best_move_type == "swap":
            current[best_move_item] = 1
            current[best_swap_drop] = 0
            current_value += (
                instance.values[best_move_item]
                - instance.values[best_swap_drop]
            )
            current_weight += (
                instance.weights[best_move_item]
                - instance.weights[best_swap_drop]
            )
            tabu_dict[best_move_item] = iteration + tabu_tenure
            tabu_dict[best_swap_drop] = iteration + tabu_tenure

        # Update best
        if current_value > best_value and current_weight <= instance.capacity + 1e-10:
            best_value = current_value
            best = current.copy()

    items = list(np.where(best)[0])
    return KnapsackSolution(
        items=items,
        value=instance.total_value(items),
        weight=instance.total_weight(items),
    )


if __name__ == "__main__":
    from instance import small_knapsack_4, medium_knapsack_8

    print("=== Tabu Search for Knapsack ===\n")

    inst = small_knapsack_4()
    sol = tabu_search(inst, seed=42)
    print(f"small4: {sol}")

    inst2 = medium_knapsack_8()
    sol2 = tabu_search(inst2, seed=42)
    print(f"medium8: {sol2}")
