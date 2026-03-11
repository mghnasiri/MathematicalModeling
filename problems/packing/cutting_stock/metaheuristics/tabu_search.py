"""
Tabu Search for 1D Cutting Stock Problem (CSP).

Problem: CSP1D (1D Cutting Stock)

Representation: A flat list of individual items (expanded from demands).
Decoded into rolls using First Fit. Neighborhoods modify the permutation
to change how items are packed.

Neighborhoods:
- Swap: exchange two items of different types in the permutation
- Insert: remove an item and re-insert at a different position

Uses short-term memory preventing recently moved items from being moved
again. Aspiration criterion overrides tabu for new global best.

Warm-started with FFD ordering (descending size).

Complexity: O(iterations * N) per iteration, where N = total items.

References:
    Loh, K.H., Golden, B. & Wasil, E. (2008). Solving the
    one-dimensional bin packing problem with a weight annealing
    heuristic. Computers & Operations Research, 35(7), 2283-2291.
    https://doi.org/10.1016/j.cor.2006.10.021

    Falkenauer, E. (1996). A hybrid grouping genetic algorithm for bin
    packing. Journal of Heuristics, 2(1), 5-30.
    https://doi.org/10.1007/BF00226291
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


_inst = _load_mod("csp_instance_ts", os.path.join(_parent_dir, "instance.py"))
CuttingStockInstance = _inst.CuttingStockInstance
CuttingStockSolution = _inst.CuttingStockSolution
validate_solution = _inst.validate_solution


def _decode_ff(
    lengths: np.ndarray,
    stock_length: float,
    perm: list[int],
) -> list[list[int]]:
    """Decode a permutation into rolls using First Fit."""
    rolls: list[list[int]] = []
    remaining: list[float] = []

    for item_type in perm:
        size = lengths[item_type]
        placed = False
        for r in range(len(rolls)):
            if remaining[r] >= size - 1e-10:
                rolls[r].append(item_type)
                remaining[r] -= size
                placed = True
                break
        if not placed:
            rolls.append([item_type])
            remaining.append(stock_length - size)

    return rolls


def _rolls_to_solution(
    instance: CuttingStockInstance,
    rolls: list[list[int]],
) -> CuttingStockSolution:
    """Convert decoded rolls to a CuttingStockSolution."""
    pattern_dict: dict[tuple, int] = {}
    for roll in rolls:
        counts = np.zeros(instance.m, dtype=int)
        for item_type in roll:
            counts[item_type] += 1
        key = tuple(counts)
        pattern_dict[key] = pattern_dict.get(key, 0) + 1

    patterns = [
        (np.array(key, dtype=int), freq)
        for key, freq in pattern_dict.items()
    ]
    num_rolls = sum(freq for _, freq in patterns)
    return CuttingStockSolution(patterns=patterns, num_rolls=num_rolls)


def tabu_search(
    instance: CuttingStockInstance,
    max_iterations: int = 5000,
    tabu_tenure: int | None = None,
    time_limit: float | None = None,
    seed: int | None = None,
) -> CuttingStockSolution:
    """Solve 1D Cutting Stock using Tabu Search.

    Args:
        instance: Cutting stock instance.
        max_iterations: Maximum iterations.
        tabu_tenure: Iterations a move stays tabu. Default: sqrt(N).
        time_limit: Time limit in seconds.
        seed: Random seed for reproducibility.

    Returns:
        Best CuttingStockSolution found.
    """
    rng = np.random.default_rng(seed)
    start_time = time.time()

    # Expand demands into individual items
    items: list[int] = []
    for i in range(instance.m):
        items.extend([i] * instance.demands[i])

    total_items = len(items)
    if total_items == 0:
        return CuttingStockSolution(patterns=[], num_rolls=0)

    if tabu_tenure is None:
        tabu_tenure = max(3, int(total_items ** 0.5))

    # Initialize with FFD ordering
    perm = sorted(items, key=lambda i: -instance.lengths[i])
    rolls = _decode_ff(instance.lengths, instance.stock_length, perm)
    current_num_rolls = len(rolls)

    best_perm = list(perm)
    best_num_rolls = current_num_rolls

    # Tabu list: (position, item_type) -> iteration when tabu expires
    tabu_dict: dict[tuple[int, int], int] = {}

    for iteration in range(max_iterations):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        if total_items < 2:
            break

        best_delta = float("inf")
        best_move = None

        # Sample candidate moves
        n_candidates = min(total_items * 3, total_items * total_items)
        for _ in range(n_candidates):
            if rng.random() < 0.5 or total_items < 3:
                # Swap move
                i, j = rng.choice(total_items, size=2, replace=False)
                if perm[i] == perm[j]:
                    continue

                is_tabu = (
                    ((i, perm[j]) in tabu_dict
                     and tabu_dict[(i, perm[j])] > iteration)
                    or ((j, perm[i]) in tabu_dict
                        and tabu_dict[(j, perm[i])] > iteration)
                )

                new_perm = list(perm)
                new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
                new_rolls = _decode_ff(
                    instance.lengths, instance.stock_length, new_perm
                )
                delta = len(new_rolls) - current_num_rolls

                if is_tabu and current_num_rolls + delta >= best_num_rolls:
                    continue

                if delta < best_delta:
                    best_delta = delta
                    best_move = ("swap", i, j, new_perm, new_rolls)
            else:
                # Insert move
                i = rng.integers(0, total_items)
                j = rng.integers(0, total_items)
                if i == j:
                    continue

                item = perm[i]
                is_tabu = (
                    (j, item) in tabu_dict
                    and tabu_dict[(j, item)] > iteration
                )

                new_perm = list(perm)
                new_perm.pop(i)
                insert_pos = j if j < i else j - 1
                insert_pos = max(0, min(insert_pos, len(new_perm)))
                new_perm.insert(insert_pos, item)

                new_rolls = _decode_ff(
                    instance.lengths, instance.stock_length, new_perm
                )
                delta = len(new_rolls) - current_num_rolls

                if is_tabu and current_num_rolls + delta >= best_num_rolls:
                    continue

                if delta < best_delta:
                    best_delta = delta
                    best_move = ("insert", i, j, new_perm, new_rolls)

        if best_move is None:
            tabu_dict.clear()
            continue

        # Apply move
        if best_move[0] == "swap":
            _, i, j, new_perm, new_rolls = best_move
            tabu_dict[(i, perm[i])] = iteration + tabu_tenure
            tabu_dict[(j, perm[j])] = iteration + tabu_tenure
        else:
            _, i, j, new_perm, new_rolls = best_move
            tabu_dict[(j, perm[i])] = iteration + tabu_tenure

        perm = new_perm
        current_num_rolls = len(new_rolls)

        if current_num_rolls < best_num_rolls:
            best_num_rolls = current_num_rolls
            best_perm = list(perm)

    best_rolls = _decode_ff(instance.lengths, instance.stock_length, best_perm)
    return _rolls_to_solution(instance, best_rolls)


if __name__ == "__main__":
    from instance import simple_csp_3, classic_csp_4

    print("=== TS for Cutting Stock ===\n")

    for name, inst_fn in [
        ("simple3", simple_csp_3),
        ("classic4", classic_csp_4),
    ]:
        inst = inst_fn()
        sol = tabu_search(inst, seed=42)
        print(f"{name} (LB={inst.lower_bound()}): TS={sol.num_rolls} rolls")
