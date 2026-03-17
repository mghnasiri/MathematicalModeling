"""
Variable Neighborhood Search for 1D Cutting Stock.

Problem notation: CSP1D

VNS uses multiple neighborhood structures on cutting patterns:
    N1: Move — move items between rolls
    N2: Swap — exchange items between rolls
    N3: Multi-move — move k items simultaneously

Local search tries to empty rolls by redistributing their items.
Warm-started with greedy largest-first heuristic.

Complexity: O(iterations * k_max * N * R) per run,
where N = total items and R = number of rolls.

References:
    Mladenović, N. & Hansen, P. (1997). Variable neighborhood search.
    Computers & Operations Research, 24(11), 1097-1100.
    https://doi.org/10.1016/S0305-0548(97)00031-2

    Gilmore, P.C. & Gomory, R.E. (1961). A linear programming approach
    to the cutting-stock problem. Operations Research, 9(6), 849-859.
    https://doi.org/10.1287/opre.9.6.849
"""

from __future__ import annotations

import sys
import os
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


_inst = _load_mod("csp_instance_vns", os.path.join(_parent_dir, "instance.py"))
CuttingStockInstance = _inst.CuttingStockInstance
CuttingStockSolution = _inst.CuttingStockSolution

_greedy = _load_mod(
    "csp_greedy_vns",
    os.path.join(_parent_dir, "heuristics", "greedy_csp.py"),
)
greedy_largest_first = _greedy.greedy_largest_first


def vns(
    instance: CuttingStockInstance,
    max_iterations: int = 500,
    k_max: int = 3,
    time_limit: float | None = None,
    seed: int | None = None,
) -> CuttingStockSolution:
    """Solve 1D Cutting Stock using Variable Neighborhood Search.

    Args:
        instance: A CuttingStockInstance.
        max_iterations: Maximum number of iterations.
        k_max: Maximum neighborhood size.
        time_limit: Maximum wall-clock time in seconds.
        seed: Random seed for reproducibility.

    Returns:
        CuttingStockSolution with the best patterns found.
    """
    rng = np.random.default_rng(seed)
    start_time = time.time()

    # Warm-start with greedy
    init_sol = greedy_largest_first(instance)
    rolls = _expand_to_rolls(instance, init_sol)

    best_rolls = [r[:] for r in rolls]
    best_num = len(rolls)

    for iteration in range(max_iterations):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        k = 1
        while k <= k_max:
            if time_limit is not None and time.time() - start_time >= time_limit:
                break

            # Shaking
            shaken = [r[:] for r in rolls]
            _shake(instance, shaken, k, rng)
            shaken = [r for r in shaken if r]

            # Local search
            _local_search(instance, shaken)
            shaken = [r for r in shaken if r]

            if len(shaken) < len(rolls):
                rolls = shaken
                k = 1

                if len(rolls) < best_num:
                    best_num = len(rolls)
                    best_rolls = [r[:] for r in rolls]
            else:
                k += 1

    best_rolls = [r for r in best_rolls if r]
    return _compact_to_solution(instance, best_rolls)


def _expand_to_rolls(
    instance: CuttingStockInstance, solution: CuttingStockSolution
) -> list[list[int]]:
    """Expand pattern-frequency representation to individual rolls.

    Each roll is a list of item type indices (with repetition).
    """
    rolls = []
    for pattern, freq in solution.patterns:
        for _ in range(freq):
            roll = []
            for item_type in range(instance.m):
                roll.extend([item_type] * int(pattern[item_type]))
            rolls.append(roll)
    return rolls


def _compact_to_solution(
    instance: CuttingStockInstance, rolls: list[list[int]]
) -> CuttingStockSolution:
    """Compact rolls back to pattern-frequency representation."""
    pattern_map = {}
    for roll in rolls:
        counts = np.zeros(instance.m, dtype=int)
        for item_type in roll:
            counts[item_type] += 1
        key = tuple(counts)
        if key in pattern_map:
            pattern_map[key] += 1
        else:
            pattern_map[key] = 1

    patterns = [(np.array(k, dtype=int), v) for k, v in pattern_map.items()]
    num_rolls = sum(f for _, f in patterns)
    return CuttingStockSolution(patterns=patterns, num_rolls=num_rolls)


def _roll_usage(instance: CuttingStockInstance, roll: list[int]) -> float:
    """Compute total length used on a roll."""
    return sum(instance.lengths[i] for i in roll)


def _shake(
    instance: CuttingStockInstance,
    rolls: list[list[int]],
    k: int,
    rng: np.random.Generator,
) -> None:
    """Shake: move k random items between rolls."""
    for _ in range(k):
        non_empty = [i for i in range(len(rolls)) if rolls[i]]
        if len(non_empty) < 2:
            break
        src_idx = non_empty[rng.integers(len(non_empty))]
        if not rolls[src_idx]:
            continue
        item_idx = rng.integers(len(rolls[src_idx]))
        item = rolls[src_idx][item_idx]

        others = [i for i in range(len(rolls)) if i != src_idx]
        if not others:
            continue
        dst_idx = others[rng.integers(len(others))]

        used = _roll_usage(instance, rolls[dst_idx])
        if used + instance.lengths[item] <= instance.stock_length + 1e-10:
            rolls[src_idx].pop(item_idx)
            rolls[dst_idx].append(item)


def _local_search(
    instance: CuttingStockInstance,
    rolls: list[list[int]],
) -> None:
    """Local search: try to empty rolls by redistributing items."""
    improved = True
    while improved:
        improved = False
        # Sort by usage (least used first — most likely to be emptied)
        roll_usage = [_roll_usage(instance, r) for r in rolls]
        order = sorted(range(len(rolls)), key=lambda i: roll_usage[i])

        for ri in order:
            if not rolls[ri]:
                continue
            items = rolls[ri][:]

            # Compute remaining capacity for other rolls
            remaining = []
            for j in range(len(rolls)):
                if j == ri:
                    remaining.append(0.0)
                else:
                    remaining.append(instance.stock_length - _roll_usage(instance, rolls[j]))

            # Try to place all items (largest first)
            items_sorted = sorted(items, key=lambda i: instance.lengths[i], reverse=True)
            placement = {}
            rem_copy = remaining[:]

            success = True
            for item in items_sorted:
                best_bin = -1
                best_rem = float("inf")
                for j in range(len(rolls)):
                    if j == ri:
                        continue
                    if rem_copy[j] >= instance.lengths[item] - 1e-10:
                        if rem_copy[j] < best_rem:
                            best_rem = rem_copy[j]
                            best_bin = j
                if best_bin >= 0:
                    if item not in placement:
                        placement[item] = (best_bin, [])
                    placement[item][1].append(best_bin)
                    rem_copy[best_bin] -= instance.lengths[item]
                else:
                    success = False
                    break

            if success:
                rolls[ri].clear()
                for item in items_sorted:
                    target = placement[item][1].pop(0)
                    rolls[target].append(item)
                improved = True
                break

        rolls[:] = [r for r in rolls if r]


if __name__ == "__main__":
    inst = CuttingStockInstance.random(m=5, seed=42)
    print(f"Cutting Stock: {inst.m} types, L={inst.stock_length}")
    print(f"Lower bound: {inst.lower_bound()}")

    greedy_sol = greedy_largest_first(inst)
    print(f"Greedy: {greedy_sol.num_rolls} rolls")

    vns_sol = vns(inst, seed=42)
    print(f"VNS: {vns_sol.num_rolls} rolls")
