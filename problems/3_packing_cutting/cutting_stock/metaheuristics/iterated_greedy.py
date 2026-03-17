"""
Iterated Greedy for 1D Cutting Stock.

Problem notation: CSP1D

Iterated Greedy repeatedly destroys and reconstructs solutions:
    1. Destroy: remove items from random rolls
    2. Repair: reinsert using best-fit strategy
    3. Accept: keep if number of rolls is reduced or equal

Works on expanded item-level representation internally.
Warm-started with greedy largest-first heuristic.

Complexity: O(iterations * d * R) per run.

References:
    Ruiz, R. & Stuetzle, T. (2007). A simple and effective iterated
    greedy algorithm for the permutation flowshop scheduling problem.
    European Journal of Operational Research, 177(3), 2033-2049.
    https://doi.org/10.1016/j.ejor.2005.12.009

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


_inst = _load_mod("csp_instance_ig", os.path.join(_parent_dir, "instance.py"))
CuttingStockInstance = _inst.CuttingStockInstance
CuttingStockSolution = _inst.CuttingStockSolution

_greedy = _load_mod(
    "csp_greedy_ig",
    os.path.join(_parent_dir, "heuristics", "greedy_csp.py"),
)
greedy_largest_first = _greedy.greedy_largest_first


def iterated_greedy(
    instance: CuttingStockInstance,
    max_iterations: int = 3000,
    d: int | None = None,
    time_limit: float | None = None,
    seed: int | None = None,
) -> CuttingStockSolution:
    """Solve 1D Cutting Stock using Iterated Greedy.

    Args:
        instance: A CuttingStockInstance.
        max_iterations: Maximum number of iterations.
        d: Number of items to remove per iteration.
        time_limit: Maximum wall-clock time in seconds.
        seed: Random seed for reproducibility.

    Returns:
        CuttingStockSolution with the best patterns found.
    """
    rng = np.random.default_rng(seed)
    start_time = time.time()

    # Warm-start
    init_sol = greedy_largest_first(instance)
    rolls = _expand(instance, init_sol)

    total_items = sum(len(r) for r in rolls)
    if d is None:
        d = max(1, total_items // 4)

    best_rolls = [r[:] for r in rolls]
    best_num = len(rolls)

    for iteration in range(max_iterations):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        # Destroy
        all_items = [(ri, ci, item)
                     for ri, r in enumerate(rolls)
                     for ci, item in enumerate(r)]
        if not all_items:
            break

        d_actual = min(d, len(all_items))
        chosen = rng.choice(len(all_items), size=d_actual, replace=False)
        removed = []
        remove_info = sorted(
            [all_items[i] for i in chosen],
            key=lambda x: (x[0], x[1]),
            reverse=True,
        )
        for ri, ci, item in remove_info:
            rolls[ri].pop(ci)
            removed.append(item)

        rolls = [r for r in rolls if r]

        # Repair: reinsert largest first using best-fit
        removed.sort(key=lambda t: instance.lengths[t], reverse=True)
        remaining = [
            instance.stock_length - sum(instance.lengths[i] for i in r)
            for r in rolls
        ]

        for item in removed:
            best_bin = -1
            best_rem = float("inf")
            for ri in range(len(rolls)):
                if remaining[ri] >= instance.lengths[item] - 1e-10:
                    if remaining[ri] < best_rem:
                        best_rem = remaining[ri]
                        best_bin = ri

            if best_bin >= 0:
                rolls[best_bin].append(item)
                remaining[best_bin] -= instance.lengths[item]
            else:
                rolls.append([item])
                remaining.append(instance.stock_length - instance.lengths[item])

        rolls = [r for r in rolls if r]

        if len(rolls) <= best_num:
            best_num = len(rolls)
            best_rolls = [r[:] for r in rolls]

    best_rolls = [r for r in best_rolls if r]
    return _compact(instance, best_rolls)


def _expand(
    instance: CuttingStockInstance, solution: CuttingStockSolution
) -> list[list[int]]:
    """Expand pattern-frequency to individual rolls."""
    rolls = []
    for pattern, freq in solution.patterns:
        for _ in range(freq):
            roll = []
            for t in range(instance.m):
                roll.extend([t] * int(pattern[t]))
            rolls.append(roll)
    return rolls


def _compact(
    instance: CuttingStockInstance, rolls: list[list[int]]
) -> CuttingStockSolution:
    """Compact rolls to pattern-frequency representation."""
    pattern_map = {}
    for roll in rolls:
        counts = np.zeros(instance.m, dtype=int)
        for t in roll:
            counts[t] += 1
        key = tuple(counts)
        pattern_map[key] = pattern_map.get(key, 0) + 1

    patterns = [(np.array(k, dtype=int), v) for k, v in pattern_map.items()]
    num_rolls = sum(f for _, f in patterns)
    return CuttingStockSolution(patterns=patterns, num_rolls=num_rolls)


if __name__ == "__main__":
    inst = CuttingStockInstance.random(m=5, seed=42)
    print(f"Cutting Stock: {inst.m} types, L={inst.stock_length}")
    print(f"Lower bound: {inst.lower_bound()}")

    greedy_sol = greedy_largest_first(inst)
    print(f"Greedy: {greedy_sol.num_rolls} rolls")

    ig_sol = iterated_greedy(inst, seed=42)
    print(f"IG: {ig_sol.num_rolls} rolls")
