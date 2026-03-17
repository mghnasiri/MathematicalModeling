"""
Simulated Annealing for 1D Cutting Stock Problem (CSP).

Problem: CSP1D (1D Cutting Stock)

Representation: A flat list of individual items (expanded from demands).
Each item is identified by its type index. The list is decoded into rolls
using First Fit Decreasing: process items in list order, placing each into
the first roll where it fits.

Neighborhoods:
- Swap: exchange two items of different types in the permutation
- Insert: remove an item and re-insert it at a different position

Warm-started with FFD ordering (descending size).

Complexity: O(iterations * N) per iteration, where N = total items.

References:
    Kirkpatrick, S., Gelatt, C.D. & Vecchi, M.P. (1983). Optimization
    by simulated annealing. Science, 220(4598), 671-680.
    https://doi.org/10.1126/science.220.4598.671

    Falkenauer, E. (1996). A hybrid grouping genetic algorithm for bin
    packing. Journal of Heuristics, 2(1), 5-30.
    https://doi.org/10.1007/BF00226291
"""

from __future__ import annotations

import os
import sys
import math
import time
import importlib.util

import numpy as np

_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_mod(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_inst = _load_mod("csp_instance_sa", os.path.join(_parent_dir, "instance.py"))
CuttingStockInstance = _inst.CuttingStockInstance
CuttingStockSolution = _inst.CuttingStockSolution

_greedy = _load_mod(
    "csp_greedy_sa", os.path.join(_parent_dir, "heuristics", "greedy_csp.py")
)
ffd_based = _greedy.ffd_based


def _decode_ff(
    lengths: np.ndarray,
    stock_length: float,
    perm: list[int],
) -> list[list[int]]:
    """Decode a permutation of items into rolls using First Fit.

    Args:
        lengths: Array of item lengths indexed by type.
        stock_length: Roll capacity.
        perm: Permutation of item indices (each is a type index).

    Returns:
        List of rolls, each a list of item type indices.
    """
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
    """Convert decoded rolls to a CuttingStockSolution with aggregated patterns."""
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


def simulated_annealing(
    instance: CuttingStockInstance,
    max_iterations: int = 10000,
    initial_temp: float | None = None,
    cooling_rate: float = 0.999,
    time_limit: float | None = None,
    seed: int | None = None,
) -> CuttingStockSolution:
    """Solve 1D Cutting Stock using Simulated Annealing.

    Args:
        instance: Cutting stock instance.
        max_iterations: Maximum iterations.
        initial_temp: Initial temperature. If None, auto-calibrated.
        cooling_rate: Geometric cooling factor.
        time_limit: Time limit in seconds.
        seed: Random seed for reproducibility.

    Returns:
        Best CuttingStockSolution found.
    """
    rng = np.random.default_rng(seed)
    start_time = time.time()

    # Expand demands into individual items, sorted by length (descending)
    items: list[int] = []
    for i in range(instance.m):
        items.extend([i] * instance.demands[i])

    total_items = len(items)
    if total_items == 0:
        return CuttingStockSolution(patterns=[], num_rolls=0)

    # Initialize with FFD ordering
    perm = sorted(items, key=lambda i: -instance.lengths[i])
    rolls = _decode_ff(instance.lengths, instance.stock_length, perm)
    current_num_rolls = len(rolls)

    best_perm = list(perm)
    best_num_rolls = current_num_rolls

    if initial_temp is None:
        initial_temp = max(1.0, total_items * 0.3)
    temp = initial_temp

    for iteration in range(max_iterations):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        if total_items < 2:
            break

        new_perm = list(perm)

        if rng.random() < 0.5 or total_items < 3:
            # Swap two items of different types
            i = rng.integers(0, total_items)
            attempts = 0
            j = rng.integers(0, total_items)
            while new_perm[j] == new_perm[i] and attempts < 10:
                j = rng.integers(0, total_items)
                attempts += 1
            new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
        else:
            # Insert: remove and re-insert
            i = rng.integers(0, total_items)
            item = new_perm.pop(i)
            j = rng.integers(0, total_items)
            if j >= i:
                j = max(0, j - 1)
            new_perm.insert(j, item)

        new_rolls = _decode_ff(instance.lengths, instance.stock_length, new_perm)
        new_num_rolls = len(new_rolls)

        delta = new_num_rolls - current_num_rolls
        if delta <= 0 or rng.random() < math.exp(-delta / max(temp, 1e-10)):
            perm = new_perm
            current_num_rolls = new_num_rolls

            if current_num_rolls < best_num_rolls:
                best_num_rolls = current_num_rolls
                best_perm = list(perm)

        temp *= cooling_rate

    # Build final solution from best permutation
    best_rolls = _decode_ff(instance.lengths, instance.stock_length, best_perm)
    return _rolls_to_solution(instance, best_rolls)


if __name__ == "__main__":
    from instance import simple_csp_3, classic_csp_4

    print("=== SA for Cutting Stock ===\n")

    for name, inst_fn in [
        ("simple3", simple_csp_3),
        ("classic4", classic_csp_4),
    ]:
        inst = inst_fn()
        sol = simulated_annealing(inst, seed=42)
        print(f"{name} (LB={inst.lower_bound()}): SA={sol.num_rolls} rolls")
