"""
Local Search for 1D Cutting Stock Problem.

Problem: 1D Cutting Stock (CSP)

Iterative improvement on pattern assignments. Operates on an expanded
item-level representation (individual items assigned to rolls), then
aggregates back to pattern-frequency format.

Neighborhoods:
    - Relocate: move an item to a different roll
    - Swap: exchange items between rolls
    - Merge: try to empty a roll by redistributing its items

Warm-started with FFD-based heuristic.

Complexity: O(iterations * N * R) where N = total items, R = rolls.

References:
    Belov, G. & Scheithauer, G. (2006). A branch-and-cut-and-price
    algorithm for one-dimensional stock cutting and two-dimensional
    two-stage cutting. European Journal of Operational Research,
    171(1), 85-106.
    https://doi.org/10.1016/j.ejor.2004.08.036

    Fleszar, K. & Hindi, K.S. (2002). New heuristics for one-dimensional
    bin packing. Computers & Operations Research, 29(7), 821-839.
    https://doi.org/10.1016/S0305-0548(00)00082-4
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


_inst = _load_mod("csp_instance_ls", os.path.join(_parent_dir, "instance.py"))
CuttingStockInstance = _inst.CuttingStockInstance
CuttingStockSolution = _inst.CuttingStockSolution
validate_solution = _inst.validate_solution

_greedy = _load_mod(
    "csp_greedy_ls",
    os.path.join(_parent_dir, "heuristics", "greedy_csp.py"),
)
ffd_based = _greedy.ffd_based


def local_search(
    instance: CuttingStockInstance,
    max_iterations: int = 1000,
    time_limit: float | None = None,
    seed: int | None = None,
) -> CuttingStockSolution:
    """Solve Cutting Stock using Local Search.

    Args:
        instance: A CuttingStockInstance.
        max_iterations: Maximum number of iterations.
        time_limit: Maximum wall-clock time in seconds.
        seed: Random seed for reproducibility.

    Returns:
        CuttingStockSolution with the best packing found.
    """
    rng = np.random.default_rng(seed)
    start_time = time.time()

    # Warm-start with FFD
    init_sol = ffd_based(instance)

    # Expand to item-level: rolls[r] = list of (type_index, length)
    rolls = _expand_solution(instance, init_sol)
    num_rolls = len(rolls)

    best_rolls = [r[:] for r in rolls]
    best_num = num_rolls

    no_improve = 0

    for iteration in range(max_iterations):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        improved = False

        # Try to merge: empty the least-loaded roll
        roll_loads = [sum(length for _, length in r) for r in rolls]
        if rolls:
            lightest = int(np.argmin(roll_loads))
            if _try_merge(instance, rolls, lightest):
                rolls.pop(lightest)
                num_rolls = len(rolls)
                improved = True

        # Relocate: move items from heavier rolls to lighter rolls
        if not improved:
            improved = _try_relocate(instance, rolls, rng)

        # Swap: exchange items between rolls
        if not improved:
            improved = _try_swap(instance, rolls, rng)

        if improved:
            num_rolls = len(rolls)
            no_improve = 0
            if num_rolls < best_num:
                best_num = num_rolls
                best_rolls = [r[:] for r in rolls]
        else:
            no_improve += 1
            if no_improve >= 10:
                _perturb(instance, rolls, rng)
                no_improve = 0

    return _compact_solution(instance, best_rolls)


def _expand_solution(
    instance: CuttingStockInstance,
    sol: CuttingStockSolution,
) -> list[list[tuple[int, float]]]:
    """Expand pattern-frequency solution to item-level rolls."""
    rolls = []
    for pattern, freq in sol.patterns:
        for _ in range(freq):
            roll = []
            for t in range(instance.m):
                for _ in range(int(pattern[t])):
                    roll.append((t, float(instance.lengths[t])))
            if roll:
                rolls.append(roll)
    return rolls


def _compact_solution(
    instance: CuttingStockInstance,
    rolls: list[list[tuple[int, float]]],
) -> CuttingStockSolution:
    """Compact item-level rolls back to pattern-frequency format."""
    pattern_map: dict[tuple[int, ...], int] = {}
    for roll in rolls:
        counts = [0] * instance.m
        for t, _ in roll:
            counts[t] += 1
        key = tuple(counts)
        pattern_map[key] = pattern_map.get(key, 0) + 1

    patterns = []
    for key, freq in pattern_map.items():
        patterns.append((np.array(key, dtype=int), freq))

    num_rolls = sum(f for _, f in patterns)
    return CuttingStockSolution(patterns=patterns, num_rolls=num_rolls)


def _roll_remaining(instance: CuttingStockInstance, roll: list[tuple[int, float]]) -> float:
    """Remaining capacity of a roll."""
    return instance.stock_length - sum(length for _, length in roll)


def _try_merge(
    instance: CuttingStockInstance,
    rolls: list[list[tuple[int, float]]],
    target: int,
) -> bool:
    """Try to empty roll `target` by distributing items to other rolls."""
    if not rolls[target]:
        return True

    items = rolls[target][:]
    # Sort by decreasing size for best-fit
    items.sort(key=lambda x: -x[1])

    # Try to place each item in another roll
    placements: list[tuple[int, int]] = []  # (item_idx, dest_roll)
    remaining = [_roll_remaining(instance, r) for r in rolls]
    remaining[target] = 0  # Don't place back in target

    for item_idx, (t, length) in enumerate(items):
        placed = False
        # Best fit: smallest sufficient remaining
        best_fit = -1
        best_gap = float("inf")
        for ri in range(len(rolls)):
            if ri == target:
                continue
            gap = remaining[ri] - length
            if gap >= -1e-10 and gap < best_gap:
                best_gap = gap
                best_fit = ri
        if best_fit >= 0:
            placements.append((item_idx, best_fit))
            remaining[best_fit] -= length
            placed = True
        if not placed:
            return False

    # Apply placements
    for item_idx, dest_roll in placements:
        rolls[dest_roll].append(items[item_idx])
    rolls[target].clear()
    return True


def _try_relocate(
    instance: CuttingStockInstance,
    rolls: list[list[tuple[int, float]]],
    rng: np.random.Generator,
) -> bool:
    """Try relocating an item to reduce total waste distribution."""
    if len(rolls) < 2:
        return False

    # Pick a random roll and item
    ri = rng.integers(len(rolls))
    if not rolls[ri]:
        return False

    item_idx = rng.integers(len(rolls[ri]))
    t, length = rolls[ri][item_idx]

    # Try to fit in a roll with enough space (not the source)
    for rj in range(len(rolls)):
        if rj == ri:
            continue
        if _roll_remaining(instance, rolls[rj]) >= length - 1e-10:
            rolls[rj].append((t, length))
            rolls[ri].pop(item_idx)
            if not rolls[ri]:
                rolls.pop(ri)
            return True

    return False


def _try_swap(
    instance: CuttingStockInstance,
    rolls: list[list[tuple[int, float]]],
    rng: np.random.Generator,
) -> bool:
    """Try swapping items between two rolls to improve packing."""
    if len(rolls) < 2:
        return False

    ri, rj = rng.choice(len(rolls), size=2, replace=False)
    if not rolls[ri] or not rolls[rj]:
        return False

    ii = rng.integers(len(rolls[ri]))
    ij = rng.integers(len(rolls[rj]))

    t_i, len_i = rolls[ri][ii]
    t_j, len_j = rolls[rj][ij]

    # Check if swap is feasible
    rem_i = _roll_remaining(instance, rolls[ri])
    rem_j = _roll_remaining(instance, rolls[rj])

    new_rem_i = rem_i + len_i - len_j
    new_rem_j = rem_j + len_j - len_i

    if new_rem_i >= -1e-10 and new_rem_j >= -1e-10:
        # Check if swap reduces max waste variance (balances better)
        old_min_rem = min(rem_i, rem_j)
        new_min_rem = min(new_rem_i, new_rem_j)
        if new_min_rem > old_min_rem + 1e-10:
            rolls[ri][ii] = (t_j, len_j)
            rolls[rj][ij] = (t_i, len_i)
            return True

    return False


def _perturb(
    instance: CuttingStockInstance,
    rolls: list[list[tuple[int, float]]],
    rng: np.random.Generator,
) -> None:
    """Random perturbation: relocate a few items randomly."""
    if len(rolls) < 2:
        return

    for _ in range(3):
        non_empty = [i for i in range(len(rolls)) if rolls[i]]
        if len(non_empty) < 2:
            break
        ri = non_empty[rng.integers(len(non_empty))]
        if not rolls[ri]:
            continue
        item_idx = rng.integers(len(rolls[ri]))
        t, length = rolls[ri][item_idx]

        targets = [j for j in range(len(rolls)) if j != ri
                    and _roll_remaining(instance, rolls[j]) >= length - 1e-10]
        if targets:
            rj = targets[rng.integers(len(targets))]
            rolls[rj].append((t, length))
            rolls[ri].pop(item_idx)
            if not rolls[ri]:
                rolls.pop(ri)


if __name__ == "__main__":
    inst = CuttingStockInstance.random(m=5, seed=42)
    print(f"CSP: {inst.m} types, L={inst.stock_length}, LB={inst.lower_bound()}")

    ffd_sol = ffd_based(inst)
    print(f"FFD: {ffd_sol.num_rolls} rolls")

    ls_sol = local_search(inst, seed=42)
    print(f"LS: {ls_sol.num_rolls} rolls")
