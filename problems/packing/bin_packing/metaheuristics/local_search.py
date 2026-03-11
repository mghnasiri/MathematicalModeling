"""
Local Search for 1D Bin Packing.

Problem: 1D Bin Packing (BPP)

Iterative improvement using swap and relocate neighborhoods on item-to-bin
assignments. Aims to reduce the number of bins by repacking items.

Neighborhoods:
    - Swap: exchange two items between different bins
    - Relocate: move an item to a different bin
    - Merge: try to empty a bin by redistributing its items

Warm-started with First Fit Decreasing heuristic.

Complexity: O(iterations * n * B) where B = number of bins.

References:
    Fleszar, K. & Hindi, K.S. (2002). New heuristics for one-dimensional
    bin packing. Computers & Operations Research, 29(7), 821-839.
    https://doi.org/10.1016/S0305-0548(00)00082-4

    Alvim, A.C.F., Ribeiro, C.C., Glover, F. & Aloise, D.J. (2004).
    A hybrid improvement heuristic for the one-dimensional bin packing
    problem. Journal of Heuristics, 10(2), 205-229.
    https://doi.org/10.1023/B:HEUR.0000026267.44673.ed
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


_inst = _load_module("bpp_instance_ls", os.path.join(_parent_dir, "instance.py"))
BinPackingInstance = _inst.BinPackingInstance
BinPackingSolution = _inst.BinPackingSolution


def local_search(
    instance: BinPackingInstance,
    max_iterations: int = 1000,
    time_limit: float | None = None,
    seed: int | None = None,
) -> BinPackingSolution:
    """Solve 1D Bin Packing using local search.

    Args:
        instance: A BinPackingInstance.
        max_iterations: Maximum number of iterations.
        time_limit: Maximum wall-clock time in seconds.
        seed: Random seed for reproducibility.

    Returns:
        BinPackingSolution with the best packing found.
    """
    rng = np.random.default_rng(seed)
    n = instance.n
    C = instance.capacity
    start_time = time.time()

    # Warm-start with FFD
    _ff_mod = _load_module(
        "bpp_ff_ls", os.path.join(_parent_dir, "heuristics", "first_fit.py")
    )
    init_sol = _ff_mod.first_fit_decreasing(instance)
    bins = [b[:] for b in init_sol.bins]
    remaining = [C - sum(instance.sizes[i] for i in b) for b in bins]

    best_num_bins = len(bins)
    best_bins = [b[:] for b in bins]

    for iteration in range(max_iterations):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        # Try to merge the least-loaded bin into others
        improved = _try_merge(instance, bins, remaining, C)

        if not improved:
            # Try relocate moves
            improved = _try_relocate(instance, bins, remaining, C, rng)

        if not improved:
            # Try swap moves
            improved = _try_swap(instance, bins, remaining, C, rng)

        # Clean up empty bins
        bins = [b for b in bins if b]
        remaining = [C - sum(instance.sizes[i] for i in b) for b in bins]

        if len(bins) < best_num_bins:
            best_num_bins = len(bins)
            best_bins = [b[:] for b in bins]

        if not improved:
            # Perturbation: random relocate
            if len(bins) >= 2:
                src_idx = rng.integers(0, len(bins))
                if bins[src_idx]:
                    item_pos = rng.integers(0, len(bins[src_idx]))
                    item = bins[src_idx][item_pos]
                    size = instance.sizes[item]

                    candidates = [i for i in range(len(bins))
                                  if i != src_idx and remaining[i] >= size - 1e-10]
                    if candidates:
                        dst_idx = rng.choice(candidates)
                        bins[src_idx].pop(item_pos)
                        remaining[src_idx] += size
                        bins[dst_idx].append(item)
                        remaining[dst_idx] -= size

    best_bins = [b for b in best_bins if b]
    return BinPackingSolution(bins=best_bins, num_bins=len(best_bins))


def _try_merge(
    instance: BinPackingInstance,
    bins: list[list[int]],
    remaining: list[float],
    C: float,
) -> bool:
    """Try to empty the least-loaded bin by distributing items to others."""
    if len(bins) <= 1:
        return False

    # Find the bin with least total load
    loads = [C - r for r in remaining]
    min_idx = int(np.argmin(loads))
    items_to_move = bins[min_idx][:]

    # Try to place all items into other bins
    placements = []  # (item, dst_bin)
    temp_remaining = remaining[:]

    for item in items_to_move:
        size = instance.sizes[item]
        placed = False
        for dst in range(len(bins)):
            if dst == min_idx:
                continue
            if temp_remaining[dst] >= size - 1e-10:
                placements.append((item, dst))
                temp_remaining[dst] -= size
                placed = True
                break
        if not placed:
            return False  # Can't empty this bin

    # Apply placements
    for item, dst in placements:
        bins[dst].append(item)
        remaining[dst] -= instance.sizes[item]

    bins[min_idx].clear()
    remaining[min_idx] = C
    return True


def _try_relocate(
    instance: BinPackingInstance,
    bins: list[list[int]],
    remaining: list[float],
    C: float,
    rng: np.random.Generator,
) -> bool:
    """Try relocating items to make a bin empty."""
    non_empty = [i for i in range(len(bins)) if bins[i]]
    if len(non_empty) < 2:
        return False

    # Sample some relocate moves
    for _ in range(min(len(non_empty) * 5, 50)):
        src_idx = rng.choice(non_empty)
        if not bins[src_idx]:
            continue

        item_pos = rng.integers(0, len(bins[src_idx]))
        item = bins[src_idx][item_pos]
        size = instance.sizes[item]

        # Find a bin that can fit this item and would reduce fragmentation
        best_dst = -1
        best_waste = float("inf")
        for dst in range(len(bins)):
            if dst == src_idx:
                continue
            if remaining[dst] >= size - 1e-10:
                waste = remaining[dst] - size
                if waste < best_waste:
                    best_waste = waste
                    best_dst = dst

        if best_dst >= 0 and best_waste < remaining[src_idx] - size:
            bins[src_idx].pop(item_pos)
            remaining[src_idx] += size
            bins[best_dst].append(item)
            remaining[best_dst] -= size
            return True

    return False


def _try_swap(
    instance: BinPackingInstance,
    bins: list[list[int]],
    remaining: list[float],
    C: float,
    rng: np.random.Generator,
) -> bool:
    """Try swapping items between bins to reduce waste."""
    non_empty = [i for i in range(len(bins)) if bins[i]]
    if len(non_empty) < 2:
        return False

    for _ in range(min(len(non_empty) * 3, 30)):
        b1_idx, b2_idx = rng.choice(non_empty, size=2, replace=False)
        if not bins[b1_idx] or not bins[b2_idx]:
            continue

        i1 = rng.integers(0, len(bins[b1_idx]))
        i2 = rng.integers(0, len(bins[b2_idx]))

        item1 = bins[b1_idx][i1]
        item2 = bins[b2_idx][i2]
        s1, s2 = instance.sizes[item1], instance.sizes[item2]

        # Check feasibility after swap
        new_rem1 = remaining[b1_idx] + s1 - s2
        new_rem2 = remaining[b2_idx] + s2 - s1

        if new_rem1 >= -1e-10 and new_rem2 >= -1e-10:
            # Only swap if it reduces max waste (helps future merges)
            old_waste = max(remaining[b1_idx], remaining[b2_idx])
            new_waste = max(new_rem1, new_rem2)
            if new_waste < old_waste - 1e-10:
                bins[b1_idx][i1] = item2
                bins[b2_idx][i2] = item1
                remaining[b1_idx] = new_rem1
                remaining[b2_idx] = new_rem2
                return True

    return False


if __name__ == "__main__":
    from instance import easy_bpp_6, tight_bpp_8, uniform_bpp_10

    print("=== Local Search for 1D Bin Packing ===\n")

    for name, inst_fn in [
        ("easy6", easy_bpp_6),
        ("tight8", tight_bpp_8),
        ("uniform10", uniform_bpp_10),
    ]:
        inst = inst_fn()
        sol = local_search(inst, seed=42)
        print(f"{name}: {sol.num_bins} bins (L1={inst.lower_bound_l1()}, "
              f"L2={inst.lower_bound_l2()})")
