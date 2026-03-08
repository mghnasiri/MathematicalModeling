"""
Local Search Neighborhoods for Fm | prmu | Cmax

Provides standalone local search procedures that can be used independently
or as building blocks inside metaheuristics (IG, SA, GA, etc.).

Three classical neighborhoods are implemented:

1. **Pairwise Swap** (adjacent & general):
   - Adjacent swap: swap jobs at positions k and k+1
   - General swap: swap jobs at any two positions i, j
   - Neighborhood size: O(n) adjacent, O(n²) general

2. **Insertion (Or-opt-1)**:
   - Remove a job from position i and re-insert at position j
   - This is the strongest single-job neighborhood
   - Neighborhood size: O(n²)

3. **Or-opt (block moves)**:
   - Remove a block of 2 or 3 consecutive jobs and re-insert elsewhere
   - Generalizes insertion to small blocks
   - Neighborhood size: O(n²) per block size

Each neighborhood supports two search strategies:
- **First improvement**: accept the first improving move found
- **Best improvement**: evaluate all neighbors, pick the best

The search iterates until no improvement is found (local optimum).

Reference:
    Taillard, E. (1990). "Some efficient heuristic methods for the flow
    shop sequencing problem"
    EJOR, 47(1):65-74.

    Ruiz, R. & Stützle, T. (2007). "A simple and effective iterated
    greedy algorithm for the PFSP"
    EJOR, 177(3):2033-2049.
"""

from __future__ import annotations
import sys
import os
from enum import Enum
from typing import Callable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from instance import FlowShopInstance, FlowShopSolution, compute_makespan


class SearchStrategy(Enum):
    """Local search pivoting rule."""
    FIRST_IMPROVEMENT = "first"
    BEST_IMPROVEMENT = "best"


# ──────────────────────────────────────────────
# Pairwise Swap Neighborhood
# ──────────────────────────────────────────────

def swap_local_search(
    instance: FlowShopInstance,
    initial_permutation: list[int],
    strategy: SearchStrategy = SearchStrategy.FIRST_IMPROVEMENT,
    adjacent_only: bool = False,
) -> FlowShopSolution:
    """
    Improve a permutation using the swap neighborhood.

    Args:
        instance: A FlowShopInstance.
        initial_permutation: Starting permutation.
        strategy: FIRST_IMPROVEMENT or BEST_IMPROVEMENT.
        adjacent_only: If True, only swap adjacent pairs (faster, smaller neighborhood).

    Returns:
        FlowShopSolution at a local optimum w.r.t. swaps.
    """
    perm = list(initial_permutation)
    n = len(perm)
    current_ms = compute_makespan(instance, perm)
    improved = True

    while improved:
        improved = False
        best_ms = current_ms
        best_i, best_j = -1, -1

        for i in range(n - 1):
            j_range = [i + 1] if adjacent_only else range(i + 1, n)
            for j in j_range:
                # Swap positions i and j
                perm[i], perm[j] = perm[j], perm[i]
                new_ms = compute_makespan(instance, perm)

                if new_ms < best_ms:
                    if strategy == SearchStrategy.FIRST_IMPROVEMENT:
                        current_ms = new_ms
                        improved = True
                        break  # Accept immediately
                    else:
                        best_ms = new_ms
                        best_i, best_j = i, j

                # Undo swap
                perm[i], perm[j] = perm[j], perm[i]

            if improved and strategy == SearchStrategy.FIRST_IMPROVEMENT:
                break

        if strategy == SearchStrategy.BEST_IMPROVEMENT and best_ms < current_ms:
            perm[best_i], perm[best_j] = perm[best_j], perm[best_i]
            current_ms = best_ms
            improved = True

    return FlowShopSolution(permutation=perm, makespan=current_ms)


# ──────────────────────────────────────────────
# Insertion Neighborhood
# ──────────────────────────────────────────────

def insertion_local_search(
    instance: FlowShopInstance,
    initial_permutation: list[int],
    strategy: SearchStrategy = SearchStrategy.FIRST_IMPROVEMENT,
) -> FlowShopSolution:
    """
    Improve a permutation using the insertion neighborhood.

    For each job, tries removing it and re-inserting at every other position.
    This is the most effective single-job neighborhood for PFSP.

    Args:
        instance: A FlowShopInstance.
        initial_permutation: Starting permutation.
        strategy: FIRST_IMPROVEMENT or BEST_IMPROVEMENT.

    Returns:
        FlowShopSolution at a local optimum w.r.t. insertions.
    """
    perm = list(initial_permutation)
    n = len(perm)
    current_ms = compute_makespan(instance, perm)
    improved = True

    while improved:
        improved = False
        best_ms = current_ms
        best_remove = -1
        best_insert = -1

        for i in range(n):
            # Remove job at position i
            job = perm[i]
            remaining = perm[:i] + perm[i + 1:]

            for j in range(n):
                if j == i:
                    continue
                # Insert job at position j in the remaining sequence
                candidate = remaining[:j] + [job] + remaining[j:]
                new_ms = compute_makespan(instance, candidate)

                if new_ms < best_ms:
                    if strategy == SearchStrategy.FIRST_IMPROVEMENT:
                        perm = candidate
                        current_ms = new_ms
                        improved = True
                        break
                    else:
                        best_ms = new_ms
                        best_remove = i
                        best_insert = j

            if improved and strategy == SearchStrategy.FIRST_IMPROVEMENT:
                break

        if strategy == SearchStrategy.BEST_IMPROVEMENT and best_ms < current_ms:
            job = perm[best_remove]
            remaining = perm[:best_remove] + perm[best_remove + 1:]
            perm = remaining[:best_insert] + [job] + remaining[best_insert:]
            current_ms = best_ms
            improved = True

    return FlowShopSolution(permutation=perm, makespan=current_ms)


# ──────────────────────────────────────────────
# Or-opt (Block Move) Neighborhood
# ──────────────────────────────────────────────

def oropt_local_search(
    instance: FlowShopInstance,
    initial_permutation: list[int],
    block_sizes: list[int] | None = None,
    strategy: SearchStrategy = SearchStrategy.FIRST_IMPROVEMENT,
) -> FlowShopSolution:
    """
    Improve a permutation using the Or-opt (block move) neighborhood.

    Removes a block of consecutive jobs and re-inserts it at another position.
    Or-opt with block_size=1 is equivalent to the insertion neighborhood.

    Args:
        instance: A FlowShopInstance.
        initial_permutation: Starting permutation.
        block_sizes: Sizes of blocks to try (default: [2, 3]).
        strategy: FIRST_IMPROVEMENT or BEST_IMPROVEMENT.

    Returns:
        FlowShopSolution at a local optimum w.r.t. block moves.
    """
    if block_sizes is None:
        block_sizes = [2, 3]

    perm = list(initial_permutation)
    n = len(perm)
    current_ms = compute_makespan(instance, perm)
    improved = True

    while improved:
        improved = False
        best_ms = current_ms
        best_perm = None

        for block_size in block_sizes:
            if block_size >= n:
                continue

            for i in range(n - block_size + 1):
                # Extract block starting at position i
                block = perm[i:i + block_size]
                remaining = perm[:i] + perm[i + block_size:]

                for j in range(len(remaining) + 1):
                    if j == i:
                        continue  # Same position
                    candidate = remaining[:j] + block + remaining[j:]
                    new_ms = compute_makespan(instance, candidate)

                    if new_ms < best_ms:
                        if strategy == SearchStrategy.FIRST_IMPROVEMENT:
                            perm = candidate
                            current_ms = new_ms
                            improved = True
                            break
                        else:
                            best_ms = new_ms
                            best_perm = candidate

                if improved and strategy == SearchStrategy.FIRST_IMPROVEMENT:
                    break
            if improved and strategy == SearchStrategy.FIRST_IMPROVEMENT:
                break

        if strategy == SearchStrategy.BEST_IMPROVEMENT and best_perm is not None:
            perm = best_perm
            current_ms = best_ms
            improved = True

    return FlowShopSolution(permutation=perm, makespan=current_ms)


# ──────────────────────────────────────────────
# Combined / Variable Neighborhood Descent
# ──────────────────────────────────────────────

def variable_neighborhood_descent(
    instance: FlowShopInstance,
    initial_permutation: list[int],
    neighborhoods: list[str] | None = None,
) -> FlowShopSolution:
    """
    Variable Neighborhood Descent (VND) — systematically cycle through
    multiple neighborhoods until no improvement is found in any.

    The VND framework (Hansen & Mladenović, 2001) applies neighborhood
    structures sequentially. When one neighborhood finds an improvement,
    restart from the first neighborhood. Stop when no neighborhood improves.

    Args:
        instance: A FlowShopInstance.
        initial_permutation: Starting permutation.
        neighborhoods: Ordered list of neighborhood names to apply.
            Options: "swap", "adjacent_swap", "insertion", "oropt"
            Default: ["insertion", "swap"] — insertion first (strongest).

    Returns:
        FlowShopSolution at a local optimum w.r.t. all neighborhoods.
    """
    if neighborhoods is None:
        neighborhoods = ["insertion", "swap"]

    # Map names to search functions
    search_fns: dict[str, Callable] = {
        "swap": lambda inst, perm: swap_local_search(
            inst, perm, SearchStrategy.FIRST_IMPROVEMENT, adjacent_only=False
        ),
        "adjacent_swap": lambda inst, perm: swap_local_search(
            inst, perm, SearchStrategy.FIRST_IMPROVEMENT, adjacent_only=True
        ),
        "insertion": lambda inst, perm: insertion_local_search(
            inst, perm, SearchStrategy.FIRST_IMPROVEMENT
        ),
        "oropt": lambda inst, perm: oropt_local_search(
            inst, perm, strategy=SearchStrategy.FIRST_IMPROVEMENT
        ),
    }

    perm = list(initial_permutation)
    current_ms = compute_makespan(instance, perm)
    k = 0

    while k < len(neighborhoods):
        name = neighborhoods[k]
        fn = search_fns[name]
        sol = fn(instance, perm)

        if sol.makespan < current_ms:
            perm = sol.permutation
            current_ms = sol.makespan
            k = 0  # Restart from first neighborhood
        else:
            k += 1  # Move to next neighborhood

    return FlowShopSolution(permutation=perm, makespan=current_ms)


if __name__ == "__main__":
    import numpy as np

    print("=== Local Search Methods ===\n")

    instance = FlowShopInstance.random(n=20, m=5, seed=42)

    # Start from Palmer's (weak) solution to see improvement
    from heuristics.palmers_slope import palmers_slope
    from heuristics.neh import neh

    palmer_sol = palmers_slope(instance)
    neh_sol = neh(instance)
    print(f"Palmer initial:  {palmer_sol.makespan}")
    print(f"NEH reference:   {neh_sol.makespan}")

    # Test each neighborhood on Palmer's solution
    print("\n--- Improving Palmer's solution ---")

    sol_adj_swap = swap_local_search(
        instance, palmer_sol.permutation,
        strategy=SearchStrategy.FIRST_IMPROVEMENT, adjacent_only=True
    )
    print(f"Adjacent swap:   {sol_adj_swap.makespan}")

    sol_swap = swap_local_search(
        instance, palmer_sol.permutation,
        strategy=SearchStrategy.FIRST_IMPROVEMENT, adjacent_only=False
    )
    print(f"General swap:    {sol_swap.makespan}")

    sol_insert = insertion_local_search(
        instance, palmer_sol.permutation,
        strategy=SearchStrategy.FIRST_IMPROVEMENT
    )
    print(f"Insertion:       {sol_insert.makespan}")

    sol_oropt = oropt_local_search(
        instance, palmer_sol.permutation,
        strategy=SearchStrategy.FIRST_IMPROVEMENT
    )
    print(f"Or-opt(2,3):     {sol_oropt.makespan}")

    sol_vnd = variable_neighborhood_descent(
        instance, palmer_sol.permutation,
        neighborhoods=["insertion", "swap", "oropt"]
    )
    print(f"VND (I+S+O):     {sol_vnd.makespan}")

    # Best improvement vs First improvement
    print("\n--- Best vs First Improvement (swap on Palmer) ---")
    sol_best = swap_local_search(
        instance, palmer_sol.permutation,
        strategy=SearchStrategy.BEST_IMPROVEMENT, adjacent_only=False
    )
    print(f"Best improvement:  {sol_best.makespan}")
    print(f"First improvement: {sol_swap.makespan}")

    # VND on NEH solution (already good)
    print("\n--- VND on NEH solution ---")
    sol_vnd_neh = variable_neighborhood_descent(
        instance, neh_sol.permutation,
        neighborhoods=["insertion", "swap", "oropt"]
    )
    print(f"NEH:             {neh_sol.makespan}")
    print(f"NEH + VND:       {sol_vnd_neh.makespan}")
