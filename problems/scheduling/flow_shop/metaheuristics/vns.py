"""
Variable Neighborhood Search (VNS) for Fm | prmu | Cmax.

VNS is a metaheuristic framework based on systematic changes of
neighborhoods (Mladenović & Hansen, 1997). It uses a shaking step
in progressively larger neighborhoods to escape local optima, then
applies local search (descent) in the smallest neighborhood.

The key insight is that a local optimum in one neighborhood structure
is not necessarily local in another. By switching neighborhoods, VNS
can escape basins of attraction that trap simpler local search methods.

Algorithm (General VNS / GVNS):
    1. Initialize with NEH solution, define k_max neighborhoods.
    2. Set k = 1.
    3. Shaking: generate a random neighbor in N_k (k random moves).
    4. Local Search: apply VND (Variable Neighborhood Descent) to the
       shaken solution using insert and swap neighborhoods.
    5. Move or not:
       - If improved, adopt new solution, reset k = 1.
       - Otherwise, k = k + 1 (try next neighborhood).
    6. If k > k_max, reset k = 1.
    7. Repeat until stopping criterion.

Neighborhoods (ordered by increasing perturbation):
    N_1: single insertion move
    N_2: single swap move
    N_3: two consecutive insertion moves
    N_4: Or-opt (block of 2-3 jobs relocated)
    N_5: three random insertion moves

Notation: Fm | prmu | Cmax
Complexity: O(iterations * n^2 * m) per VND pass.

Reference:
    Mladenović, N. & Hansen, P. (1997). Variable neighborhood search.
    Computers & Operations Research, 24(11), 1097-1100.
    https://doi.org/10.1016/S0305-0548(97)00031-2

    Zobolas, G.I., Tarantilis, C.D. & Ioannou, G. (2009). Minimizing
    makespan in permutation flow shop scheduling problems using a
    hybrid metaheuristic algorithm. Computers & Operations Research,
    36(4), 1249-1267.
    https://doi.org/10.1016/j.cor.2008.01.007
"""

from __future__ import annotations

import sys
import os
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from instance import FlowShopInstance, FlowShopSolution, compute_makespan
from heuristics.neh import neh


def _insertion_ls(
    instance: FlowShopInstance,
    perm: list[int],
    makespan: int,
) -> tuple[list[int], int]:
    """First-improvement insertion local search."""
    n = len(perm)
    current = list(perm)
    current_ms = makespan
    improved = True

    while improved:
        improved = False
        for i in range(n):
            job = current[i]
            remaining = current[:i] + current[i + 1:]
            for j in range(n):
                if j == i:
                    continue
                candidate = remaining[:j] + [job] + remaining[j:]
                ms = compute_makespan(instance, candidate)
                if ms < current_ms:
                    current = candidate
                    current_ms = ms
                    improved = True
                    break
            if improved:
                break

    return current, current_ms


def _swap_ls(
    instance: FlowShopInstance,
    perm: list[int],
    makespan: int,
) -> tuple[list[int], int]:
    """First-improvement swap local search."""
    n = len(perm)
    current = list(perm)
    current_ms = makespan
    improved = True

    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                candidate = list(current)
                candidate[i], candidate[j] = candidate[j], candidate[i]
                ms = compute_makespan(instance, candidate)
                if ms < current_ms:
                    current = candidate
                    current_ms = ms
                    improved = True
                    break
            if improved:
                break

    return current, current_ms


def _vnd(
    instance: FlowShopInstance,
    perm: list[int],
    makespan: int,
) -> tuple[list[int], int]:
    """Variable Neighborhood Descent: insertion then swap."""
    current, current_ms = list(perm), makespan

    changed = True
    while changed:
        changed = False

        new_perm, new_ms = _insertion_ls(instance, current, current_ms)
        if new_ms < current_ms:
            current, current_ms = new_perm, new_ms
            changed = True

        new_perm, new_ms = _swap_ls(instance, current, current_ms)
        if new_ms < current_ms:
            current, current_ms = new_perm, new_ms
            changed = True

    return current, current_ms


def _shake(
    perm: list[int],
    k: int,
    rng: np.random.Generator,
) -> list[int]:
    """Shaking: apply k random insertion moves."""
    result = list(perm)
    n = len(result)
    if n < 2:
        return result

    for _ in range(k):
        i = rng.integers(0, n)
        job = result.pop(i)
        j = rng.integers(0, n)
        if j >= i:
            j = max(0, j - 1)
        result.insert(j, job)

    return result


def vns(
    instance: FlowShopInstance,
    k_max: int = 5,
    max_iterations: int = 200,
    time_limit: float | None = None,
    seed: int | None = None,
) -> FlowShopSolution:
    """Solve PFSP using Variable Neighborhood Search.

    Args:
        instance: Flow shop instance.
        k_max: Maximum neighborhood index (shake strength).
        max_iterations: Maximum outer iterations.
        time_limit: Time limit in seconds.
        seed: Random seed for reproducibility.

    Returns:
        Best FlowShopSolution found.
    """
    rng = np.random.default_rng(seed)
    start_time = time.time()

    # Initialize with NEH
    neh_sol = neh(instance)
    current = list(neh_sol.permutation)
    current_ms = neh_sol.makespan
    best = list(current)
    best_ms = current_ms

    iteration = 0
    while iteration < max_iterations:
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        k = 1
        while k <= k_max:
            if time_limit is not None and time.time() - start_time >= time_limit:
                break

            # Shaking
            shaken = _shake(current, k, rng)
            shaken_ms = compute_makespan(instance, shaken)

            # Local search (VND)
            improved, improved_ms = _vnd(instance, shaken, shaken_ms)

            # Move or not
            if improved_ms < current_ms:
                current = improved
                current_ms = improved_ms
                k = 1  # Reset to smallest neighborhood

                if current_ms < best_ms:
                    best = list(current)
                    best_ms = current_ms
            else:
                k += 1

        iteration += 1

    return FlowShopSolution(permutation=best, makespan=best_ms)


if __name__ == "__main__":
    instance = FlowShopInstance.random(n=20, m=5, seed=42)

    neh_sol = neh(instance)
    print(f"NEH: makespan = {neh_sol.makespan}")

    vns_sol = vns(instance, max_iterations=50, seed=42)
    print(f"VNS: makespan = {vns_sol.makespan}")
    print(
        f"Improvement: "
        f"{(neh_sol.makespan - vns_sol.makespan) / neh_sol.makespan * 100:.1f}%"
    )
