"""
Scatter Search (SS) for Fm | prmu | Cmax.

Scatter Search is a population-based metaheuristic that operates on a
reference set (RefSet) of diverse, high-quality solutions. Unlike genetic
algorithms that rely on randomized crossover, SS uses systematic combination
methods and structured solution improvement (Glover, 1977; Glover et al., 2000).

Algorithm (Fernandez-Viagas & Framinan, 2019):
    1. Generate a diverse initial population using NEH + random perturbations.
    2. Select RefSet: b1 best solutions + b2 most diverse solutions.
    3. Repeat until termination:
       a. Generate subsets (pairs) from RefSet.
       b. Combine pairs using path relinking (insert moves along path
          from one solution to another).
       c. Improve combined solutions with local search.
       d. Update RefSet if new solutions improve quality or diversity.
    4. Return the best solution found.

Complexity: O(iterations * |RefSet|^2 * n^2 * m) per iteration.

Reference:
    Glover, F., Laguna, M. & Marti, R. (2000). Fundamentals of scatter
    search and path relinking. Control and Cybernetics, 29(3), 653-684.

    Fernandez-Viagas, V. & Framinan, J.M. (2019). A best-of-breed
    iterated greedy for the permutation flowshop scheduling problem.
    Computers & Operations Research, 112, 104766.

    Framinan, J.M., Gupta, J.N.D. & Leisten, R. (2004). A review and
    classification of heuristics for permutation flow-shop scheduling
    with makespan objective. Journal of the Operational Research Society,
    55(12), 1243-1255.
    https://doi.org/10.1057/palgrave.jors.2601784
"""

from __future__ import annotations

import sys
import os
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from instance import FlowShopInstance, FlowShopSolution, compute_makespan
from heuristics.neh import neh


def _distance(perm1: list[int], perm2: list[int]) -> int:
    """Compute the Kendall tau distance (number of pairwise disagreements).

    Args:
        perm1: First permutation.
        perm2: Second permutation.

    Returns:
        Number of pairwise disagreements.
    """
    n = len(perm1)
    pos1 = [0] * n
    pos2 = [0] * n
    for i in range(n):
        pos1[perm1[i]] = i
        pos2[perm2[i]] = i

    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if (pos1[i] - pos1[j]) * (pos2[i] - pos2[j]) < 0:
                count += 1
    return count


def _insertion_improve(
    instance: FlowShopInstance,
    permutation: list[int],
) -> tuple[list[int], int]:
    """Insertion-based local search (one full pass).

    Args:
        instance: Flow shop instance.
        permutation: Current permutation.

    Returns:
        Tuple of (improved permutation, makespan).
    """
    perm = list(permutation)
    ms = compute_makespan(instance, perm)
    n = len(perm)

    for i in range(n):
        job = perm.pop(i)
        best_pos = i
        best_ms = ms

        for j in range(n):
            perm.insert(j, job)
            new_ms = compute_makespan(instance, perm)
            if new_ms < best_ms:
                best_ms = new_ms
                best_pos = j
            perm.pop(j)

        perm.insert(best_pos, job)
        ms = best_ms

    return perm, ms


def _path_relink(
    instance: FlowShopInstance,
    source: list[int],
    target: list[int],
) -> tuple[list[int], int]:
    """Path relinking from source to target using insertion moves.

    At each step, selects the insertion move that moves the source
    permutation closer to the target while minimizing makespan.

    Args:
        instance: Flow shop instance.
        source: Source permutation.
        target: Target permutation.

    Returns:
        Best permutation found along the path and its makespan.
    """
    current = list(source)
    best_perm = list(source)
    best_ms = compute_makespan(instance, source)
    n = len(current)

    # Target positions
    target_pos = [0] * n
    for i, job in enumerate(target):
        target_pos[job] = i

    for _ in range(n):
        # Find jobs not in their target position
        current_pos = {job: i for i, job in enumerate(current)}
        candidates = []
        for job in range(n):
            if current_pos[job] != target_pos[job]:
                candidates.append(job)

        if not candidates:
            break

        # Try moving each candidate to its target position
        move_best_perm = None
        move_best_ms = float("inf")

        for job in candidates:
            trial = list(current)
            trial.remove(job)
            insert_pos = min(target_pos[job], len(trial))
            trial.insert(insert_pos, job)
            ms = compute_makespan(instance, trial)

            if ms < move_best_ms:
                move_best_ms = ms
                move_best_perm = trial

        current = move_best_perm
        if move_best_ms < best_ms:
            best_ms = move_best_ms
            best_perm = list(move_best_perm)

    return best_perm, best_ms


def scatter_search(
    instance: FlowShopInstance,
    refset_size: int = 10,
    pop_size: int = 30,
    max_iterations: int = 100,
    time_limit: float | None = None,
    seed: int | None = None,
) -> FlowShopSolution:
    """Solve PFSP using Scatter Search with path relinking.

    Args:
        instance: Flow shop instance.
        refset_size: Size of the reference set.
        pop_size: Size of the initial diverse population.
        max_iterations: Maximum number of RefSet update iterations.
        time_limit: Time limit in seconds (overrides max_iterations).
        seed: Random seed for reproducibility.

    Returns:
        Best FlowShopSolution found.
    """
    rng = np.random.default_rng(seed)
    n = instance.n
    start_time = time.time()

    # ── Diversification: Generate initial population ─────────────────────
    neh_sol = neh(instance)
    population = [(list(neh_sol.permutation), neh_sol.makespan)]

    for _ in range(pop_size - 1):
        perm = list(range(n))
        rng.shuffle(perm)
        ms = compute_makespan(instance, perm)
        population.append((perm, ms))

    # Improve top half with local search
    population.sort(key=lambda x: x[1])
    for i in range(min(refset_size, len(population))):
        perm, ms = _insertion_improve(instance, population[i][0])
        population[i] = (perm, ms)

    # ── Build Reference Set ──────────────────────────────────────────────
    population.sort(key=lambda x: x[1])
    b1 = refset_size // 2  # quality solutions
    b2 = refset_size - b1   # diverse solutions

    refset = list(population[:b1])

    # Add b2 most diverse solutions
    remaining = population[b1:]
    for _ in range(b2):
        if not remaining:
            break
        best_div_idx = 0
        best_div = -1
        for j, (perm, _) in enumerate(remaining):
            min_dist = min(_distance(perm, r[0]) for r in refset)
            if min_dist > best_div:
                best_div = min_dist
                best_div_idx = j
        refset.append(remaining.pop(best_div_idx))

    gbest_perm = refset[0][0]
    gbest_ms = refset[0][1]

    # ── Main loop ────────────────────────────────────────────────────────
    for iteration in range(max_iterations):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        new_solutions = []

        # Generate all pairs from RefSet
        for i in range(len(refset)):
            for j in range(i + 1, len(refset)):
                if time_limit is not None and time.time() - start_time >= time_limit:
                    break

                # Path relinking (forward and backward)
                perm_fwd, ms_fwd = _path_relink(
                    instance, refset[i][0], refset[j][0],
                )
                perm_bwd, ms_bwd = _path_relink(
                    instance, refset[j][0], refset[i][0],
                )

                # Take the better one and improve it
                if ms_fwd <= ms_bwd:
                    perm, ms = _insertion_improve(instance, perm_fwd)
                else:
                    perm, ms = _insertion_improve(instance, perm_bwd)

                new_solutions.append((perm, ms))

                if ms < gbest_ms:
                    gbest_ms = ms
                    gbest_perm = list(perm)

        # Update RefSet with new solutions
        if not new_solutions:
            break

        updated = False
        for perm, ms in new_solutions:
            worst_idx = max(range(len(refset)), key=lambda k: refset[k][1])
            if ms < refset[worst_idx][1]:
                # Check it's not a duplicate
                is_dup = any(perm == r[0] for r in refset)
                if not is_dup:
                    refset[worst_idx] = (perm, ms)
                    updated = True

        if not updated:
            break

    return FlowShopSolution(
        permutation=gbest_perm,
        makespan=gbest_ms,
    )


if __name__ == "__main__":
    instance = FlowShopInstance.random(n=20, m=5, seed=42)

    neh_sol = neh(instance)
    print(f"NEH: makespan = {neh_sol.makespan}")

    ss_sol = scatter_search(instance, max_iterations=50, seed=42)
    print(f"SS:  makespan = {ss_sol.makespan}")
    print(f"Improvement: {(neh_sol.makespan - ss_sol.makespan) / neh_sol.makespan * 100:.1f}%")
