"""
Differential Evolution (DE) for Fm | prmu | Cmax.

Differential Evolution is an evolutionary algorithm that uses vector
differences for mutation, originally designed for continuous optimization
(Storn & Price, 1997). For the discrete PFSP, a position-based encoding
is used where continuous vectors are decoded to permutations via the
Largest Rank Value (LRV) rule (Onwubolu & Davendra, 2006).

Algorithm (DE/rand/1/bin variant):
    1. Initialize population of NP real-valued vectors of dimension n.
    2. For each target vector x_i:
       a. Select three distinct random vectors x_r1, x_r2, x_r3.
       b. Mutant vector: v = x_r1 + F * (x_r2 - x_r3).
       c. Crossover: trial u_j = v_j if rand < CR else x_i_j.
       d. Decode both target and trial to permutations.
       e. Select the one with better makespan.
    3. Repeat until termination.

Complexity: O(generations * NP * n * m) for evaluation.

Reference:
    Onwubolu, G. & Davendra, D. (2006). Scheduling flow shops using
    differential evolution algorithm. European Journal of Operational
    Research, 171(2), 674-692.
    https://doi.org/10.1016/j.ejor.2004.08.043

    Storn, R. & Price, K. (1997). Differential evolution — a simple and
    efficient heuristic for global optimization over continuous spaces.
    Journal of Global Optimization, 11(4), 341-359.
    https://doi.org/10.1023/A:1008202821328

    Pan, Q.K., Tasgetiren, M.F. & Liang, Y.C. (2008). A discrete
    differential evolution algorithm for the permutation flowshop
    scheduling problem. Computers & Industrial Engineering, 55(4),
    795-816. https://doi.org/10.1016/j.cie.2008.03.003
"""

from __future__ import annotations

import sys
import os
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from instance import FlowShopInstance, FlowShopSolution, compute_makespan
from heuristics.neh import neh


def _decode_lrv(vector: np.ndarray) -> list[int]:
    """Decode continuous vector to permutation using Largest Rank Value.

    Args:
        vector: Continuous vector of length n.

    Returns:
        Permutation of job indices.
    """
    return list(np.argsort(-vector))


def _referenced_insertion_ls(
    instance: FlowShopInstance,
    permutation: list[int],
    rng: np.random.Generator,
) -> tuple[list[int], int]:
    """Light random insertion local search.

    Args:
        instance: Flow shop instance.
        permutation: Current permutation.
        rng: Random number generator.

    Returns:
        Tuple of (improved permutation, makespan).
    """
    perm = list(permutation)
    ms = compute_makespan(instance, perm)
    n = len(perm)

    for _ in range(n):
        i = rng.integers(0, n)
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


def differential_evolution(
    instance: FlowShopInstance,
    population_size: int = 30,
    max_iterations: int = 500,
    F: float = 0.5,
    CR: float = 0.9,
    local_search: bool = True,
    time_limit: float | None = None,
    seed: int | None = None,
) -> FlowShopSolution:
    """Solve PFSP using Differential Evolution with LRV encoding.

    Args:
        instance: Flow shop instance.
        population_size: Population size (NP).
        max_iterations: Maximum number of generations.
        F: Mutation scaling factor (typically 0.4-1.0).
        CR: Crossover rate (typically 0.7-1.0).
        local_search: Apply insertion LS to best individual periodically.
        time_limit: Time limit in seconds (overrides max_iterations).
        seed: Random seed for reproducibility.

    Returns:
        Best FlowShopSolution found.
    """
    rng = np.random.default_rng(seed)
    n = instance.n
    NP = population_size
    start_time = time.time()

    # Initialize population
    population = rng.uniform(-4, 4, size=(NP, n))
    fitness = np.zeros(NP, dtype=int)

    # Seed one individual with NEH
    neh_sol = neh(instance)
    neh_rank = np.zeros(n)
    for rank, job in enumerate(neh_sol.permutation):
        neh_rank[job] = -rank  # negative for LRV (descending order)
    population[0] = neh_rank + rng.uniform(-0.05, 0.05, size=n)

    # Evaluate initial population
    for i in range(NP):
        perm = _decode_lrv(population[i])
        fitness[i] = compute_makespan(instance, perm)

    best_idx = np.argmin(fitness)
    gbest_perm = _decode_lrv(population[best_idx])
    gbest_ms = fitness[best_idx]

    if neh_sol.makespan < gbest_ms:
        gbest_perm = list(neh_sol.permutation)
        gbest_ms = neh_sol.makespan

    for gen in range(max_iterations):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        for i in range(NP):
            # Select three distinct random indices != i
            candidates = [j for j in range(NP) if j != i]
            r1, r2, r3 = rng.choice(candidates, size=3, replace=False)

            # Mutation: DE/rand/1
            mutant = population[r1] + F * (population[r2] - population[r3])

            # Binomial crossover
            trial = population[i].copy()
            j_rand = rng.integers(0, n)
            for j in range(n):
                if rng.random() < CR or j == j_rand:
                    trial[j] = mutant[j]

            # Selection
            trial_perm = _decode_lrv(trial)
            trial_ms = compute_makespan(instance, trial_perm)

            if trial_ms <= fitness[i]:
                population[i] = trial
                fitness[i] = trial_ms

                if trial_ms < gbest_ms:
                    gbest_ms = trial_ms
                    gbest_perm = list(trial_perm)

        # Optional local search on best
        if local_search and gen % 20 == 0 and gen > 0:
            ls_perm, ls_ms = _referenced_insertion_ls(
                instance, gbest_perm, rng,
            )
            if ls_ms < gbest_ms:
                gbest_ms = ls_ms
                gbest_perm = list(ls_perm)

    return FlowShopSolution(
        permutation=gbest_perm,
        makespan=gbest_ms,
    )


if __name__ == "__main__":
    instance = FlowShopInstance.random(n=20, m=5, seed=42)

    neh_sol = neh(instance)
    print(f"NEH: makespan = {neh_sol.makespan}")

    de_sol = differential_evolution(instance, max_iterations=200, seed=42)
    print(f"DE:  makespan = {de_sol.makespan}")
    print(f"Improvement: {(neh_sol.makespan - de_sol.makespan) / neh_sol.makespan * 100:.1f}%")
