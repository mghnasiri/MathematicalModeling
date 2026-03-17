"""
Biogeography-Based Optimization (BBO) for Fm | prmu | Cmax.

BBO models solutions as island habitats with a Habitat Suitability Index
(HSI) (Simon, 2008). High-HSI habitats (good solutions) have high emigration
rates (they share features) and low immigration rates, while low-HSI habitats
receive features from good solutions.

For the discrete PFSP, migration transfers job subsequences between
permutations. Mutation provides random exploration.

Algorithm:
    1. Initialize population of habitats (NEH + random).
    2. Compute immigration (lambda) and emigration (mu) rates per habitat
       based on fitness ranking.
    3. Migration: for each habitat, with prob lambda_i, select an
       emigrating habitat proportional to mu, then apply a crossover
       (position-based inheritance) to transfer a subsequence.
    4. Mutation: with small probability, apply random insert/swap.
    5. Greedy selection: keep the better of old and new habitat.
    6. Track global best and repeat.

Notation: Fm | prmu | Cmax
Complexity: O(iterations * pop_size * n * m) per iteration.

Reference:
    Simon, D. (2008). Biogeography-based optimization. IEEE Transactions
    on Evolutionary Computation, 12(6), 702-713.
    https://doi.org/10.1109/TEVC.2008.919004

    Yin, M. & Li, X. (2011). A hybrid bio-geography based optimization
    for permutation flow shop scheduling. Scientific Research Essays,
    6(13), 2078-2090.
"""

from __future__ import annotations

import sys
import os
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from instance import FlowShopInstance, FlowShopSolution, compute_makespan
from heuristics.neh import neh


def _insert_move(perm: list[int], rng: np.random.Generator) -> list[int]:
    """Apply a random insertion move."""
    n = len(perm)
    if n < 2:
        return list(perm)
    result = list(perm)
    i = rng.integers(0, n)
    job = result.pop(i)
    j = rng.integers(0, n)
    if j >= i:
        j = max(0, j - 1)
    result.insert(j, job)
    return result


def _migrate(
    target: list[int],
    source: list[int],
    rng: np.random.Generator,
) -> list[int]:
    """Migration operator: transfer a subsequence from source to target.

    Selects a random segment from the source permutation and applies
    position-based crossover to create a new permutation.

    Args:
        target: Receiving habitat (immigration).
        source: Donating habitat (emigration).
        rng: Random number generator.

    Returns:
        New permutation combining features of target and source.
    """
    n = len(target)
    if n < 3:
        return list(source) if rng.random() < 0.5 else list(target)

    # Select a random segment from source
    seg_len = rng.integers(1, max(2, n // 3))
    start = rng.integers(0, n - seg_len + 1)
    segment = source[start: start + seg_len]
    segment_set = set(segment)

    # Build result: place segment at the same position,
    # fill remaining positions with target's order
    remaining = [j for j in target if j not in segment_set]

    result = []
    rem_idx = 0
    for pos in range(n):
        if start <= pos < start + seg_len:
            result.append(segment[pos - start])
        else:
            result.append(remaining[rem_idx])
            rem_idx += 1

    return result


def bbo(
    instance: FlowShopInstance,
    population_size: int = 20,
    mutation_rate: float = 0.05,
    max_iterations: int = 500,
    time_limit: float | None = None,
    seed: int | None = None,
) -> FlowShopSolution:
    """Solve PFSP using Biogeography-Based Optimization.

    Args:
        instance: Flow shop instance.
        population_size: Number of habitats.
        mutation_rate: Probability of random mutation per habitat.
        max_iterations: Maximum iterations.
        time_limit: Time limit in seconds.
        seed: Random seed for reproducibility.

    Returns:
        Best FlowShopSolution found.
    """
    rng = np.random.default_rng(seed)
    n = instance.n
    pop_size = population_size
    start_time = time.time()

    # ── Initialize population ────────────────────────────────────────────
    neh_sol = neh(instance)
    population: list[list[int]] = [list(neh_sol.permutation)]
    fitnesses: list[int] = [neh_sol.makespan]

    for i in range(1, pop_size):
        if i < pop_size // 3:
            perm = list(neh_sol.permutation)
            for _ in range(rng.integers(1, max(2, n // 3))):
                perm = _insert_move(perm, rng)
        else:
            perm = list(range(n))
            rng.shuffle(perm)
        ms = compute_makespan(instance, perm)
        population.append(perm)
        fitnesses.append(ms)

    gbest_idx = int(np.argmin(fitnesses))
    gbest_perm = list(population[gbest_idx])
    gbest_ms = fitnesses[gbest_idx]

    for iteration in range(max_iterations):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        # ── Compute immigration/emigration rates ─────────────────────────
        # Sort habitats by fitness (best=highest rank)
        ranked = sorted(range(pop_size), key=lambda i: fitnesses[i])
        rank = [0] * pop_size
        for r, idx in enumerate(ranked):
            rank[idx] = r  # 0=best, pop_size-1=worst

        # Immigration rate lambda_i: high for bad habitats (high rank)
        # Emigration rate mu_i: high for good habitats (low rank)
        immigration = [r / (pop_size - 1) if pop_size > 1 else 0.5
                       for r in rank]
        emigration = [1.0 - lam for lam in immigration]

        # ── Migration phase ──────────────────────────────────────────────
        new_population = [list(p) for p in population]
        new_fitnesses = list(fitnesses)

        for i in range(pop_size):
            if rng.random() < immigration[i]:
                # Select emigrating habitat proportional to emigration rate
                emu_total = sum(emigration[j] for j in range(pop_size)
                                if j != i)
                if emu_total <= 0:
                    continue
                probs = []
                indices = []
                for j in range(pop_size):
                    if j != i:
                        probs.append(emigration[j] / emu_total)
                        indices.append(j)

                source_idx = rng.choice(indices, p=probs)
                new_perm = _migrate(population[i], population[source_idx], rng)
                new_ms = compute_makespan(instance, new_perm)

                # Greedy selection
                if new_ms <= fitnesses[i]:
                    new_population[i] = new_perm
                    new_fitnesses[i] = new_ms

        # ── Mutation phase ───────────────────────────────────────────────
        for i in range(pop_size):
            if rng.random() < mutation_rate:
                mutant = _insert_move(new_population[i], rng)
                mut_ms = compute_makespan(instance, mutant)
                if mut_ms <= new_fitnesses[i]:
                    new_population[i] = mutant
                    new_fitnesses[i] = mut_ms

        population = new_population
        fitnesses = new_fitnesses

        # Update global best
        for i in range(pop_size):
            if fitnesses[i] < gbest_ms:
                gbest_ms = fitnesses[i]
                gbest_perm = list(population[i])

    return FlowShopSolution(
        permutation=gbest_perm,
        makespan=gbest_ms,
    )


if __name__ == "__main__":
    instance = FlowShopInstance.random(n=20, m=5, seed=42)

    neh_sol = neh(instance)
    print(f"NEH: makespan = {neh_sol.makespan}")

    bbo_sol = bbo(instance, max_iterations=500, seed=42)
    print(f"BBO: makespan = {bbo_sol.makespan}")
    print(
        f"Improvement: "
        f"{(neh_sol.makespan - bbo_sol.makespan) / neh_sol.makespan * 100:.1f}%"
    )
