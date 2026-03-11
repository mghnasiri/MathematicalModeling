"""
Memetic Algorithm (MA) for Fm | prmu | Cmax.

A Memetic Algorithm is a hybrid evolutionary approach that combines a
population-based Genetic Algorithm with individual-level local search
(Moscato, 1989). The term "memetic" refers to cultural evolution through
"memes" — ideas that improve through local refinement, in contrast to
purely genetic (random) variation.

For PFSP, the MA augments a standard GA with insertion-based local search
applied to each offspring. This significantly improves convergence compared
to a plain GA, at the cost of increased computation per generation.

Algorithm:
    1. Initialize population: NEH + perturbations + random.
    2. For each generation:
       a. Select parents via binary tournament.
       b. Apply OX crossover to produce offspring.
       c. Apply insertion mutation with probability p_mut.
       d. LOCAL SEARCH: improve each offspring via first-improvement
          insertion local search (the "memetic" step).
       e. Replace worst members of population with improved offspring.
    3. Track and return global best.

Notation: Fm | prmu | Cmax
Complexity: O(generations * pop_size * n^2 * m) per generation
            (dominated by the local search step).

Reference:
    Moscato, P. (1989). On evolution, search, optimization, genetic
    algorithms and martial arts: towards memetic algorithms. Technical
    Report C3P 826, Caltech.

    Chen, C.L., Vempati, V.S. & Aljaber, N. (1995). An application of
    genetic algorithms for flow shop problems. European Journal of
    Operational Research, 80(2), 389-396.
    https://doi.org/10.1016/0377-2217(93)E0350-7
"""

from __future__ import annotations

import sys
import os
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from instance import FlowShopInstance, FlowShopSolution, compute_makespan
from heuristics.neh import neh


def _ox_crossover(
    parent1: list[int],
    parent2: list[int],
    rng: np.random.Generator,
) -> list[int]:
    """Order Crossover (OX)."""
    n = len(parent1)
    if n < 3:
        return list(parent1) if rng.random() < 0.5 else list(parent2)

    i, j = sorted(rng.choice(n, size=2, replace=False))
    segment = parent1[i: j + 1]
    seg_set = set(segment)

    remaining = [g for g in parent2 if g not in seg_set]
    child = remaining[: i] + segment + remaining[i:]
    return child


def _insertion_ls(
    instance: FlowShopInstance,
    perm: list[int],
    makespan: int,
    max_no_improve: int = 0,
) -> tuple[list[int], int]:
    """First-improvement insertion local search (limited passes).

    Args:
        instance: Flow shop instance.
        perm: Current permutation.
        makespan: Current makespan.
        max_no_improve: Max passes without improvement (0 = single pass).

    Returns:
        Improved permutation and makespan.
    """
    n = len(perm)
    current = list(perm)
    current_ms = makespan

    no_improve = 0
    while no_improve <= max_no_improve:
        improved = False
        for i in range(n):
            job = current[i]
            remaining = current[:i] + current[i + 1:]
            best_ms = current_ms
            best_pos = i

            for j in range(n):
                if j == i:
                    continue
                candidate = remaining[:j] + [job] + remaining[j:]
                ms = compute_makespan(instance, candidate)
                if ms < best_ms:
                    best_ms = ms
                    best_pos = j
                    break  # first improvement

            if best_ms < current_ms:
                current = remaining[:best_pos] + [job] + remaining[best_pos:]
                current_ms = best_ms
                improved = True
                break

        if not improved:
            no_improve += 1
        else:
            no_improve = 0

    return current, current_ms


def memetic_algorithm(
    instance: FlowShopInstance,
    population_size: int = 20,
    generations: int = 50,
    crossover_rate: float = 0.9,
    mutation_rate: float = 0.2,
    ls_passes: int = 1,
    time_limit: float | None = None,
    seed: int | None = None,
) -> FlowShopSolution:
    """Solve PFSP using a Memetic Algorithm.

    Args:
        instance: Flow shop instance.
        population_size: Number of individuals.
        generations: Maximum number of generations.
        crossover_rate: Probability of applying crossover.
        mutation_rate: Probability of mutation per offspring.
        ls_passes: Number of local search passes per offspring.
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
                idx = rng.integers(0, n)
                job = perm.pop(idx)
                pos = rng.integers(0, n)
                if pos >= idx:
                    pos = max(0, pos - 1)
                perm.insert(pos, job)
        else:
            perm = list(range(n))
            rng.shuffle(perm)
        ms = compute_makespan(instance, perm)
        population.append(perm)
        fitnesses.append(ms)

    gbest_idx = int(np.argmin(fitnesses))
    gbest_perm = list(population[gbest_idx])
    gbest_ms = fitnesses[gbest_idx]

    for gen in range(generations):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        offspring_pool: list[tuple[list[int], int]] = []

        # Generate offspring
        n_offspring = pop_size // 2
        for _ in range(n_offspring):
            # Binary tournament selection
            def tournament():
                a, b = rng.choice(pop_size, size=2, replace=False)
                return a if fitnesses[a] <= fitnesses[b] else b

            p1 = tournament()
            p2 = tournament()

            # Crossover
            if rng.random() < crossover_rate:
                child = _ox_crossover(population[p1], population[p2], rng)
            else:
                child = list(population[p1])

            # Mutation (insertion move)
            if rng.random() < mutation_rate and n >= 2:
                i = rng.integers(0, n)
                job = child.pop(i)
                j = rng.integers(0, n)
                if j >= i:
                    j = max(0, j - 1)
                child.insert(j, job)

            child_ms = compute_makespan(instance, child)

            # Local search (the memetic step)
            child, child_ms = _insertion_ls(
                instance, child, child_ms, max_no_improve=ls_passes - 1,
            )

            offspring_pool.append((child, child_ms))

        # Replace worst members with offspring (if better)
        for child, child_ms in offspring_pool:
            worst_idx = int(np.argmax(fitnesses))
            if child_ms < fitnesses[worst_idx]:
                population[worst_idx] = child
                fitnesses[worst_idx] = child_ms

                if child_ms < gbest_ms:
                    gbest_ms = child_ms
                    gbest_perm = list(child)

    return FlowShopSolution(permutation=gbest_perm, makespan=gbest_ms)


if __name__ == "__main__":
    instance = FlowShopInstance.random(n=20, m=5, seed=42)

    neh_sol = neh(instance)
    print(f"NEH: makespan = {neh_sol.makespan}")

    ma_sol = memetic_algorithm(instance, generations=30, seed=42)
    print(f"MA:  makespan = {ma_sol.makespan}")
    print(
        f"Improvement: "
        f"{(neh_sol.makespan - ma_sol.makespan) / neh_sol.makespan * 100:.1f}%"
    )
