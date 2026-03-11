"""
Estimation of Distribution Algorithm (EDA) for Fm | prmu | Cmax.

EDAs replace crossover/mutation operators with probabilistic models.
Instead of recombining parent solutions directly, they build a
probability distribution from the best solutions and sample new
solutions from it (Larrañaga & Lozano, 2001).

For permutation problems, the probability model is a position-based
matrix P[i][j] = probability that job j is placed at position i.
The model is learned from the top-k solutions in the population,
and new solutions are sampled by constructing permutations from
the probability matrix using a stochastic roulette-wheel approach.

Algorithm:
    1. Initialize population (NEH + random permutations).
    2. Select top-k solutions (truncation selection).
    3. Build probability matrix from selected solutions:
       P[i][j] = (count of job j at position i in selected) / k
    4. Sample new solutions from P using roulette-wheel construction:
       For each position, select an unplaced job with probability
       proportional to P[pos][job].
    5. Replace worst solutions with new samples.
    6. Apply optional local search to best new solution.
    7. Repeat.

Notation: Fm | prmu | Cmax
Complexity: O(generations * pop_size * n^2) for model + sampling,
            plus O(n^2 * m) per makespan evaluation.

Reference:
    Larrañaga, P. & Lozano, J.A. (2001). Estimation of Distribution
    Algorithms: A New Tool for Evolutionary Computation. Springer.
    https://doi.org/10.1007/978-1-4615-1539-5

    Jarboui, B., Eddaly, M. & Siarry, P. (2009). An estimation of
    distribution algorithm for minimizing the total weighted tardiness
    in permutation flowshop scheduling problems. Computers & Operations
    Research, 36(8), 2463-2475.
    https://doi.org/10.1016/j.cor.2008.09.012
"""

from __future__ import annotations

import sys
import os
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from instance import FlowShopInstance, FlowShopSolution, compute_makespan
from heuristics.neh import neh


def _sample_permutation(
    prob_matrix: np.ndarray,
    rng: np.random.Generator,
) -> list[int]:
    """Sample a permutation from a position-based probability matrix.

    Args:
        prob_matrix: n x n matrix where P[i][j] = probability of job j
                     at position i.
        rng: Random number generator.

    Returns:
        A valid permutation of n jobs.
    """
    n = prob_matrix.shape[0]
    perm: list[int] = []
    available = set(range(n))

    for pos in range(n):
        if len(available) == 1:
            perm.append(available.pop())
            break

        # Get probabilities for available jobs at this position
        avail_list = sorted(available)
        probs = np.array([prob_matrix[pos, j] for j in avail_list])

        # Ensure no zero probabilities (smoothing)
        probs = probs + 1e-8
        probs = probs / probs.sum()

        chosen = rng.choice(avail_list, p=probs)
        perm.append(int(chosen))
        available.remove(int(chosen))

    return perm


def eda(
    instance: FlowShopInstance,
    population_size: int = 30,
    selection_ratio: float = 0.4,
    learning_rate: float = 0.5,
    generations: int = 200,
    time_limit: float | None = None,
    seed: int | None = None,
) -> FlowShopSolution:
    """Solve PFSP using Estimation of Distribution Algorithm.

    Args:
        instance: Flow shop instance.
        population_size: Number of solutions in population.
        selection_ratio: Fraction of population used to build model.
        learning_rate: Blending rate between old and new model (0-1).
            0 = keep old model, 1 = fully replace with new.
        generations: Maximum number of generations.
        time_limit: Time limit in seconds.
        seed: Random seed for reproducibility.

    Returns:
        Best FlowShopSolution found.
    """
    rng = np.random.default_rng(seed)
    n = instance.n
    pop_size = population_size
    n_select = max(2, int(pop_size * selection_ratio))
    start_time = time.time()

    # ── Initialize population ────────────────────────────────────────────
    neh_sol = neh(instance)
    population: list[list[int]] = [list(neh_sol.permutation)]
    fitnesses: list[int] = [neh_sol.makespan]

    for i in range(1, pop_size):
        if i < pop_size // 4:
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

    # Initialize probability matrix (uniform)
    prob_matrix = np.ones((n, n)) / n

    for gen in range(generations):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        # ── Selection: top-k truncation ──────────────────────────────────
        sorted_indices = sorted(range(pop_size), key=lambda i: fitnesses[i])
        selected = [population[i] for i in sorted_indices[:n_select]]

        # ── Build new probability model ──────────────────────────────────
        new_model = np.zeros((n, n))
        for perm in selected:
            for pos, job in enumerate(perm):
                new_model[pos, job] += 1.0
        new_model /= n_select

        # Blend with previous model
        prob_matrix = (1.0 - learning_rate) * prob_matrix + learning_rate * new_model

        # ── Sample new solutions ─────────────────────────────────────────
        n_new = pop_size - n_select
        new_solutions: list[tuple[list[int], int]] = []

        for _ in range(n_new):
            new_perm = _sample_permutation(prob_matrix, rng)
            new_ms = compute_makespan(instance, new_perm)
            new_solutions.append((new_perm, new_ms))

        # Replace worst solutions
        worst_indices = sorted_indices[n_select:]
        for idx, (new_perm, new_ms) in zip(worst_indices, new_solutions):
            population[idx] = new_perm
            fitnesses[idx] = new_ms

            if new_ms < gbest_ms:
                gbest_ms = new_ms
                gbest_perm = list(new_perm)

        # Also check if selected solutions include a new best
        for i in sorted_indices[:n_select]:
            if fitnesses[i] < gbest_ms:
                gbest_ms = fitnesses[i]
                gbest_perm = list(population[i])

    return FlowShopSolution(permutation=gbest_perm, makespan=gbest_ms)


if __name__ == "__main__":
    instance = FlowShopInstance.random(n=20, m=5, seed=42)

    neh_sol = neh(instance)
    print(f"NEH: makespan = {neh_sol.makespan}")

    eda_sol = eda(instance, generations=200, seed=42)
    print(f"EDA: makespan = {eda_sol.makespan}")
    print(
        f"Improvement: "
        f"{(neh_sol.makespan - eda_sol.makespan) / neh_sol.makespan * 100:.1f}%"
    )
