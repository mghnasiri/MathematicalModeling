"""
Teaching-Learning-Based Optimization (TLBO) for Fm | prmu | Cmax.

TLBO mimics the teaching-learning process in a classroom (Rao et al., 2011).
A key advantage is that it has NO algorithm-specific parameters to tune
(no mutation rate, crossover rate, pheromone decay, etc.) — only population
size and termination criteria.

Two phases per iteration:
1. Teacher phase: the best solution (teacher) influences all learners.
   Each learner moves toward the teacher via insert moves derived from
   the difference between teacher and learner.
2. Learner phase: random pairs interact; the worse learner moves toward
   the better one via insert moves.

For the discrete PFSP, "moving toward" a permutation means applying
insert moves that reduce the Kendall tau distance (number of pairwise
inversions) to the target permutation.

Algorithm:
    1. Initialize population: NEH solution + random perturbations.
    2. Teacher phase: for each learner, apply d insert moves toward
       the teacher, where d is proportional to distance.
    3. Learner phase: pair each learner with a random other; worse
       learner applies insert moves toward the better one.
    4. Greedy selection: keep the better of old and new for each learner.
    5. Update teacher (global best) and repeat.

Notation: Fm | prmu | Cmax
Complexity: O(iterations * pop_size * n * m) per iteration.

Reference:
    Rao, R.V., Savsani, V.J. & Vakharia, D.P. (2011). Teaching-learning-
    based optimization: a novel method for constrained mechanical design
    optimization problems. Computer-Aided Design, 43(3), 303-315.
    https://doi.org/10.1016/j.cad.2010.12.015

    Xie, Z., Zhang, C., Shao, X., Lin, W. & Zhu, H. (2014). An effective
    hybrid teaching-learning-based optimization algorithm for permutation
    flow shop scheduling problem. Advances in Mechanical Engineering, 6,
    592125. https://doi.org/10.1155/2014/592125
"""

from __future__ import annotations

import sys
import os
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from instance import FlowShopInstance, FlowShopSolution, compute_makespan
from heuristics.neh import neh


def _insert_toward(
    perm: list[int],
    target: list[int],
    n_moves: int,
    rng: np.random.Generator,
) -> list[int]:
    """Apply insert moves that move perm toward target permutation.

    Identifies positions where perm and target differ, then applies
    insert moves to reduce the positional distance.

    Args:
        perm: Current permutation.
        target: Target permutation to move toward.
        n_moves: Number of insert moves to apply.
        rng: Random number generator.

    Returns:
        New permutation closer to target.
    """
    result = list(perm)
    n = len(result)

    for _ in range(n_moves):
        # Find positions where result differs from target
        diffs = []
        target_pos = {job: i for i, job in enumerate(target)}
        for i, job in enumerate(result):
            if target_pos[job] != i:
                diffs.append(i)

        if not diffs:
            break  # Already identical

        # Pick a differing position, remove its job, insert at target position
        idx = rng.choice(len(diffs))
        pos = diffs[idx]
        job = result[pos]
        desired_pos = target_pos[job]

        result.pop(pos)
        # Adjust desired position after removal
        if desired_pos > pos:
            desired_pos -= 1
        desired_pos = min(desired_pos, len(result))
        result.insert(desired_pos, job)

    return result


def _random_insert(
    perm: list[int],
    rng: np.random.Generator,
) -> list[int]:
    """Apply a single random insertion move."""
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


def _positional_distance(a: list[int], b: list[int]) -> int:
    """Count the number of positions where a and b differ."""
    return sum(1 for x, y in zip(a, b) if x != y)


def tlbo(
    instance: FlowShopInstance,
    population_size: int = 20,
    max_iterations: int = 500,
    time_limit: float | None = None,
    seed: int | None = None,
) -> FlowShopSolution:
    """Solve PFSP using Teaching-Learning-Based Optimization.

    TLBO is parameter-free beyond population size and termination criteria.

    Args:
        instance: Flow shop instance.
        population_size: Number of learners in the classroom.
        max_iterations: Maximum number of iterations.
        time_limit: Time limit in seconds (overrides max_iterations).
        seed: Random seed for reproducibility.

    Returns:
        Best FlowShopSolution found.
    """
    rng = np.random.default_rng(seed)
    n = instance.n
    start_time = time.time()

    # ── Initialize population ────────────────────────────────────────────
    neh_sol = neh(instance)
    population: list[list[int]] = [list(neh_sol.permutation)]
    fitnesses: list[int] = [neh_sol.makespan]

    for i in range(1, population_size):
        if i < population_size // 3:
            # Perturbation of NEH
            perm = list(neh_sol.permutation)
            for _ in range(rng.integers(1, max(2, n // 3))):
                perm = _random_insert(perm, rng)
        else:
            # Random permutation
            perm = list(range(n))
            rng.shuffle(perm)
        ms = compute_makespan(instance, perm)
        population.append(perm)
        fitnesses.append(ms)

    # Track global best
    best_idx = int(np.argmin(fitnesses))
    gbest_perm = list(population[best_idx])
    gbest_ms = fitnesses[best_idx]

    for iteration in range(max_iterations):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        # ── Teacher phase ────────────────────────────────────────────────
        teacher_idx = int(np.argmin(fitnesses))
        teacher = population[teacher_idx]

        for i in range(population_size):
            if i == teacher_idx:
                # Teacher does random exploration
                new_perm = _random_insert(population[i], rng)
            else:
                # Move toward teacher: number of moves proportional to distance
                dist = _positional_distance(population[i], teacher)
                n_moves = max(1, dist // 4)
                # Teaching factor: randomly 1 or 2
                tf = int(rng.integers(1, 3))
                n_moves = max(1, n_moves // tf)
                new_perm = _insert_toward(population[i], teacher, n_moves, rng)

            new_ms = compute_makespan(instance, new_perm)

            # Greedy selection
            if new_ms <= fitnesses[i]:
                population[i] = new_perm
                fitnesses[i] = new_ms
                if new_ms < gbest_ms:
                    gbest_ms = new_ms
                    gbest_perm = list(new_perm)

        # ── Learner phase ────────────────────────────────────────────────
        for i in range(population_size):
            # Pick a random partner
            partner = rng.integers(0, population_size)
            while partner == i:
                partner = rng.integers(0, population_size)

            if fitnesses[i] <= fitnesses[partner]:
                # i is better: partner moves toward i (handled in partner's turn)
                # i does random exploration
                new_perm = _random_insert(population[i], rng)
            else:
                # partner is better: i moves toward partner
                dist = _positional_distance(population[i], population[partner])
                n_moves = max(1, dist // 4)
                new_perm = _insert_toward(
                    population[i], population[partner], n_moves, rng,
                )

            new_ms = compute_makespan(instance, new_perm)

            # Greedy selection
            if new_ms <= fitnesses[i]:
                population[i] = new_perm
                fitnesses[i] = new_ms
                if new_ms < gbest_ms:
                    gbest_ms = new_ms
                    gbest_perm = list(new_perm)

    return FlowShopSolution(
        permutation=gbest_perm,
        makespan=gbest_ms,
    )


if __name__ == "__main__":
    instance = FlowShopInstance.random(n=20, m=5, seed=42)

    neh_sol = neh(instance)
    print(f"NEH:  makespan = {neh_sol.makespan}")

    tlbo_sol = tlbo(instance, max_iterations=300, seed=42)
    print(f"TLBO: makespan = {tlbo_sol.makespan}")
    print(
        f"Improvement: "
        f"{(neh_sol.makespan - tlbo_sol.makespan) / neh_sol.makespan * 100:.1f}%"
    )
