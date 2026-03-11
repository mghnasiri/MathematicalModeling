"""
Whale Optimization Algorithm (WOA) for Fm | prmu | Cmax.

WOA simulates the bubble-net hunting strategy of humpback whales
(Mirjalili & Lewis, 2016). Three operators:

1. Encircling prey (exploitation): whales swim toward the current best
   solution using insertion moves derived from positional differences.
2. Bubble-net attack (exploitation): a spiral movement toward the best
   solution using a mix of insert and swap moves with decreasing intensity.
3. Random search (exploration): a whale moves toward a random other whale
   instead of the best, promoting diversity.

The parameter `a` decreases linearly from 2 to 0 over iterations, shifting
from exploration to exploitation. When |A| >= 1, random search is used;
when |A| < 1, encircling/spiral is used (50/50 probability).

For the discrete PFSP, continuous position vectors are replaced by
permutations, and arithmetic operations are replaced by insert/swap moves.

Notation: Fm | prmu | Cmax
Complexity: O(iterations * pop_size * n * m) per iteration.

Reference:
    Mirjalili, S. & Lewis, A. (2016). The Whale Optimization Algorithm.
    Advances in Engineering Software, 95, 51-67.
    https://doi.org/10.1016/j.advengsoft.2016.01.008

    Abdel-Basset, M., Manogaran, G., El-Shahat, D. & Mirjalili, S. (2018).
    A hybrid whale optimization algorithm based on local search strategy
    for the permutation flow shop scheduling problem. Future Generation
    Computer Systems, 85, 129-145.
    https://doi.org/10.1016/j.future.2018.03.020
"""

from __future__ import annotations

import sys
import os
import math
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
    """Apply insert moves to move perm closer to target.

    Args:
        perm: Current permutation.
        target: Target permutation.
        n_moves: Number of insert moves to apply.
        rng: Random number generator.

    Returns:
        New permutation moved toward target.
    """
    result = list(perm)
    n = len(result)

    for _ in range(n_moves):
        target_pos = {job: i for i, job in enumerate(target)}
        diffs = [i for i, job in enumerate(result) if target_pos[job] != i]

        if not diffs:
            break

        idx = rng.choice(len(diffs))
        pos = diffs[idx]
        job = result[pos]
        desired = target_pos[job]
        result.pop(pos)
        if desired > pos:
            desired -= 1
        desired = min(desired, len(result))
        result.insert(desired, job)

    return result


def _random_insert(perm: list[int], rng: np.random.Generator) -> list[int]:
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


def _spiral_move(
    perm: list[int],
    target: list[int],
    intensity: float,
    rng: np.random.Generator,
) -> list[int]:
    """Spiral movement: mix of targeted inserts and random perturbations.

    Args:
        perm: Current permutation.
        target: Best known permutation.
        intensity: How strongly to move toward target (0 to 1).
        rng: Random number generator.

    Returns:
        New permutation from spiral movement.
    """
    n = len(perm)
    n_targeted = max(1, int(intensity * n // 3))
    n_random = max(1, int((1 - intensity) * n // 4))

    result = _insert_toward(perm, target, n_targeted, rng)
    for _ in range(n_random):
        result = _random_insert(result, rng)

    return result


def _positional_distance(a: list[int], b: list[int]) -> int:
    """Count positions where a and b differ."""
    return sum(1 for x, y in zip(a, b) if x != y)


def whale_optimization(
    instance: FlowShopInstance,
    population_size: int = 20,
    max_iterations: int = 500,
    time_limit: float | None = None,
    seed: int | None = None,
) -> FlowShopSolution:
    """Solve PFSP using the Whale Optimization Algorithm.

    Args:
        instance: Flow shop instance.
        population_size: Number of whales.
        max_iterations: Maximum iterations.
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
            perm = list(neh_sol.permutation)
            for _ in range(rng.integers(1, max(2, n // 3))):
                perm = _random_insert(perm, rng)
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

        # Linearly decrease `a` from 2 to 0
        a = 2.0 * (1.0 - iteration / max_iterations)

        for i in range(population_size):
            # Random coefficients
            r = rng.random()
            A = 2.0 * a * r - a  # A in [-a, a]
            p_switch = rng.random()  # probability for spiral vs encircle

            if abs(A) >= 1.0:
                # ── Random search (exploration) ──────────────────────
                rand_idx = rng.integers(0, population_size)
                target = population[rand_idx]
                dist = _positional_distance(population[i], target)
                n_moves = max(1, int(abs(A) * dist // 4))
                new_perm = _insert_toward(population[i], target, n_moves, rng)
            elif p_switch < 0.5:
                # ── Encircling prey (exploitation) ───────────────────
                dist = _positional_distance(population[i], gbest_perm)
                n_moves = max(1, int((1.0 - abs(A)) * dist // 3))
                new_perm = _insert_toward(
                    population[i], gbest_perm, n_moves, rng,
                )
            else:
                # ── Bubble-net spiral (exploitation) ─────────────────
                l_param = rng.uniform(-1, 1)
                intensity = math.exp(-abs(l_param)) * math.cos(
                    2 * math.pi * l_param
                )
                intensity = max(0.0, min(1.0, (intensity + 1) / 2))
                new_perm = _spiral_move(
                    population[i], gbest_perm, intensity, rng,
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

    woa_sol = whale_optimization(instance, max_iterations=300, seed=42)
    print(f"WOA:  makespan = {woa_sol.makespan}")
    print(
        f"Improvement: "
        f"{(neh_sol.makespan - woa_sol.makespan) / neh_sol.makespan * 100:.1f}%"
    )
