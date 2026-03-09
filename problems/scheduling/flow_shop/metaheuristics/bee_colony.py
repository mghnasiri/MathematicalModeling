"""
Artificial Bee Colony (ABC) for Fm | prmu | Cmax.

The Artificial Bee Colony algorithm simulates the foraging behavior of
honey bees (Karaboga, 2005). The colony consists of three groups:
employed bees exploit known food sources, onlooker bees select food
sources probabilistically based on quality, and scout bees discover
new sources when existing ones are exhausted.

For the discrete PFSP, food sources are permutations. Employed bees
apply insertion/swap moves as neighborhood operators. Onlooker bees
use fitness-proportional selection to concentrate search on promising
regions. Abandoned sources (stagnant for `limit` iterations) are
replaced by random or NEH-seeded new solutions.

Algorithm (Tasgetiren et al., 2011):
    1. Initialize SN food sources (half with NEH perturbations, half random).
    2. Employed bee phase: each bee applies insert/swap to its source,
       keeping the move if it improves.
    3. Onlooker bee phase: select sources proportionally to fitness,
       apply insert/swap to selected sources.
    4. Scout bee phase: replace abandoned sources (trial > limit).
    5. Track global best across all phases.
    6. Repeat until termination.

Complexity: O(iterations * SN * n * m) per iteration for evaluation.

Reference:
    Tasgetiren, M.F., Pan, Q.K., Suganthan, P.N. & Chen, A.H.L. (2011).
    A discrete artificial bee colony algorithm for the total flowtime
    minimization in permutation flow shops. Information Sciences,
    181(16), 3459-3475.
    https://doi.org/10.1016/j.ins.2011.04.018

    Karaboga, D. (2005). An idea based on honey bee swarm for numerical
    optimization. Technical Report TR06, Erciyes University.

    Pan, Q.K., Tasgetiren, M.F., Suganthan, P.N. & Chua, T.J. (2011).
    A discrete artificial bee colony algorithm for the lot-streaming
    flow shop scheduling problem. Information Sciences, 181(12),
    2455-2468. https://doi.org/10.1016/j.ins.2009.12.025
"""

from __future__ import annotations

import sys
import os
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from instance import FlowShopInstance, FlowShopSolution, compute_makespan
from heuristics.neh import neh


def _insert_move(
    perm: list[int],
    rng: np.random.Generator,
) -> list[int]:
    """Apply a random insertion move.

    Args:
        perm: Current permutation.
        rng: Random number generator.

    Returns:
        New permutation with one job re-inserted.
    """
    n = len(perm)
    new_perm = list(perm)
    i = rng.integers(0, n)
    job = new_perm.pop(i)
    j = rng.integers(0, n)
    if j >= i:
        j = max(0, j - 1)  # avoid same position
    new_perm.insert(j, job)
    return new_perm


def _swap_move(
    perm: list[int],
    rng: np.random.Generator,
) -> list[int]:
    """Apply a random swap move.

    Args:
        perm: Current permutation.
        rng: Random number generator.

    Returns:
        New permutation with two jobs swapped.
    """
    n = len(perm)
    if n < 2:
        return list(perm)
    new_perm = list(perm)
    i, j = rng.choice(n, size=2, replace=False)
    new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
    return new_perm


def _neighborhood_move(
    instance: FlowShopInstance,
    perm: list[int],
    rng: np.random.Generator,
) -> tuple[list[int], int]:
    """Apply insert or swap move, return best of the two.

    Args:
        instance: Flow shop instance.
        perm: Current permutation.
        rng: Random number generator.

    Returns:
        Tuple of (new permutation, makespan).
    """
    ins_perm = _insert_move(perm, rng)
    ins_ms = compute_makespan(instance, ins_perm)

    swap_perm = _swap_move(perm, rng)
    swap_ms = compute_makespan(instance, swap_perm)

    if ins_ms <= swap_ms:
        return ins_perm, ins_ms
    return swap_perm, swap_ms


def artificial_bee_colony(
    instance: FlowShopInstance,
    colony_size: int = 30,
    max_iterations: int = 500,
    limit: int | None = None,
    time_limit: float | None = None,
    seed: int | None = None,
) -> FlowShopSolution:
    """Solve PFSP using Artificial Bee Colony algorithm.

    Args:
        instance: Flow shop instance.
        colony_size: Number of food sources (SN = colony_size // 2 employed bees).
        max_iterations: Maximum number of iterations.
        limit: Abandonment threshold. Default: SN * n.
        time_limit: Time limit in seconds (overrides max_iterations).
        seed: Random seed for reproducibility.

    Returns:
        Best FlowShopSolution found.
    """
    rng = np.random.default_rng(seed)
    n = instance.n
    SN = colony_size // 2  # number of food sources
    start_time = time.time()

    if limit is None:
        limit = SN * n

    # ── Initialize food sources ──────────────────────────────────────────
    neh_sol = neh(instance)
    sources: list[list[int]] = [list(neh_sol.permutation)]
    fitness: list[int] = [neh_sol.makespan]

    # Generate remaining sources: perturbations of NEH + random
    for i in range(1, SN):
        if i < SN // 2:
            # Perturbation of NEH
            perm = list(neh_sol.permutation)
            for _ in range(rng.integers(1, max(2, n // 3))):
                perm = _insert_move(perm, rng)
        else:
            # Random permutation
            perm = list(range(n))
            rng.shuffle(perm)
        ms = compute_makespan(instance, perm)
        sources.append(perm)
        fitness.append(ms)

    trials = [0] * SN  # stagnation counter per source
    gbest_perm = list(sources[0])
    gbest_ms = fitness[0]

    # Find initial best
    for i in range(SN):
        if fitness[i] < gbest_ms:
            gbest_ms = fitness[i]
            gbest_perm = list(sources[i])

    for iteration in range(max_iterations):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        # ── Employed bee phase ───────────────────────────────────────────
        for i in range(SN):
            new_perm, new_ms = _neighborhood_move(instance, sources[i], rng)
            if new_ms <= fitness[i]:
                sources[i] = new_perm
                fitness[i] = new_ms
                trials[i] = 0
                if new_ms < gbest_ms:
                    gbest_ms = new_ms
                    gbest_perm = list(new_perm)
            else:
                trials[i] += 1

        # ── Onlooker bee phase ───────────────────────────────────────────
        # Fitness-proportional selection (inverse makespan)
        max_ms = max(fitness)
        fit_values = [max_ms - f + 1.0 for f in fitness]
        total_fit = sum(fit_values)
        probs = [f / total_fit for f in fit_values]

        for _ in range(SN):
            # Roulette wheel selection
            selected = rng.choice(SN, p=probs)
            new_perm, new_ms = _neighborhood_move(
                instance, sources[selected], rng,
            )
            if new_ms <= fitness[selected]:
                sources[selected] = new_perm
                fitness[selected] = new_ms
                trials[selected] = 0
                if new_ms < gbest_ms:
                    gbest_ms = new_ms
                    gbest_perm = list(new_perm)
            else:
                trials[selected] += 1

        # ── Scout bee phase ──────────────────────────────────────────────
        for i in range(SN):
            if trials[i] >= limit:
                # Abandon and reinitialize
                perm = list(range(n))
                rng.shuffle(perm)
                ms = compute_makespan(instance, perm)
                sources[i] = perm
                fitness[i] = ms
                trials[i] = 0

    return FlowShopSolution(
        permutation=gbest_perm,
        makespan=gbest_ms,
    )


if __name__ == "__main__":
    instance = FlowShopInstance.random(n=20, m=5, seed=42)

    neh_sol = neh(instance)
    print(f"NEH:  makespan = {neh_sol.makespan}")

    abc_sol = artificial_bee_colony(instance, max_iterations=300, seed=42)
    print(f"ABC:  makespan = {abc_sol.makespan}")
    print(f"Improvement: {(neh_sol.makespan - abc_sol.makespan) / neh_sol.makespan * 100:.1f}%")
