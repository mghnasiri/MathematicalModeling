"""
Particle Swarm Optimization (PSO) for Fm | prmu | Cmax.

Particle Swarm Optimization is a population-based metaheuristic inspired by the
social behavior of bird flocking and fish schooling (Kennedy & Eberhart, 1995).
For the discrete PFSP, the standard continuous PSO is adapted using the
Smallest Position Value (SPV) encoding (Tasgetiren et al., 2007), where each
particle's continuous position vector is decoded into a job permutation by
sorting jobs according to position values.

Algorithm (Tasgetiren et al., 2007):
    1. Initialize a swarm of particles with random positions/velocities.
    2. Evaluate each particle by decoding to a permutation and computing makespan.
    3. Track each particle's personal best (pbest) and the global best (gbest).
    4. Update velocity: v = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x).
    5. Update position: x = x + v.
    6. Decode to permutation, apply optional local search on best particle.
    7. Repeat until termination.

Complexity: O(generations * pop_size * n * m) per generation for evaluation,
    plus local search overhead on the best particle.

Reference:
    Tasgetiren, M.F., Liang, Y.C., Sevkli, M. & Gencyilmaz, G. (2007).
    A particle swarm optimization algorithm for makespan and total flowtime
    minimization in the permutation flowshop sequencing problem. European
    Journal of Operational Research, 177(3), 1930-1947.
    https://doi.org/10.1016/j.ejor.2005.12.024

    Kennedy, J. & Eberhart, R. (1995). Particle swarm optimization.
    Proceedings of ICNN'95, 4, 1942-1948.
    https://doi.org/10.1109/ICNN.1995.488968
"""

from __future__ import annotations

import sys
import os
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from instance import FlowShopInstance, FlowShopSolution, compute_makespan
from heuristics.neh import neh


def _decode_spv(position: np.ndarray) -> list[int]:
    """Decode continuous position to permutation using Smallest Position Value.

    Args:
        position: Continuous position vector of length n.

    Returns:
        Permutation of job indices.
    """
    return list(np.argsort(position))


def _insertion_local_search(
    instance: FlowShopInstance,
    permutation: list[int],
    max_no_improve: int = 5,
) -> tuple[list[int], int]:
    """Light insertion-based local search for the best particle.

    Args:
        instance: Flow shop instance.
        permutation: Current permutation.
        max_no_improve: Stop after this many passes without improvement.

    Returns:
        Tuple of (improved permutation, makespan).
    """
    perm = list(permutation)
    ms = compute_makespan(instance, perm)
    no_improve = 0

    while no_improve < max_no_improve:
        improved = False
        for i in range(len(perm)):
            job = perm.pop(i)
            best_pos = i
            best_ms = ms

            for j in range(len(perm) + 1):
                perm.insert(j, job)
                new_ms = compute_makespan(instance, perm)
                if new_ms < best_ms:
                    best_ms = new_ms
                    best_pos = j
                    improved = True
                perm.pop(j)

            perm.insert(best_pos, job)
            ms = best_ms

        if not improved:
            no_improve += 1
        else:
            no_improve = 0

    return perm, ms


def particle_swarm_optimization(
    instance: FlowShopInstance,
    swarm_size: int = 30,
    max_iterations: int = 500,
    w: float = 0.7,
    c1: float = 1.5,
    c2: float = 1.5,
    local_search: bool = True,
    time_limit: float | None = None,
    seed: int | None = None,
) -> FlowShopSolution:
    """Solve PFSP using Particle Swarm Optimization with SPV encoding.

    Args:
        instance: Flow shop instance.
        swarm_size: Number of particles in the swarm.
        max_iterations: Maximum number of iterations.
        w: Inertia weight (controls velocity momentum).
        c1: Cognitive coefficient (attraction to personal best).
        c2: Social coefficient (attraction to global best).
        local_search: Apply insertion LS to global best each iteration.
        time_limit: Time limit in seconds (overrides max_iterations).
        seed: Random seed for reproducibility.

    Returns:
        Best FlowShopSolution found.
    """
    rng = np.random.default_rng(seed)
    n = instance.n
    start_time = time.time()

    # Initialize swarm
    positions = rng.uniform(-4, 4, size=(swarm_size, n))
    velocities = rng.uniform(-1, 1, size=(swarm_size, n))

    # Seed one particle with NEH solution
    neh_sol = neh(instance)
    neh_rank = np.zeros(n)
    for rank, job in enumerate(neh_sol.permutation):
        neh_rank[job] = rank
    positions[0] = neh_rank + rng.uniform(-0.1, 0.1, size=n)

    # Evaluate initial swarm
    pbest_pos = positions.copy()
    pbest_ms = np.full(swarm_size, np.iinfo(np.int64).max)

    gbest_perm = list(neh_sol.permutation)
    gbest_ms = neh_sol.makespan

    for i in range(swarm_size):
        perm = _decode_spv(positions[i])
        ms = compute_makespan(instance, perm)
        pbest_ms[i] = ms
        if ms < gbest_ms:
            gbest_ms = ms
            gbest_perm = list(perm)

    # Encode gbest as position for velocity update
    gbest_pos = np.zeros(n)
    for rank, job in enumerate(gbest_perm):
        gbest_pos[job] = rank

    for iteration in range(max_iterations):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        r1 = rng.uniform(0, 1, size=(swarm_size, n))
        r2 = rng.uniform(0, 1, size=(swarm_size, n))

        # Update velocities
        velocities = (w * velocities
                      + c1 * r1 * (pbest_pos - positions)
                      + c2 * r2 * (gbest_pos - positions))

        # Clamp velocities
        v_max = 4.0
        velocities = np.clip(velocities, -v_max, v_max)

        # Update positions
        positions = positions + velocities

        # Evaluate and update personal/global bests
        for i in range(swarm_size):
            perm = _decode_spv(positions[i])
            ms = compute_makespan(instance, perm)

            if ms < pbest_ms[i]:
                pbest_ms[i] = ms
                pbest_pos[i] = positions[i].copy()

            if ms < gbest_ms:
                gbest_ms = ms
                gbest_perm = list(perm)
                gbest_pos = positions[i].copy()

        # Optional local search on global best
        if local_search and iteration % 10 == 0:
            ls_perm, ls_ms = _insertion_local_search(
                instance, gbest_perm, max_no_improve=2,
            )
            if ls_ms < gbest_ms:
                gbest_ms = ls_ms
                gbest_perm = list(ls_perm)
                for rank, job in enumerate(gbest_perm):
                    gbest_pos[job] = rank

    return FlowShopSolution(
        permutation=gbest_perm,
        makespan=gbest_ms,
    )


if __name__ == "__main__":
    instance = FlowShopInstance.random(n=20, m=5, seed=42)

    neh_sol = neh(instance)
    print(f"NEH:  makespan = {neh_sol.makespan}")

    pso_sol = particle_swarm_optimization(instance, max_iterations=200, seed=42)
    print(f"PSO:  makespan = {pso_sol.makespan}")
    print(f"Improvement: {(neh_sol.makespan - pso_sol.makespan) / neh_sol.makespan * 100:.1f}%")
