"""
Ant Colony Optimization (ACO) — Population-Based Metaheuristic for Fm | prmu | Cmax

An Ant Colony Optimization algorithm for the permutation flow shop scheduling
problem. Artificial ants construct solutions by probabilistically selecting
jobs based on pheromone trails and heuristic information, mimicking the
foraging behavior of real ants.

Algorithm:
    1. Initialize pheromone matrix uniformly.
    2. Repeat until termination:
       a. Each ant constructs a complete permutation:
          - At each step, select the next job with probability proportional
            to tau[i][j]^alpha * eta[i][j]^beta, where tau is pheromone and
            eta is heuristic desirability.
       b. (Optional) Apply local search to each ant's solution.
       c. Update pheromones:
          - Evaporate: tau *= (1 - rho)
          - Deposit: best ant deposits pheromone proportional to 1/makespan
    3. Return the best solution found.

Heuristic information (eta):
    Uses NEH-style insertion cost: for each unscheduled job, eta is inversely
    proportional to the incremental makespan when appending that job.

Pheromone model:
    Position-based: tau[j][k] represents desirability of placing job j at
    position k in the permutation.

Key parameters:
    - n_ants: Number of ants per iteration (typically 10-20)
    - alpha: Pheromone influence exponent (typically 1.0)
    - beta: Heuristic influence exponent (typically 2.0)
    - rho: Evaporation rate (typically 0.1-0.3)

Notation: Fm | prmu | Cmax
Complexity: O(n_ants * n^2 * m) per iteration
Reference: Stützle, T. (1998). "An Ant Approach to the Flow Shop Problem"
           Proceedings of the 6th European Congress on Intelligent Techniques
           & Soft Computing (EUFIT '98), pp. 1560-1564.

           Rajendran, C. & Ziegler, H. (2004). "Ant-Colony Algorithms for
           Permutation Flowshop Scheduling to Minimize Makespan/Total Flowtime
           of Jobs"
           European Journal of Operational Research, 155(2):426-438.
           DOI: 10.1016/S0377-2217(02)00908-6
"""

from __future__ import annotations
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from instance import FlowShopInstance, FlowShopSolution, compute_makespan
from heuristics.neh import neh


def ant_colony_optimization(
    instance: FlowShopInstance,
    n_ants: int = 10,
    alpha: float = 1.0,
    beta: float = 2.0,
    rho: float = 0.2,
    time_limit: float | None = None,
    max_iterations: int = 100,
    use_local_search: bool = False,
    seed: int | None = None,
) -> FlowShopSolution:
    """
    Apply Ant Colony Optimization to a permutation flow shop instance.

    Args:
        instance: A FlowShopInstance.
        n_ants: Number of ants per iteration.
        alpha: Pheromone trail importance exponent.
        beta: Heuristic information importance exponent.
        rho: Pheromone evaporation rate (0 < rho < 1).
        time_limit: Maximum runtime in seconds. If None, uses max_iterations.
        max_iterations: Maximum number of ACO iterations (if no time_limit).
        use_local_search: If True, apply insertion local search to each ant's
            solution. Significantly slower but higher quality.
        seed: Random seed for reproducibility.

    Returns:
        FlowShopSolution with the best permutation found.
    """
    rng = np.random.default_rng(seed)
    n = instance.n

    # Initialize pheromone matrix (position-based model)
    # tau[j][k] = desirability of placing job j at position k
    initial_sol = neh(instance)
    tau_0 = 1.0 / (n * initial_sol.makespan)
    tau = np.full((n, n), tau_0, dtype=float)

    best_perm = list(initial_sol.permutation)
    best_ms = initial_sol.makespan

    start_time = time.time()

    for iteration in range(max_iterations):
        # Check time limit
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        iteration_best_perm = None
        iteration_best_ms = float('inf')

        # Each ant constructs a solution
        for _ in range(n_ants):
            perm = _construct_solution(instance, tau, alpha, beta, rng)
            ms = compute_makespan(instance, perm)

            # Optional local search
            if use_local_search:
                perm = _local_search_insert(instance, perm)
                ms = compute_makespan(instance, perm)

            if ms < iteration_best_ms:
                iteration_best_perm = perm
                iteration_best_ms = ms

        # Update global best
        if iteration_best_ms < best_ms:
            best_perm = list(iteration_best_perm)
            best_ms = iteration_best_ms

        # Pheromone update
        # Evaporation
        tau *= (1.0 - rho)

        # Deposit by iteration-best ant
        deposit = 1.0 / iteration_best_ms
        for pos, job in enumerate(iteration_best_perm):
            tau[job, pos] += deposit

        # Also deposit by global best (elitist strategy)
        deposit_best = 1.0 / best_ms
        for pos, job in enumerate(best_perm):
            tau[job, pos] += deposit_best

        # Enforce pheromone bounds (MAX-MIN Ant System style)
        tau_max = 1.0 / (rho * best_ms)
        tau_min = tau_max / (2.0 * n)
        np.clip(tau, tau_min, tau_max, out=tau)

    return FlowShopSolution(permutation=best_perm, makespan=best_ms)


def _construct_solution(
    instance: FlowShopInstance,
    tau: np.ndarray,
    alpha: float,
    beta: float,
    rng: np.random.Generator,
) -> list[int]:
    """
    Construct a permutation using the ACO probability rule.

    At each position k, the ant selects an unscheduled job j with probability:
        p(j, k) = [tau(j, k)]^alpha * [eta(j, k)]^beta / sum(...)

    Heuristic eta(j, k) is based on total processing time (shorter jobs
    preferred earlier, similar to SPT-like guidance).

    Args:
        instance: A FlowShopInstance.
        tau: Pheromone matrix of shape (n, n).
        alpha: Pheromone exponent.
        beta: Heuristic exponent.
        rng: Random number generator.

    Returns:
        Constructed permutation.
    """
    n = instance.n
    p = instance.processing_times

    # Precompute heuristic: inverse of total processing time per job
    total_times = p.sum(axis=0).astype(float)
    eta_base = 1.0 / total_times

    perm = []
    unscheduled = set(range(n))

    for pos in range(n):
        jobs = list(unscheduled)

        # Compute selection probabilities
        probs = np.zeros(len(jobs))
        for idx, job in enumerate(jobs):
            pheromone = tau[job, pos] ** alpha
            heuristic = eta_base[job] ** beta
            probs[idx] = pheromone * heuristic

        # Normalize
        total = probs.sum()
        if total > 0:
            probs /= total
        else:
            probs = np.ones(len(jobs)) / len(jobs)

        # Select job
        chosen_idx = rng.choice(len(jobs), p=probs)
        chosen_job = jobs[chosen_idx]

        perm.append(chosen_job)
        unscheduled.remove(chosen_job)

    return perm


def _local_search_insert(
    instance: FlowShopInstance,
    permutation: list[int],
) -> list[int]:
    """
    First-improvement insertion local search.

    Args:
        instance: A FlowShopInstance.
        permutation: Starting permutation.

    Returns:
        Locally optimal permutation.
    """
    perm = list(permutation)
    current_ms = compute_makespan(instance, perm)
    improved = True

    while improved:
        improved = False
        for i in range(len(perm)):
            job = perm[i]
            remaining = perm[:i] + perm[i + 1:]

            for pos in range(len(remaining) + 1):
                if pos == i:
                    continue
                candidate = remaining[:pos] + [job] + remaining[pos:]
                ms = compute_makespan(instance, candidate)
                if ms < current_ms:
                    perm = candidate
                    current_ms = ms
                    improved = True
                    break
            if improved:
                break

    return perm


if __name__ == "__main__":
    print("=" * 60)
    print("Ant Colony Optimization — Permutation Flow Shop")
    print("=" * 60)

    instance = FlowShopInstance.random(n=20, m=5, seed=42)

    from heuristics.cds import cds
    sol_cds = cds(instance)
    sol_neh = neh(instance)

    print(f"\nCDS  Makespan:  {sol_cds.makespan}")
    print(f"NEH  Makespan:  {sol_neh.makespan}")

    # ACO
    sol_aco = ant_colony_optimization(instance, max_iterations=50, seed=42)
    print(f"ACO  Makespan:  {sol_aco.makespan}")

    # ACO with local search (memetic)
    sol_aco_ls = ant_colony_optimization(
        instance, max_iterations=20, use_local_search=True, seed=42
    )
    print(f"ACO+LS Makespan: {sol_aco_ls.makespan}")

    # Compare with other metaheuristics
    from metaheuristics.iterated_greedy import iterated_greedy
    sol_ig = iterated_greedy(instance, max_iterations=500, seed=42)
    print(f"IG   Makespan:  {sol_ig.makespan}")

    # Larger instance
    print("\n" + "=" * 60)
    print("Larger Instance: 50x10")
    print("=" * 60)

    large_instance = FlowShopInstance.random(n=50, m=10, seed=123)
    sol_neh_lg = neh(large_instance)
    print(f"NEH Makespan:   {sol_neh_lg.makespan}")

    t0 = time.time()
    sol_aco_lg = ant_colony_optimization(large_instance, time_limit=2.0, seed=42)
    elapsed = time.time() - t0
    print(f"ACO Makespan:   {sol_aco_lg.makespan}  ({elapsed:.1f}s)")
