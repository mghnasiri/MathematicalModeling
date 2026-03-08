"""
Iterated Greedy (IG) — State-of-the-Art Metaheuristic for Fm | prmu | Cmax

The IG algorithm by Ruiz & Stützle (2007) is one of the most effective and
elegant algorithms for the permutation flow shop makespan problem. Despite
its simplicity, it matches or beats far more complex metaheuristics.

Algorithm:
    1. Generate initial solution using NEH.
    2. Repeat until termination:
       a. DESTRUCTION: Remove d randomly chosen jobs from the sequence.
       b. CONSTRUCTION: Reinsert removed jobs one by one using NEH-style
          insertion (try every position, keep the best).
       c. LOCAL SEARCH: Apply pairwise insertion local search.
       d. ACCEPTANCE: Accept the new solution if it improves, or with a
          probability based on a temperature parameter (SA-like).

Key parameters:
    - d: Number of jobs to destroy (typically 4-8, or n_jobs/m)
    - T: Temperature for acceptance criterion
    - Termination: Usually a time limit (e.g., n*m*30 ms)

Notation: Fm | prmu | Cmax
Reference: Ruiz, R. & Stützle, T. (2007). "A Simple and Effective Iterated
           Greedy Algorithm for the Permutation Flowshop Scheduling Problem"
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


def iterated_greedy(
    instance: FlowShopInstance,
    d: int | None = None,
    temperature_factor: float = 0.5,
    time_limit: float | None = None,
    max_iterations: int = 1000,
    seed: int | None = None,
) -> FlowShopSolution:
    """
    Apply the Iterated Greedy algorithm to a permutation flow shop instance.

    Args:
        instance: A FlowShopInstance.
        d: Number of jobs to remove in destruction phase.
           Default: min(4, n//2) following Ruiz & Stützle.
        temperature_factor: Controls acceptance probability.
           T = temperature_factor * (sum of all processing times) / (n * m * 10).
        time_limit: Maximum runtime in seconds. If None, uses max_iterations.
        max_iterations: Maximum number of IG iterations (if no time_limit).
        seed: Random seed for reproducibility.

    Returns:
        FlowShopSolution with the best permutation found.
    """
    rng = np.random.default_rng(seed)
    p = instance.processing_times
    n = instance.n
    m = instance.m

    # Default destruction size: following the paper's recommendation
    if d is None:
        d = min(4, max(2, n // 5))

    # Temperature for acceptance criterion (Ruiz & Stützle formula)
    total_processing = float(p.sum())
    temperature = temperature_factor * total_processing / (n * m * 10)

    # Step 1: Initial solution via NEH
    initial = neh(instance)
    current_perm = list(initial.permutation)
    current_ms = initial.makespan

    best_perm = list(current_perm)
    best_ms = current_ms

    # Iteration loop
    start_time = time.time()
    iteration = 0

    while True:
        # Check termination
        if time_limit is not None:
            if time.time() - start_time >= time_limit:
                break
        else:
            if iteration >= max_iterations:
                break

        iteration += 1

        # Step 2a: DESTRUCTION — remove d random jobs
        destroyed_perm = list(current_perm)
        removed_jobs = []

        indices_to_remove = sorted(
            rng.choice(len(destroyed_perm), size=d, replace=False),
            reverse=True  # Remove from back to front to preserve indices
        )
        for idx in indices_to_remove:
            removed_jobs.append(destroyed_perm.pop(idx))

        # Shuffle the removed jobs (random reinsertion order)
        rng.shuffle(removed_jobs)

        # Step 2b: CONSTRUCTION — reinsert using NEH-style insertion
        for job in removed_jobs:
            best_insert_ms = float('inf')
            best_insert_pos = 0

            for pos in range(len(destroyed_perm) + 1):
                candidate = destroyed_perm[:pos] + [job] + destroyed_perm[pos:]
                ms = compute_makespan(instance, candidate)

                if ms < best_insert_ms:
                    best_insert_ms = ms
                    best_insert_pos = pos

            destroyed_perm.insert(best_insert_pos, job)

        # Step 2c: LOCAL SEARCH — insertion neighborhood
        new_perm = _local_search_insert(instance, destroyed_perm)
        new_ms = compute_makespan(instance, new_perm)

        # Step 2d: ACCEPTANCE CRITERION
        # Accept if improvement, or with SA-like probability
        delta = new_ms - current_ms

        if delta <= 0:
            current_perm = new_perm
            current_ms = new_ms
        else:
            # Metropolis-like acceptance
            accept_prob = math.exp(-delta / temperature) if temperature > 0 else 0
            if rng.random() < accept_prob:
                current_perm = new_perm
                current_ms = new_ms

        # Update best
        if current_ms < best_ms:
            best_perm = list(current_perm)
            best_ms = current_ms

    return FlowShopSolution(permutation=best_perm, makespan=best_ms)


def _local_search_insert(instance: FlowShopInstance,
                         permutation: list[int]) -> list[int]:
    """
    Insertion-based local search.

    For each job, try removing it and reinserting it at every other position.
    Accept the first improving move (first-improvement strategy).
    Repeat until no improvement is found.
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
                    continue  # Skip the original position

                candidate = remaining[:pos] + [job] + remaining[pos:]
                ms = compute_makespan(instance, candidate)

                if ms < current_ms:
                    perm = candidate
                    current_ms = ms
                    improved = True
                    break  # First improvement — restart

            if improved:
                break

    return perm


if __name__ == "__main__":
    # Compare all algorithms on a 20×5 random instance
    print("=" * 60)
    print("Permutation Flow Shop — Algorithm Comparison")
    print("=" * 60)

    rand_instance = FlowShopInstance.random(n=20, m=5, seed=42)

    # CDS
    from heuristics.cds import cds
    sol_cds = cds(rand_instance)
    print(f"\nCDS  Makespan:  {sol_cds.makespan}")

    # NEH
    sol_neh = neh(rand_instance)
    print(f"NEH  Makespan:  {sol_neh.makespan}")

    # IG (limited iterations for demo)
    sol_ig = iterated_greedy(rand_instance, max_iterations=200, seed=42)
    print(f"IG   Makespan:  {sol_ig.makespan}")

    print(f"\nIG improvement over NEH: "
          f"{sol_neh.makespan - sol_ig.makespan} "
          f"({(sol_neh.makespan - sol_ig.makespan) / sol_neh.makespan * 100:.1f}%)")
    print(f"IG improvement over CDS: "
          f"{sol_cds.makespan - sol_ig.makespan} "
          f"({(sol_cds.makespan - sol_ig.makespan) / sol_cds.makespan * 100:.1f}%)")

    # Larger test
    print("\n" + "=" * 60)
    print("Larger Instance: 50×10")
    print("=" * 60)

    large_instance = FlowShopInstance.random(n=50, m=10, seed=123)
    sol_cds_lg = cds(large_instance)
    sol_neh_lg = neh(large_instance)

    t0 = time.time()
    sol_ig_lg = iterated_greedy(large_instance, time_limit=2.0, seed=123)
    elapsed = time.time() - t0

    print(f"CDS  Makespan:  {sol_cds_lg.makespan}")
    print(f"NEH  Makespan:  {sol_neh_lg.makespan}")
    print(f"IG   Makespan:  {sol_ig_lg.makespan}  ({elapsed:.1f}s)")
