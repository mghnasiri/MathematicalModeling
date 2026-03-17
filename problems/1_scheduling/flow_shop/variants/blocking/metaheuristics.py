"""
Metaheuristics for Blocking Flow Shop — Fm | prmu, blocking | Cmax

Iterated Greedy adapted for the blocking flow shop constraint. Uses the
same destroy-and-repair framework as the standard IG (Ruiz & Stuetzle, 2007)
with blocking-aware makespan evaluation.

Reference: Grabowski, J. & Pempera, J. (2007). "The Permutation Flow Shop
           Problem with Blocking. A Tabu Search Approach"
           Omega, 35(3):302-311.
           DOI: 10.1016/j.omega.2005.07.004

           Ribas, I., Companys, R. & Tort-Martorell, X. (2011). "An Iterated
           Greedy Algorithm for the Flowshop Scheduling Problem with Blocking"
           Omega, 39(3):293-301.
           DOI: 10.1016/j.omega.2010.07.007
"""

from __future__ import annotations
import sys
import os
import math
import time
import importlib.util
import numpy as np

# Use direct path-based imports to avoid collision with flow_shop/instance.py
_this_dir = os.path.dirname(os.path.abspath(__file__))

_instance_path = os.path.join(_this_dir, "instance.py")
_spec = importlib.util.spec_from_file_location("blk_instance", _instance_path)
_blk_instance = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("blk_instance", _blk_instance)
_spec.loader.exec_module(_blk_instance)

BlockingFlowShopInstance = _blk_instance.BlockingFlowShopInstance
BlockingFlowShopSolution = _blk_instance.BlockingFlowShopSolution
compute_makespan_blocking = _blk_instance.compute_makespan_blocking

_heur_path = os.path.join(_this_dir, "heuristics.py")
_spec2 = importlib.util.spec_from_file_location("blk_heuristics", _heur_path)
_blk_heuristics = importlib.util.module_from_spec(_spec2)
sys.modules.setdefault("blk_heuristics", _blk_heuristics)
_spec2.loader.exec_module(_blk_heuristics)

neh_blocking = _blk_heuristics.neh_blocking


def iterated_greedy_blocking(
    instance: BlockingFlowShopInstance,
    d: int | None = None,
    temperature_factor: float = 0.5,
    time_limit: float | None = None,
    max_iterations: int = 1000,
    seed: int | None = None,
) -> BlockingFlowShopSolution:
    """
    Iterated Greedy algorithm adapted for blocking flow shop.

    Uses NEH-B for initialization and blocking-aware makespan evaluation
    throughout the destroy-and-repair cycle.

    Args:
        instance: A BlockingFlowShopInstance.
        d: Number of jobs to remove in destruction phase.
            Default: min(4, max(2, n // 5)).
        temperature_factor: Controls acceptance probability.
        time_limit: Maximum runtime in seconds. If None, uses max_iterations.
        max_iterations: Maximum number of iterations (if no time_limit).
        seed: Random seed for reproducibility.

    Returns:
        BlockingFlowShopSolution with the best permutation found.
    """
    rng = np.random.default_rng(seed)
    n = instance.n
    p = instance.processing_times

    if d is None:
        d = min(4, max(2, n // 5))

    # Temperature
    total_processing = float(p.sum())
    temperature = temperature_factor * total_processing / (n * instance.m * 10)

    # Initial solution
    initial = neh_blocking(instance)
    current_perm = list(initial.permutation)
    current_ms = initial.makespan

    best_perm = list(current_perm)
    best_ms = current_ms

    start_time = time.time()

    for iteration in range(max_iterations):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        # Destruction
        destroyed = list(current_perm)
        removed = []
        indices = sorted(
            rng.choice(len(destroyed), size=d, replace=False),
            reverse=True
        )
        for idx in indices:
            removed.append(destroyed.pop(idx))
        rng.shuffle(removed)

        # Construction
        for job in removed:
            best_insert_ms = float('inf')
            best_insert_pos = 0
            for pos in range(len(destroyed) + 1):
                candidate = destroyed[:pos] + [job] + destroyed[pos:]
                ms = compute_makespan_blocking(instance, candidate)
                if ms < best_insert_ms:
                    best_insert_ms = ms
                    best_insert_pos = pos
            destroyed.insert(best_insert_pos, job)

        # Local search
        new_perm = _local_search_blocking(instance, destroyed)
        new_ms = compute_makespan_blocking(instance, new_perm)

        # Acceptance
        delta = new_ms - current_ms
        if delta <= 0:
            current_perm = new_perm
            current_ms = new_ms
        elif temperature > 0:
            if rng.random() < math.exp(-delta / temperature):
                current_perm = new_perm
                current_ms = new_ms

        if current_ms < best_ms:
            best_perm = list(current_perm)
            best_ms = current_ms

    return BlockingFlowShopSolution(permutation=best_perm, makespan=best_ms)


def _local_search_blocking(
    instance: BlockingFlowShopInstance,
    permutation: list[int],
) -> list[int]:
    """First-improvement insertion local search for blocking flow shop."""
    perm = list(permutation)
    current_ms = compute_makespan_blocking(instance, perm)
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
                ms = compute_makespan_blocking(instance, candidate)
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
    print("Iterated Greedy — Blocking Flow Shop")
    print("=" * 60)

    instance = BlockingFlowShopInstance.random(n=20, m=5, seed=42)

    sol_neh = neh_blocking(instance)
    print(f"\nNEH-B  Makespan: {sol_neh.makespan}")

    sol_ig = iterated_greedy_blocking(instance, max_iterations=200, seed=42)
    print(f"IG-B   Makespan: {sol_ig.makespan}")
    print(f"Improvement:     {sol_neh.makespan - sol_ig.makespan} "
          f"({(sol_neh.makespan - sol_ig.makespan) / sol_neh.makespan * 100:.1f}%)")
