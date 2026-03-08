"""
Metaheuristics for No-Wait Flow Shop — Fm | prmu, no-wait | Cmax

Iterated Greedy adapted for the no-wait constraint. The destroy-and-repair
framework of Ruiz & Stuetzle (2007) is modified to use the no-wait makespan
computation based on inter-job delay values.

Reference: Pan, Q.-K., Tasgetiren, M.F. & Liang, Y.-C. (2008). "A Discrete
           Differential Evolution Algorithm for the Permutation Flowshop
           Scheduling Problem"
           Computers & Industrial Engineering, 55(4):795-816.
           DOI: 10.1016/j.cie.2008.03.003

           Ruiz, R. & Stuetzle, T. (2007). "A Simple and Effective Iterated
           Greedy Algorithm for the Permutation Flowshop Scheduling Problem"
           European Journal of Operational Research, 177(3):2033-2049.
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
_spec = importlib.util.spec_from_file_location("nw_instance", _instance_path)
_nw_instance = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("nw_instance", _nw_instance)
_spec.loader.exec_module(_nw_instance)

NoWaitFlowShopInstance = _nw_instance.NoWaitFlowShopInstance
NoWaitFlowShopSolution = _nw_instance.NoWaitFlowShopSolution
compute_delay_matrix = _nw_instance.compute_delay_matrix
compute_makespan_nw = _nw_instance.compute_makespan_nw

_heur_path = os.path.join(_this_dir, "heuristics.py")
_spec2 = importlib.util.spec_from_file_location("nw_heuristics", _heur_path)
_nw_heuristics = importlib.util.module_from_spec(_spec2)
sys.modules.setdefault("nw_heuristics", _nw_heuristics)
_spec2.loader.exec_module(_nw_heuristics)

neh_no_wait = _nw_heuristics.neh_no_wait


def iterated_greedy_nw(
    instance: NoWaitFlowShopInstance,
    d: int | None = None,
    temperature_factor: float = 0.5,
    time_limit: float | None = None,
    max_iterations: int = 1000,
    seed: int | None = None,
) -> NoWaitFlowShopSolution:
    """
    Iterated Greedy algorithm adapted for no-wait flow shop.

    Uses the same destroy-and-repair framework as the standard IG, but
    with no-wait makespan evaluation. The insertion during the repair
    phase places each job in the position minimizing the no-wait makespan.

    Args:
        instance: A NoWaitFlowShopInstance.
        d: Number of jobs to remove in destruction phase.
            Default: min(4, max(2, n // 5)).
        temperature_factor: Controls acceptance probability.
        time_limit: Maximum runtime in seconds. If None, uses max_iterations.
        max_iterations: Maximum number of iterations (if no time_limit).
        seed: Random seed for reproducibility.

    Returns:
        NoWaitFlowShopSolution with the best permutation found.
    """
    rng = np.random.default_rng(seed)
    n = instance.n
    p = instance.processing_times
    D = compute_delay_matrix(instance)

    if d is None:
        d = min(4, max(2, n // 5))

    # Temperature
    total_processing = float(p.sum())
    temperature = temperature_factor * total_processing / (n * instance.m * 10)

    # Initial solution via NEH-NW
    initial = neh_no_wait(instance)
    current_perm = list(initial.permutation)
    current_ms = initial.makespan

    best_perm = list(current_perm)
    best_ms = current_ms

    start_time = time.time()

    for iteration in range(max_iterations):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        # Destruction: remove d random jobs
        destroyed = list(current_perm)
        removed = []
        indices = sorted(
            rng.choice(len(destroyed), size=d, replace=False),
            reverse=True
        )
        for idx in indices:
            removed.append(destroyed.pop(idx))
        rng.shuffle(removed)

        # Construction: reinsert using best-position insertion
        for job in removed:
            best_insert_ms = float('inf')
            best_insert_pos = 0
            for pos in range(len(destroyed) + 1):
                candidate = destroyed[:pos] + [job] + destroyed[pos:]
                ms = compute_makespan_nw(instance, candidate, D)
                if ms < best_insert_ms:
                    best_insert_ms = ms
                    best_insert_pos = pos
            destroyed.insert(best_insert_pos, job)

        # Local search: first-improvement insertion
        new_perm = _local_search_nw(instance, destroyed, D)
        new_ms = compute_makespan_nw(instance, new_perm, D)

        # Acceptance criterion
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

    return NoWaitFlowShopSolution(permutation=best_perm, makespan=best_ms)


def _local_search_nw(
    instance: NoWaitFlowShopInstance,
    permutation: list[int],
    D: np.ndarray,
) -> list[int]:
    """First-improvement insertion local search for no-wait flow shop."""
    perm = list(permutation)
    current_ms = compute_makespan_nw(instance, perm, D)
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
                ms = compute_makespan_nw(instance, candidate, D)
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
    print("Iterated Greedy — No-Wait Flow Shop")
    print("=" * 60)

    instance = NoWaitFlowShopInstance.random(n=20, m=5, seed=42)

    sol_neh = neh_no_wait(instance)
    print(f"\nNEH-NW Makespan: {sol_neh.makespan}")

    sol_ig = iterated_greedy_nw(instance, max_iterations=200, seed=42)
    print(f"IG-NW  Makespan: {sol_ig.makespan}")
    print(f"Improvement:     {sol_neh.makespan - sol_ig.makespan} "
          f"({(sol_neh.makespan - sol_ig.makespan) / sol_neh.makespan * 100:.1f}%)")
