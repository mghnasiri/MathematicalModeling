"""
Metaheuristics for SDST Flow Shop — Fm | prmu, Ssd | Cmax

Iterated Greedy adapted for the permutation flow shop with sequence-dependent
setup times. Uses the same destroy-and-repair framework as Ruiz & Stützle
(2007) with SDST-aware makespan evaluation throughout.

Reference: Ruiz, R., Maroto, C. & Alcaraz, J. (2005). "Solving the Flowshop
           Scheduling Problem with Sequence Dependent Setup Times Using Advanced
           Metaheuristics"
           European Journal of Operational Research, 165(1):34-54.
           DOI: 10.1016/j.ejor.2004.01.022

           Ruiz, R. & Stützle, T. (2008). "An Iterated Greedy Heuristic for
           the Sequence Dependent Setup Times Flowshop Problem with Makespan
           and Weighted Tardiness Objectives"
           European Journal of Operational Research, 187(3):1143-1159.
           DOI: 10.1016/j.ejor.2006.07.029
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
_spec = importlib.util.spec_from_file_location("sdst_instance", _instance_path)
_sdst_instance = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("sdst_instance", _sdst_instance)
_spec.loader.exec_module(_sdst_instance)

SDSTFlowShopInstance = _sdst_instance.SDSTFlowShopInstance
SDSTFlowShopSolution = _sdst_instance.SDSTFlowShopSolution
compute_makespan_sdst = _sdst_instance.compute_makespan_sdst

_heur_path = os.path.join(_this_dir, "heuristics.py")
_spec2 = importlib.util.spec_from_file_location("sdst_heuristics", _heur_path)
_sdst_heuristics = importlib.util.module_from_spec(_spec2)
sys.modules.setdefault("sdst_heuristics", _sdst_heuristics)
_spec2.loader.exec_module(_sdst_heuristics)

neh_sdst = _sdst_heuristics.neh_sdst


def iterated_greedy_sdst(
    instance: SDSTFlowShopInstance,
    d: int | None = None,
    temperature_factor: float = 0.5,
    time_limit: float | None = None,
    max_iterations: int = 1000,
    seed: int | None = None,
) -> SDSTFlowShopSolution:
    """
    Iterated Greedy algorithm adapted for the SDST flow shop.

    Uses NEH-SDST for initialization and SDST-aware makespan evaluation
    throughout the destroy-and-repair cycle.

    Args:
        instance: An SDSTFlowShopInstance.
        d: Number of jobs to remove in destruction phase.
            Default: min(4, max(2, n // 5)).
        temperature_factor: Controls acceptance probability.
        time_limit: Maximum runtime in seconds. If None, uses max_iterations.
        max_iterations: Maximum number of iterations (if no time_limit).
        seed: Random seed for reproducibility.

    Returns:
        SDSTFlowShopSolution with the best permutation found.
    """
    rng = np.random.default_rng(seed)
    n = instance.n
    p = instance.processing_times

    if d is None:
        d = min(4, max(2, n // 5))

    # Temperature (include setup times in total for proper scaling)
    total_processing = float(p.sum()) + float(instance.setup_times.sum()) / (n + 1)
    temperature = temperature_factor * total_processing / (n * instance.m * 10)

    # Initial solution via NEH-SDST
    initial = neh_sdst(instance)
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

        # Construction: reinsert with NEH-style best-position insertion
        for job in removed:
            best_insert_ms = float('inf')
            best_insert_pos = 0
            for pos in range(len(destroyed) + 1):
                candidate = destroyed[:pos] + [job] + destroyed[pos:]
                ms = compute_makespan_sdst(instance, candidate)
                if ms < best_insert_ms:
                    best_insert_ms = ms
                    best_insert_pos = pos
            destroyed.insert(best_insert_pos, job)

        # Local search
        new_perm = _local_search_sdst(instance, destroyed)
        new_ms = compute_makespan_sdst(instance, new_perm)

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

    return SDSTFlowShopSolution(permutation=best_perm, makespan=best_ms)


def _local_search_sdst(
    instance: SDSTFlowShopInstance,
    permutation: list[int],
) -> list[int]:
    """First-improvement insertion local search for SDST flow shop."""
    perm = list(permutation)
    current_ms = compute_makespan_sdst(instance, perm)
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
                ms = compute_makespan_sdst(instance, candidate)
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
    print("Iterated Greedy — SDST Flow Shop")
    print("=" * 60)

    instance = SDSTFlowShopInstance.random(n=20, m=5, seed=42)

    sol_neh = neh_sdst(instance)
    print(f"\nNEH-SDST  Makespan: {sol_neh.makespan}")

    sol_ig = iterated_greedy_sdst(instance, max_iterations=200, seed=42)
    print(f"IG-SDST   Makespan: {sol_ig.makespan}")
    print(f"Improvement:        {sol_neh.makespan - sol_ig.makespan} "
          f"({(sol_neh.makespan - sol_ig.makespan) / sol_neh.makespan * 100:.1f}%)")
