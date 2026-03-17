"""
Constructive Heuristics for Blocking Flow Shop — Fm | prmu, blocking | Cmax

This module provides constructive heuristics adapted for the blocking
permutation flow shop scheduling problem. The blocking constraint modifies
the makespan computation (jobs block machines until downstream machines
are available), requiring adapted evaluation in all constructive procedures.

Heuristics implemented:
    1. NEH Adaptation (NEH-B): Adapts the NEH insertion heuristic to the
       blocking flow shop, using blocking-aware makespan evaluation.

    2. Profile Fitting (PF-B): A heuristic that prioritizes jobs minimizing
       total blocking time when inserted into the current partial schedule.

References:
    Ronconi, D.P. (2004). "A Note on Constructive Heuristics for the
    Flowshop Problem with Blocking"
    International Journal of Production Economics, 87(1):39-48.
    DOI: 10.1016/S0925-5273(03)00065-3

    Nawaz, M., Enscore, E.E. & Ham, I. (1983). "A Heuristic Algorithm
    for the m-Machine, n-Job Flow-Shop Sequencing Problem"
    Omega, 11(1):91-95. [Original NEH — adapted here for blocking]
"""

from __future__ import annotations
import sys
import os
import importlib.util
import numpy as np

# Use direct path-based import to avoid collision with flow_shop/instance.py
_this_dir = os.path.dirname(os.path.abspath(__file__))
_instance_path = os.path.join(_this_dir, "instance.py")
_spec = importlib.util.spec_from_file_location("blk_instance", _instance_path)
_blk_instance = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("blk_instance", _blk_instance)
_spec.loader.exec_module(_blk_instance)

BlockingFlowShopInstance = _blk_instance.BlockingFlowShopInstance
BlockingFlowShopSolution = _blk_instance.BlockingFlowShopSolution
compute_makespan_blocking = _blk_instance.compute_makespan_blocking


def neh_blocking(instance: BlockingFlowShopInstance) -> BlockingFlowShopSolution:
    """
    NEH heuristic adapted for the blocking flow shop.

    Follows the standard NEH framework: sort jobs by decreasing total
    processing time, then insert each job into the position that
    minimizes the partial blocking makespan.

    Args:
        instance: A BlockingFlowShopInstance.

    Returns:
        BlockingFlowShopSolution with the constructed permutation and makespan.
    """
    n = instance.n
    p = instance.processing_times

    # Sort by decreasing total processing time
    total_times = p.sum(axis=0)
    sorted_jobs = sorted(range(n), key=lambda j: total_times[j], reverse=True)

    # Start with first job
    sequence = [sorted_jobs[0]]

    # Insert each remaining job into the best position
    for i in range(1, len(sorted_jobs)):
        job = sorted_jobs[i]
        best_ms = float('inf')
        best_pos = 0

        for pos in range(len(sequence) + 1):
            candidate = sequence[:pos] + [job] + sequence[pos:]
            ms = compute_makespan_blocking(instance, candidate)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos

        sequence.insert(best_pos, job)

    makespan = compute_makespan_blocking(instance, sequence)
    return BlockingFlowShopSolution(permutation=sequence, makespan=makespan)


def profile_fitting_blocking(
    instance: BlockingFlowShopInstance,
) -> BlockingFlowShopSolution:
    """
    Profile Fitting heuristic for the blocking flow shop.

    Builds the schedule position by position. At each step, evaluates all
    unscheduled jobs and selects the one that causes the least additional
    blocking time when appended to the current partial schedule.

    This heuristic focuses on minimizing the blocking effect rather than
    the standard makespan, under the principle that less blocking leads
    to better machine utilization.

    Args:
        instance: A BlockingFlowShopInstance.

    Returns:
        BlockingFlowShopSolution with the constructed permutation and makespan.
    """
    n = instance.n
    m = instance.m
    p = instance.processing_times

    # Sort jobs by decreasing total processing time for initial ordering
    total_times = p.sum(axis=0)
    sorted_jobs = sorted(range(n), key=lambda j: total_times[j], reverse=True)

    # Place the first job (heaviest)
    sequence = [sorted_jobs[0]]
    unscheduled = set(sorted_jobs[1:])

    while unscheduled:
        best_job = None
        best_ms = float('inf')

        for job in unscheduled:
            # Try appending this job and compute blocking makespan
            candidate = sequence + [job]
            ms = compute_makespan_blocking(instance, candidate)
            if ms < best_ms:
                best_ms = ms
                best_job = job

        sequence.append(best_job)
        unscheduled.remove(best_job)

    makespan = compute_makespan_blocking(instance, sequence)
    return BlockingFlowShopSolution(permutation=sequence, makespan=makespan)


if __name__ == "__main__":
    # Test on a small instance
    instance = BlockingFlowShopInstance(
        n=6, m=3,
        processing_times=np.array([
            [3, 5, 2, 7, 4, 6],  # Machine 0
            [4, 2, 6, 1, 3, 5],  # Machine 1
            [2, 3, 4, 5, 1, 2],  # Machine 2
        ])
    )

    print("=== Blocking Flow Shop Heuristics ===")
    print(f"Instance: {instance.n} jobs, {instance.m} machines\n")

    sol_neh = neh_blocking(instance)
    print(f"NEH-B:            {sol_neh.permutation}  Cmax={sol_neh.makespan}")

    sol_pf = profile_fitting_blocking(instance)
    print(f"Profile Fitting:  {sol_pf.permutation}  Cmax={sol_pf.makespan}")

    # Random larger instance
    print("\n=== Random 20x5 Instance ===")
    rand_instance = BlockingFlowShopInstance.random(n=20, m=5, seed=42)

    sol_neh_r = neh_blocking(rand_instance)
    print(f"NEH-B  Makespan: {sol_neh_r.makespan}")

    sol_pf_r = profile_fitting_blocking(rand_instance)
    print(f"PF-B   Makespan: {sol_pf_r.makespan}")
