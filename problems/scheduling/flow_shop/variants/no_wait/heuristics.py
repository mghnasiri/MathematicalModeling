"""
Constructive Heuristics for No-Wait Flow Shop — Fm | prmu, no-wait | Cmax

This module provides constructive heuristics adapted for the no-wait
permutation flow shop scheduling problem. Under the no-wait constraint,
makespan depends on inter-job delays rather than the standard completion
time recursion, which fundamentally changes the problem structure.

Heuristics implemented:
    1. Nearest Neighbor (NN): Greedy TSP-like construction using the delay
       matrix. Extends the permutation by appending the unscheduled job with
       the smallest delay from the current last job.

    2. NEH Adaptation (NEH-NW): Adapts the NEH insertion heuristic to the
       no-wait setting. Jobs sorted by decreasing total processing time are
       inserted one by one into the position that minimizes the partial makespan.

    3. Gangadharan & Rajendran (GR): A composite index heuristic that uses
       weighted delay and processing time information.

References:
    Gangadharan, R. & Rajendran, C. (1993). "Heuristic Algorithms for
    Scheduling in the No-Wait Flowshop"
    International Journal of Production Economics, 32(3):285-290.
    DOI: 10.1016/0925-5273(93)90042-J

    Bertolissi, E. (2000). "Heuristic Algorithm for Scheduling in the
    No-Wait Flow-Shop"
    Journal of Materials Processing Technology, 107(1-3):459-465.
    DOI: 10.1016/S0924-0136(00)00720-2

    Nawaz, M., Enscore, E.E. & Ham, I. (1983). "A Heuristic Algorithm
    for the m-Machine, n-Job Flow-Shop Sequencing Problem"
    Omega, 11(1):91-95. [Original NEH — adapted here for no-wait]
"""

from __future__ import annotations
import sys
import os
import importlib.util
import numpy as np

# Use direct path-based import to avoid collision with flow_shop/instance.py
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


def nearest_neighbor_nw(
    instance: NoWaitFlowShopInstance,
    start_job: int | None = None,
) -> NoWaitFlowShopSolution:
    """
    Nearest Neighbor heuristic for the no-wait flow shop.

    Greedily builds a permutation by always appending the unscheduled job
    that introduces the smallest delay from the current last job. This is
    analogous to the nearest-neighbor heuristic for the Traveling Salesman
    Problem, leveraging the fact that no-wait makespan depends on
    inter-job delays.

    Args:
        instance: A NoWaitFlowShopInstance.
        start_job: Starting job index. If None, tries all starting jobs
                  and returns the best result.

    Returns:
        NoWaitFlowShopSolution with the constructed permutation and makespan.
    """
    n = instance.n
    D = compute_delay_matrix(instance)

    if start_job is not None:
        perm = _nn_from_start(n, D, start_job)
        ms = compute_makespan_nw(instance, perm, D)
        return NoWaitFlowShopSolution(permutation=perm, makespan=ms)

    # Try all starting jobs and return the best
    best_perm = None
    best_ms = float('inf')

    for s in range(n):
        perm = _nn_from_start(n, D, s)
        ms = compute_makespan_nw(instance, perm, D)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm

    return NoWaitFlowShopSolution(permutation=best_perm, makespan=best_ms)


def _nn_from_start(n: int, D: np.ndarray, start: int) -> list[int]:
    """Build a permutation using nearest neighbor from a given start job."""
    perm = [start]
    unscheduled = set(range(n)) - {start}

    while unscheduled:
        last = perm[-1]
        # Find the unscheduled job with minimum delay from last
        best_job = min(unscheduled, key=lambda j: D[last, j])
        perm.append(best_job)
        unscheduled.remove(best_job)

    return perm


def neh_no_wait(instance: NoWaitFlowShopInstance) -> NoWaitFlowShopSolution:
    """
    NEH heuristic adapted for the no-wait flow shop.

    Follows the same principle as the original NEH (Nawaz et al., 1983):
    sort jobs by decreasing total processing time, then insert each job
    into the position that minimizes the partial makespan. The key
    difference is that makespan is computed using the no-wait delay-based
    formula.

    Args:
        instance: A NoWaitFlowShopInstance.

    Returns:
        NoWaitFlowShopSolution with the constructed permutation and makespan.
    """
    n = instance.n
    p = instance.processing_times
    D = compute_delay_matrix(instance)

    # Sort jobs by decreasing total processing time
    total_times = p.sum(axis=0)
    sorted_jobs = sorted(range(n), key=lambda j: total_times[j], reverse=True)

    # Start with the first job
    sequence = [sorted_jobs[0]]

    # Insert each remaining job into the best position
    for i in range(1, len(sorted_jobs)):
        job = sorted_jobs[i]
        best_ms = float('inf')
        best_pos = 0

        for pos in range(len(sequence) + 1):
            candidate = sequence[:pos] + [job] + sequence[pos:]
            ms = compute_makespan_nw(instance, candidate, D)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos

        sequence.insert(best_pos, job)

    makespan = compute_makespan_nw(instance, sequence, D)
    return NoWaitFlowShopSolution(permutation=sequence, makespan=makespan)


def gangadharan_rajendran(
    instance: NoWaitFlowShopInstance,
) -> NoWaitFlowShopSolution:
    """
    Gangadharan & Rajendran (1993) heuristic for no-wait flow shop.

    Uses a composite priority index combining processing time information
    and delay structure. Jobs are sorted by a weighted index and then
    refined using NEH-style insertion.

    The priority index for each job j is:
        I_j = sum_{i=0}^{m-1} w_i * p[i][j]
    where w_i = (m - 2*i + 1) gives positive weight to early machines
    and negative weight to later machines (similar to Palmer's slope).

    After sorting, a second pass performs NEH-style insertion for the
    top-ranked jobs.

    Args:
        instance: A NoWaitFlowShopInstance.

    Returns:
        NoWaitFlowShopSolution with the constructed permutation and makespan.
    """
    n = instance.n
    m = instance.m
    p = instance.processing_times
    D = compute_delay_matrix(instance)

    # Compute priority index (slope-like)
    weights = [(m - 2 * i + 1) for i in range(m)]
    indices = []
    for j in range(n):
        idx = sum(weights[i] * int(p[i, j]) for i in range(m))
        indices.append((idx, j))

    # Sort by decreasing priority
    indices.sort(key=lambda x: x[0], reverse=True)
    sorted_jobs = [j for _, j in indices]

    # NEH-style insertion using sorted order
    sequence = [sorted_jobs[0]]

    for i in range(1, len(sorted_jobs)):
        job = sorted_jobs[i]
        best_ms = float('inf')
        best_pos = 0

        for pos in range(len(sequence) + 1):
            candidate = sequence[:pos] + [job] + sequence[pos:]
            ms = compute_makespan_nw(instance, candidate, D)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos

        sequence.insert(best_pos, job)

    makespan = compute_makespan_nw(instance, sequence, D)
    return NoWaitFlowShopSolution(permutation=sequence, makespan=makespan)


if __name__ == "__main__":
    # Test on a small instance
    instance = NoWaitFlowShopInstance(
        n=6, m=3,
        processing_times=np.array([
            [3, 5, 2, 7, 4, 6],  # Machine 0
            [4, 2, 6, 1, 3, 5],  # Machine 1
            [2, 3, 4, 5, 1, 2],  # Machine 2
        ])
    )

    print("=== No-Wait Flow Shop Heuristics ===")
    print(f"Instance: {instance.n} jobs, {instance.m} machines\n")

    sol_nn = nearest_neighbor_nw(instance)
    print(f"Nearest Neighbor:         {sol_nn.permutation}  Cmax={sol_nn.makespan}")

    sol_neh = neh_no_wait(instance)
    print(f"NEH (no-wait):            {sol_neh.permutation}  Cmax={sol_neh.makespan}")

    sol_gr = gangadharan_rajendran(instance)
    print(f"Gangadharan-Rajendran:    {sol_gr.permutation}  Cmax={sol_gr.makespan}")

    # Random larger instance
    print("\n=== Random 20x5 Instance ===")
    rand_instance = NoWaitFlowShopInstance.random(n=20, m=5, seed=42)

    sol_nn_r = nearest_neighbor_nw(rand_instance)
    print(f"NN   Makespan: {sol_nn_r.makespan}")

    sol_neh_r = neh_no_wait(rand_instance)
    print(f"NEH  Makespan: {sol_neh_r.makespan}")

    sol_gr_r = gangadharan_rajendran(rand_instance)
    print(f"GR   Makespan: {sol_gr_r.makespan}")
