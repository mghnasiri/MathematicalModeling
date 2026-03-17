"""
Constructive Heuristics for SDST Flow Shop — Fm | prmu, Ssd | Cmax

Heuristics adapted for the permutation flow shop with sequence-dependent
setup times. All algorithms use the SDST-aware makespan evaluation.

Heuristics implemented:
    1. NEH-SDST: NEH adaptation with setup-time-aware evaluation. Jobs are
       sorted by decreasing total processing time plus average setup time.

    2. GRASP-SDST: Greedy Randomized Adaptive Search Procedure. Constructs
       solutions by greedily inserting jobs with a randomized restricted
       candidate list (RCL), then applies local search.

References:
    Ruiz, R., Maroto, C. & Alcaraz, J. (2005). "Solving the Flowshop
    Scheduling Problem with Sequence Dependent Setup Times Using Advanced
    Metaheuristics"
    EJOR, 165(1):34-54.
    DOI: 10.1016/j.ejor.2004.01.022

    Rios-Mercado, R.Z. & Bard, J.F. (1998). "Computational Experience with a
    Branch-and-Cut Algorithm for Flowshop Scheduling with Setups"
    Computers & Operations Research, 25(5):351-366.
    DOI: 10.1016/S0305-0548(97)00079-8
"""

from __future__ import annotations
import sys
import os
import importlib.util
import numpy as np

# Use direct path-based import to avoid collision with flow_shop/instance.py
_this_dir = os.path.dirname(os.path.abspath(__file__))
_instance_path = os.path.join(_this_dir, "instance.py")
_spec = importlib.util.spec_from_file_location("sdst_instance", _instance_path)
_sdst_instance = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("sdst_instance", _sdst_instance)
_spec.loader.exec_module(_sdst_instance)

SDSTFlowShopInstance = _sdst_instance.SDSTFlowShopInstance
SDSTFlowShopSolution = _sdst_instance.SDSTFlowShopSolution
compute_makespan_sdst = _sdst_instance.compute_makespan_sdst


def neh_sdst(instance: SDSTFlowShopInstance) -> SDSTFlowShopSolution:
    """
    NEH heuristic adapted for the SDST flow shop.

    Jobs are sorted by decreasing total workload (processing time + average
    setup time across all machines). Each job is then inserted into the
    position that minimizes the partial SDST makespan.

    Args:
        instance: An SDSTFlowShopInstance.

    Returns:
        SDSTFlowShopSolution with the constructed permutation and makespan.
    """
    n = instance.n
    m = instance.m
    p = instance.processing_times
    s = instance.setup_times

    # Sort by decreasing total workload: processing time + average setup time
    # For each job j, compute p_total(j) + average setup to/from j
    workload = np.zeros(n)
    for j in range(n):
        proc_total = float(p[:, j].sum())
        # Average setup time when job j follows any other job
        avg_setup_to = float(s[:, :n, j].mean())
        # Average setup time when any job follows job j
        avg_setup_from = float(s[:, j, :].mean())
        workload[j] = proc_total + avg_setup_to + avg_setup_from

    sorted_jobs = sorted(range(n), key=lambda j: workload[j], reverse=True)

    # Build sequence using NEH insertion
    sequence = [sorted_jobs[0]]

    for i in range(1, len(sorted_jobs)):
        job = sorted_jobs[i]
        best_ms = float('inf')
        best_pos = 0

        for pos in range(len(sequence) + 1):
            candidate = sequence[:pos] + [job] + sequence[pos:]
            ms = compute_makespan_sdst(instance, candidate)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos

        sequence.insert(best_pos, job)

    makespan = compute_makespan_sdst(instance, sequence)
    return SDSTFlowShopSolution(permutation=sequence, makespan=makespan)


def grasp_sdst(
    instance: SDSTFlowShopInstance,
    alpha: float = 0.3,
    max_constructions: int = 10,
    seed: int | None = None,
) -> SDSTFlowShopSolution:
    """
    GRASP heuristic for the SDST flow shop.

    Constructs multiple solutions using a randomized greedy strategy and
    returns the best. At each step, a Restricted Candidate List (RCL) is
    built containing jobs whose insertion cost is within alpha of the best.

    Args:
        instance: An SDSTFlowShopInstance.
        alpha: RCL threshold (0 = greedy, 1 = fully random).
        max_constructions: Number of GRASP constructions.
        seed: Random seed for reproducibility.

    Returns:
        SDSTFlowShopSolution with the best permutation found.
    """
    rng = np.random.default_rng(seed)
    n = instance.n

    best_perm = None
    best_ms = float('inf')

    for _ in range(max_constructions):
        perm = _grasp_construct(instance, alpha, rng)
        # Apply local search
        perm = _local_search_sdst(instance, perm)
        ms = compute_makespan_sdst(instance, perm)

        if ms < best_ms:
            best_perm = list(perm)
            best_ms = ms

    return SDSTFlowShopSolution(permutation=best_perm, makespan=best_ms)


def _grasp_construct(
    instance: SDSTFlowShopInstance,
    alpha: float,
    rng: np.random.Generator,
) -> list[int]:
    """
    Construct a solution using GRASP's randomized greedy strategy.

    Args:
        instance: An SDSTFlowShopInstance.
        alpha: RCL threshold parameter.
        rng: Random number generator.

    Returns:
        Constructed permutation.
    """
    n = instance.n
    p = instance.processing_times

    # Sort by total processing time for initial ordering
    total_times = p.sum(axis=0)
    sorted_jobs = sorted(range(n), key=lambda j: total_times[j], reverse=True)

    # Start with first job (deterministic)
    sequence = [sorted_jobs[0]]
    unscheduled = set(sorted_jobs[1:])

    while unscheduled:
        # Evaluate inserting each unscheduled job at every position
        candidates = []
        for job in unscheduled:
            best_pos_ms = float('inf')
            best_pos = 0
            for pos in range(len(sequence) + 1):
                candidate = sequence[:pos] + [job] + sequence[pos:]
                ms = compute_makespan_sdst(instance, candidate)
                if ms < best_pos_ms:
                    best_pos_ms = ms
                    best_pos = pos
            candidates.append((job, best_pos, best_pos_ms))

        # Build RCL
        costs = [c[2] for c in candidates]
        min_cost = min(costs)
        max_cost = max(costs)
        threshold = min_cost + alpha * (max_cost - min_cost)

        rcl = [c for c in candidates if c[2] <= threshold]

        # Select randomly from RCL
        chosen = rcl[rng.integers(0, len(rcl))]
        job, pos, _ = chosen

        sequence.insert(pos, job)
        unscheduled.remove(job)

    return sequence


def _local_search_sdst(
    instance: SDSTFlowShopInstance,
    permutation: list[int],
) -> list[int]:
    """
    First-improvement insertion local search for SDST flow shop.

    Args:
        instance: An SDSTFlowShopInstance.
        permutation: Starting permutation.

    Returns:
        Locally optimal permutation.
    """
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
    print("=== SDST Flow Shop Heuristics ===")

    instance = SDSTFlowShopInstance.random(n=10, m=4, seed=42)
    print(f"Instance: {instance.n} jobs, {instance.m} machines\n")

    sol_neh = neh_sdst(instance)
    print(f"NEH-SDST:  {sol_neh.permutation}  Cmax={sol_neh.makespan}")

    sol_grasp = grasp_sdst(instance, max_constructions=5, seed=42)
    print(f"GRASP:     {sol_grasp.permutation}  Cmax={sol_grasp.makespan}")

    # Larger instance
    print("\n=== Random 20x5 Instance ===")
    rand_instance = SDSTFlowShopInstance.random(n=20, m=5, seed=42)

    sol_neh_r = neh_sdst(rand_instance)
    print(f"NEH-SDST Makespan: {sol_neh_r.makespan}")

    sol_grasp_r = grasp_sdst(rand_instance, max_constructions=10, seed=42)
    print(f"GRASP    Makespan: {sol_grasp_r.makespan}")
