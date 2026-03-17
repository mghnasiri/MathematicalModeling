"""
LR Heuristic — Liu & Reeves (2001) Constructive Heuristic

A composite index-based heuristic that combines multiple evaluation criteria
to build a permutation from scratch. It generates n candidate solutions
(one per starting job) and returns the best.

Algorithm:
    For each candidate starting job j*:
        1. Place j* first in the sequence
        2. For each remaining position, evaluate all unscheduled jobs using a
           composite index that considers:
           - Machine idle time (I_j): idle time introduced on the last machine
           - Artificial completion time (A_j): estimated completion if job j
             were placed at the current position
           - Total remaining workload
        3. Select the job minimizing the weighted composite index
    Return the best of the n complete sequences.

The composite index for each unscheduled job j at position k:
    LR_j = x₁ · IT_j + x₂ · AT_j
where:
    IT_j = total idle time on the last machine if job j is placed at position k
    AT_j = artificial total completion time estimate
    x₁, x₂ = weights (typically x₁ = m-1, x₂ = 1, or variants)

Notation: Fm | prmu | Cmax
Complexity: O(n³·m) — n starting jobs × n insertions × n candidates × m machines.
Quality: Generally competitive with NEH. Sometimes better, sometimes slightly
         worse. The n-candidate approach provides diversity.
Reference: Liu, J. & Reeves, C.R. (2001). "Constructive and composite heuristic
           solutions to the P||Cmax problem"
           European Journal of Operational Research, 132(2):439-452.

Note: While the original paper targets P||Cmax, the LR index concept has been
      adapted to flow shop scheduling. This implementation follows the flow shop
      adaptation described in various comparative studies.
"""

from __future__ import annotations
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from instance import FlowShopInstance, FlowShopSolution, compute_makespan


def lr_heuristic(
    instance: FlowShopInstance,
    n_candidates: int | None = None,
) -> FlowShopSolution:
    """
    Apply the LR constructive heuristic to PFSP.

    Generates multiple candidate sequences (one per starting job or a subset)
    and returns the best. Each sequence is built greedily using a composite
    index that balances idle time minimization with completion time estimation.

    Args:
        instance: A FlowShopInstance.
        n_candidates: Number of starting jobs to try (default: all n).
                     Reducing this speeds up computation at the cost of quality.

    Returns:
        FlowShopSolution with the best found permutation.
    """
    n = instance.n
    m = instance.m
    p = instance.processing_times  # shape (m, n)

    if n_candidates is None:
        n_candidates = n
    n_candidates = min(n_candidates, n)

    # Rank starting jobs by total processing time (descending) — heavier jobs first
    total_times = [sum(int(p[i, j]) for i in range(m)) for j in range(n)]
    ranked_jobs = sorted(range(n), key=lambda j: total_times[j], reverse=True)
    start_jobs = ranked_jobs[:n_candidates]

    best_permutation = None
    best_makespan = float("inf")

    for start_job in start_jobs:
        permutation = [start_job]
        unscheduled = set(range(n)) - {start_job}

        # Completion times for current partial sequence: shape (m,)
        # C[i] = completion time on machine i of the last placed job
        C = np.zeros((m, 1), dtype=int)
        C[0, 0] = int(p[0, start_job])
        for i in range(1, m):
            C[i, 0] = C[i - 1, 0] + int(p[i, start_job])

        while unscheduled:
            best_job = None
            best_score = float("inf")

            for j in unscheduled:
                # Simulate placing job j next
                # Compute completion times if j is appended
                C_new = np.zeros(m, dtype=int)
                C_new[0] = int(C[0, -1]) + int(p[0, j])
                for i in range(1, m):
                    C_new[i] = max(C_new[i - 1], int(C[i, -1])) + int(p[i, j])

                # Component 1: Idle time on last machine
                # = C_new[m-1] - C[m-1, -1] - p[m-1, j]
                # This is the gap on the last machine before job j starts
                idle_time = int(C_new[m - 1]) - int(C[m - 1, -1]) - int(p[m - 1, j])

                # Component 2: Artificial completion time estimate
                # Sum of remaining workload on last machine for unscheduled jobs
                remaining = sum(int(p[m - 1, r]) for r in unscheduled if r != j)
                # Estimate: current completion + remaining work
                artificial_ct = int(C_new[m - 1]) + remaining

                # Composite index: weight idle time more for more machines
                # Following the LR spirit: minimize idle + consider total time
                score = (m - 1) * idle_time + artificial_ct

                if score < best_score:
                    best_score = score
                    best_job = j
                    best_C = C_new

            # Place the best job
            permutation.append(best_job)
            unscheduled.remove(best_job)

            # Update completion matrix
            C = np.column_stack([C, best_C.reshape(m, 1)])

        ms = compute_makespan(instance, permutation)
        if ms < best_makespan:
            best_makespan = ms
            best_permutation = permutation

    return FlowShopSolution(permutation=best_permutation, makespan=best_makespan)


if __name__ == "__main__":
    # Test on the standard 5×3 instance
    instance = FlowShopInstance(
        n=5, m=3,
        processing_times=np.array([
            [5, 9, 8, 10, 1],
            [6, 3, 7, 2,  4],
            [9, 7, 5, 4,  3],
        ])
    )

    print("=== LR Heuristic ===")
    sol = lr_heuristic(instance)
    print(f"Permutation: {sol.permutation}")
    print(f"Makespan:    {sol.makespan}")

    # Compare with other heuristics on random instance
    print("\n=== Comparison on Random 20×5 ===")
    rand_instance = FlowShopInstance.random(n=20, m=5, seed=42)

    sol_lr = lr_heuristic(rand_instance)
    print(f"LR:      {sol_lr.makespan}")

    from heuristics.palmers_slope import palmers_slope
    from heuristics.guptas_algorithm import guptas_algorithm
    from heuristics.cds import cds
    from heuristics.neh import neh

    print(f"Palmer:  {palmers_slope(rand_instance).makespan}")
    print(f"Gupta:   {guptas_algorithm(rand_instance).makespan}")
    print(f"CDS:     {cds(rand_instance).makespan}")
    print(f"NEH:     {neh(rand_instance).makespan}")

    # Test with limited candidates
    print("\n=== LR with 5 candidates (faster) ===")
    sol_lr5 = lr_heuristic(rand_instance, n_candidates=5)
    print(f"LR(5):   {sol_lr5.makespan}")
