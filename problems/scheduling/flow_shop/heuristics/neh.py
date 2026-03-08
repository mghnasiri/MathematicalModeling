"""
NEH Heuristic — Best Constructive Heuristic for Fm | prmu | Cmax

The NEH (Nawaz-Enscore-Ham, 1983) algorithm is widely considered the best
constructive heuristic for the permutation flow shop makespan problem.
After 40+ years, it remains competitive with many metaheuristics for
small-to-medium instances.

Algorithm:
    1. Sort jobs by decreasing total processing time (sum across all machines).
    2. Take the first two jobs, evaluate both orderings, keep the better one.
    3. For each remaining job (in sorted order):
       - Try inserting it into every position of the current partial sequence.
       - Keep the insertion that gives the smallest partial makespan.

Notation: Fm | prmu | Cmax
Complexity: O(n² × m) — n jobs, each tried in up to n positions, each
            evaluated in O(m) time.
Reference: Nawaz, M., Enscore, E.E. & Ham, I. (1983). "A Heuristic Algorithm
           for the m-Machine, n-Job Flow-Shop Sequencing Problem"
"""

from __future__ import annotations
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from instance import FlowShopInstance, FlowShopSolution, compute_makespan


def neh(instance: FlowShopInstance) -> FlowShopSolution:
    """
    Apply the NEH heuristic to a permutation flow shop instance.

    Args:
        instance: A FlowShopInstance with any number of machines.

    Returns:
        FlowShopSolution with the constructed permutation and makespan.
    """
    p = instance.processing_times  # shape (m, n)

    # Step 1: Sort jobs by decreasing total processing time.
    # Intuition: jobs with the most total work should be placed first
    # because they have the most impact on makespan. Placing them early
    # gives the algorithm maximum flexibility to find good positions.
    total_times = p.sum(axis=0)  # sum across machines for each job
    sorted_jobs = sorted(range(instance.n), key=lambda j: total_times[j],
                         reverse=True)

    # Step 2: Initialize with the first job
    sequence: list[int] = [sorted_jobs[0]]

    # Step 3: Insert each remaining job into the best position
    for i in range(1, len(sorted_jobs)):
        job = sorted_jobs[i]
        best_makespan = float('inf')
        best_position = 0

        # Try every insertion position (0 through len(sequence))
        for pos in range(len(sequence) + 1):
            candidate = sequence[:pos] + [job] + sequence[pos:]
            ms = compute_makespan(instance, candidate)

            if ms < best_makespan:
                best_makespan = ms
                best_position = pos

        # Insert at the best position found
        sequence.insert(best_position, job)

    makespan = compute_makespan(instance, sequence)
    return FlowShopSolution(permutation=sequence, makespan=makespan)


def neh_with_tiebreaking(instance: FlowShopInstance) -> FlowShopSolution:
    """
    NEH with Fernandez-Viagas & Framinan (2014) tie-breaking.

    When two insertion positions yield the same makespan, this variant
    breaks ties by choosing the position where the idle time on the
    last machine is minimized. This small change can improve results
    by 1-3% on average.

    Reference: Fernandez-Viagas, V. & Framinan, J.M. (2014).
               "On Insertion Tie-Breaking Rules in Heuristics for the
               Permutation Flowshop Scheduling Problem"
    """
    p = instance.processing_times
    m = instance.m

    total_times = p.sum(axis=0)
    sorted_jobs = sorted(range(instance.n), key=lambda j: total_times[j],
                         reverse=True)

    sequence: list[int] = [sorted_jobs[0]]

    for i in range(1, len(sorted_jobs)):
        job = sorted_jobs[i]
        best_makespan = float('inf')
        best_idle = float('inf')
        best_position = 0

        for pos in range(len(sequence) + 1):
            candidate = sequence[:pos] + [job] + sequence[pos:]
            ms = compute_makespan(instance, candidate)

            if ms < best_makespan:
                best_makespan = ms
                best_idle = _compute_total_idle(instance, candidate)
                best_position = pos
            elif ms == best_makespan:
                # Tie-break: prefer less idle time on the last machine
                idle = _compute_total_idle(instance, candidate)
                if idle < best_idle:
                    best_idle = idle
                    best_position = pos

        sequence.insert(best_position, job)

    makespan = compute_makespan(instance, sequence)
    return FlowShopSolution(permutation=sequence, makespan=makespan)


def _compute_total_idle(instance: FlowShopInstance,
                        permutation: list[int]) -> int:
    """
    Compute total idle time on the last machine.

    Idle time = sum of gaps where the last machine waits for the
    previous machine to finish before it can start the next job.
    """
    n_jobs = len(permutation)
    m = instance.m
    p = instance.processing_times

    # Compute full completion times
    import numpy as np
    C = np.zeros((m, n_jobs), dtype=int)

    for k in range(n_jobs):
        job = permutation[k]
        C[0, k] = (C[0, k - 1] if k > 0 else 0) + p[0, job]
        for i in range(1, m):
            prev_job = C[i, k - 1] if k > 0 else 0
            C[i, k] = max(C[i - 1, k], prev_job) + p[i, job]

    # Total idle on last machine
    total_idle = 0
    for k in range(n_jobs):
        start_time = C[m - 1, k] - p[m - 1, permutation[k]]
        prev_end = C[m - 1, k - 1] if k > 0 else 0
        total_idle += start_time - prev_end

    return total_idle


if __name__ == "__main__":
    import numpy as np

    # Test 1: Small instance (5 jobs, 3 machines)
    instance = FlowShopInstance(
        n=5, m=3,
        processing_times=np.array([
            [5, 9, 8, 10, 1],  # Machine 1
            [6, 3, 7, 2,  4],  # Machine 2
            [9, 7, 5, 4,  3],  # Machine 3
        ])
    )

    print("=== NEH Heuristic ===")
    sol = neh(instance)
    print(f"Permutation: {sol.permutation}")
    print(f"Makespan:    {sol.makespan}")

    print("\n=== NEH with Tie-Breaking ===")
    sol_tb = neh_with_tiebreaking(instance)
    print(f"Permutation: {sol_tb.permutation}")
    print(f"Makespan:    {sol_tb.makespan}")

    # Test 2: Random instance
    print("\n=== Random 20x5 Instance ===")
    rand_instance = FlowShopInstance.random(n=20, m=5, seed=42)
    sol_rand = neh(rand_instance)
    print(f"Permutation: {sol_rand.permutation}")
    print(f"Makespan:    {sol_rand.makespan}")

    sol_rand_tb = neh_with_tiebreaking(rand_instance)
    print(f"With TB:     {sol_rand_tb.makespan}")
