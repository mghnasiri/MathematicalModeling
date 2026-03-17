"""
Palmer's Slope Index — Earliest Flow Shop Heuristic (1965)

Assigns each job a "slope index" measuring the tendency of its processing
times across machines. Jobs with increasing processing times (heavier on
later machines) get higher priority and are scheduled first.

Algorithm:
    1. For each job j, compute the slope index:
       S_j = sum over i=1..m of [ -(m - (2i - 1)) * p_{ij} ]
       This gives positive weight to later machines and negative weight
       to earlier machines.
    2. Sort jobs by decreasing S_j.
    3. The sorted order is the heuristic solution.

Intuition:
    A job that takes longer on later machines should go early in the
    sequence — otherwise those later machines will be starved for work
    while waiting, then create a bottleneck at the end. The slope index
    captures this "increasing workload" tendency in a single number.

Notation: Fm | prmu | Cmax
Complexity: O(n*m + n*log(n)) — compute slopes, then sort.
Quality: Weakest of the classical heuristics. Typically 10-20% above optimal.
         Useful mainly as a fast initial bound or tie-breaker.
Reference: Palmer, D.S. (1965). "Sequencing Jobs Through a Multi-Stage
           Process in the Minimum Total Time — A Quick Method of Obtaining
           a Near Optimum"
"""

from __future__ import annotations
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from instance import FlowShopInstance, FlowShopSolution, compute_makespan


def palmers_slope(instance: FlowShopInstance) -> FlowShopSolution:
    """
    Apply Palmer's Slope Index heuristic.

    Args:
        instance: A FlowShopInstance with any number of machines.

    Returns:
        FlowShopSolution with the slope-ordered permutation and makespan.
    """
    p = instance.processing_times  # shape (m, n)
    m = instance.m
    n = instance.n

    # Compute machine weights: w_i = -(m - (2i - 1)) for i = 1..m
    # i=1 (first machine): w = -(m - 1)     → most negative (penalizes)
    # i=m (last machine):  w = +(m - 1)     → most positive (rewards)
    weights = [-(m - (2 * i - 1)) for i in range(1, m + 1)]

    # Compute slope index for each job
    slope_indices: list[tuple[float, int]] = []
    for j in range(n):
        s_j = sum(weights[i] * int(p[i, j]) for i in range(m))
        slope_indices.append((s_j, j))

    # Sort by decreasing slope index (highest priority first)
    slope_indices.sort(key=lambda x: x[0], reverse=True)
    permutation = [j for _, j in slope_indices]

    makespan = compute_makespan(instance, permutation)
    return FlowShopSolution(permutation=permutation, makespan=makespan)


if __name__ == "__main__":
    import numpy as np

    # Test with the standard 5×3 instance
    instance = FlowShopInstance(
        n=5, m=3,
        processing_times=np.array([
            [5, 9, 8, 10, 1],  # Machine 1  (weight = -2)
            [6, 3, 7, 2,  4],  # Machine 2  (weight =  0)
            [9, 7, 5, 4,  3],  # Machine 3  (weight = +2)
        ])
    )

    print("=== Palmer's Slope Index ===")
    sol = palmers_slope(instance)
    print(f"Permutation: {sol.permutation}")
    print(f"Makespan:    {sol.makespan}")

    # Show slope indices
    p = instance.processing_times
    weights = [-(3 - (2 * i - 1)) for i in range(1, 4)]
    print(f"\nWeights per machine: {weights}")
    for j in range(5):
        s = sum(weights[i] * int(p[i, j]) for i in range(3))
        print(f"  Job {j}: M1={p[0,j]:2d}, M2={p[1,j]:2d}, M3={p[2,j]:2d}  →  S={s:+d}")

    # Compare all heuristics
    print("\n=== Comparison on Random 20×5 ===")
    rand_instance = FlowShopInstance.random(n=20, m=5, seed=42)

    sol_palmer = palmers_slope(rand_instance)
    print(f"Palmer:  {sol_palmer.makespan}")

    from heuristics.cds import cds
    from heuristics.neh import neh

    sol_cds = cds(rand_instance)
    print(f"CDS:     {sol_cds.makespan}")

    sol_neh = neh(rand_instance)
    print(f"NEH:     {sol_neh.makespan}")
