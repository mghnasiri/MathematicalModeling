"""
Dannenbring's Rapid Access (RA) Heuristic — Flow Shop Scheduling (1977)

A constructive heuristic for Fm | prmu | Cmax that reduces the m-machine
problem to a single virtual 2-machine problem using weighted processing
times, then applies Johnson's Rule.

Algorithm:
    1. Construct a virtual 2-machine problem with weighted aggregation:
       - Virtual machine 1 time for job j:
           a_j = sum over i=1..m of [ (m - i + 1) * p[i][j] ]
       - Virtual machine 2 time for job j:
           b_j = sum over i=1..m of [ i * p[i][j] ]
    2. Apply Johnson's Rule to the virtual 2-machine instance.
    3. Evaluate the resulting permutation on the original m-machine instance.

Intuition:
    Unlike CDS, which generates (m-1) candidate solutions from separate
    2-machine sub-problems, Dannenbring's RA creates a single virtual
    2-machine problem by weighting all machines simultaneously. Early
    machines receive higher weight in the virtual first machine, and later
    machines receive higher weight in the virtual second machine. This
    captures the "slope" of the processing time profile in a way that
    Johnson's Rule can exploit optimally.

Comparison with other heuristics:
    - Faster than CDS (single Johnson's application vs. m-1)
    - Generally weaker than CDS and NEH in solution quality
    - Better than Palmer in most cases
    - Serves as a quick initial bound or starting solution

Notation: Fm | prmu | Cmax
Complexity: O(n*m + n*log(n)) — compute weights, then one Johnson sort
Reference: Dannenbring, D.G. (1977). "An Evaluation of Flow Shop
           Sequencing Heuristics"
           Management Science, 23(11):1174-1182.
           DOI: 10.1287/mnsc.23.11.1174
"""

from __future__ import annotations
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from instance import FlowShopInstance, FlowShopSolution, compute_makespan
from exact.johnsons_rule import johnsons_rule


def dannenbring_ra(instance: FlowShopInstance) -> FlowShopSolution:
    """
    Apply Dannenbring's Rapid Access heuristic to a flow shop instance.

    Constructs a single virtual 2-machine problem using linearly weighted
    processing times, solves it optimally with Johnson's Rule, and evaluates
    the resulting permutation on the original instance.

    Args:
        instance: A FlowShopInstance with m >= 2 machines.

    Returns:
        FlowShopSolution with the RA permutation and its makespan.
    """
    p = instance.processing_times  # shape (m, n)
    m = instance.m
    n = instance.n

    # Compute weighted virtual processing times
    # Virtual M1: a_j = sum_{i=0}^{m-1} (m - i) * p[i][j]
    #   (weight decreases from m to 1 as machine index increases)
    # Virtual M2: b_j = sum_{i=0}^{m-1} (i + 1) * p[i][j]
    #   (weight increases from 1 to m as machine index increases)
    virtual_m1 = np.zeros(n, dtype=float)
    virtual_m2 = np.zeros(n, dtype=float)

    for i in range(m):
        weight_m1 = m - i       # m, m-1, ..., 1
        weight_m2 = i + 1       # 1, 2, ..., m
        virtual_m1 += weight_m1 * p[i, :]
        virtual_m2 += weight_m2 * p[i, :]

    # Create virtual 2-machine instance and solve with Johnson's Rule
    virtual_instance = FlowShopInstance(
        n=n, m=2,
        processing_times=np.stack([virtual_m1, virtual_m2]).astype(int)
    )

    virtual_solution = johnsons_rule(virtual_instance)
    permutation = virtual_solution.permutation

    # Evaluate on the original instance
    makespan = compute_makespan(instance, permutation)

    return FlowShopSolution(permutation=permutation, makespan=makespan)


if __name__ == "__main__":
    # Test on the standard 5x3 instance
    instance = FlowShopInstance(
        n=5, m=3,
        processing_times=np.array([
            [5, 9, 8, 10, 1],  # Machine 1
            [6, 3, 7, 2,  4],  # Machine 2
            [9, 7, 5, 4,  3],  # Machine 3
        ])
    )

    print("=== Dannenbring's Rapid Access ===")
    sol = dannenbring_ra(instance)
    print(f"Permutation: {sol.permutation}")
    print(f"Makespan:    {sol.makespan}")

    # Show virtual processing times
    p = instance.processing_times
    m = 3
    print("\nVirtual 2-machine instance:")
    for j in range(5):
        a = sum((m - i) * int(p[i, j]) for i in range(m))
        b = sum((i + 1) * int(p[i, j]) for i in range(m))
        print(f"  Job {j}: a={a:3d}, b={b:3d}")

    # Compare with other heuristics on random instance
    print("\n=== Comparison on Random 20x5 ===")
    rand_instance = FlowShopInstance.random(n=20, m=5, seed=42)

    sol_ra = dannenbring_ra(rand_instance)
    print(f"Dannenbring RA:  {sol_ra.makespan}")

    from heuristics.palmers_slope import palmers_slope
    from heuristics.cds import cds
    from heuristics.neh import neh

    print(f"Palmer:          {palmers_slope(rand_instance).makespan}")
    print(f"CDS:             {cds(rand_instance).makespan}")
    print(f"NEH:             {neh(rand_instance).makespan}")
