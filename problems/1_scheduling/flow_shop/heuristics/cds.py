"""
CDS Heuristic — Campbell, Dudek & Smith (1970)

Extends Johnson's Rule to m-machine flow shops by constructing (m-1)
artificial 2-machine sub-problems and applying Johnson's Rule to each.

Algorithm:
    For k = 1, 2, ..., m-1:
        1. Construct a virtual 2-machine problem:
           - Virtual machine 1 time: a_j = sum(p[0..k-1, j])
           - Virtual machine 2 time: b_j = sum(p[m-k..m-1, j])
        2. Apply Johnson's Rule to the virtual instance.
        3. Evaluate the resulting permutation on the ORIGINAL m-machine instance.
    Return the best permutation found across all (m-1) iterations.

Intuition:
    At k=1, we look at just the first and last machines (the "endpoints").
    As k grows, we aggregate more machines into each virtual machine,
    capturing more of the flow shop structure. This is a systematic way
    to apply a 2-machine optimal algorithm to an m-machine problem.

Notation: Fm | prmu | Cmax
Complexity: O(m × n log n) — runs Johnson's Rule (m-1) times
Quality: Moderate — typically 5-10% above optimal. NEH is usually better,
         but CDS is simpler and faster.
Reference: Campbell, H.G., Dudek, R.A. & Smith, M.L. (1970). "A Heuristic
           Algorithm for the n Job, m Machine Sequencing Problem"
"""

from __future__ import annotations
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from instance import FlowShopInstance, FlowShopSolution, compute_makespan
from exact.johnsons_rule import johnsons_rule


def cds(instance: FlowShopInstance) -> FlowShopSolution:
    """
    Apply the CDS heuristic to a permutation flow shop instance.

    Generates (m-1) candidate solutions by applying Johnson's Rule
    to constructed 2-machine sub-problems, returns the best.

    Args:
        instance: A FlowShopInstance with m >= 2 machines.

    Returns:
        FlowShopSolution with the best permutation found.
    """
    p = instance.processing_times  # shape (m, n)
    m = instance.m
    n = instance.n

    best_permutation: list[int] = []
    best_makespan = float('inf')

    for k in range(1, m):
        # Construct virtual 2-machine instance:
        #   Virtual M1: sum of processing times on machines 0..k-1
        #   Virtual M2: sum of processing times on machines (m-k)..m-1
        virtual_m1 = p[:k, :].sum(axis=0)       # shape (n,)
        virtual_m2 = p[m - k:, :].sum(axis=0)   # shape (n,)

        virtual_instance = FlowShopInstance(
            n=n, m=2,
            processing_times=np.stack([virtual_m1, virtual_m2])
        )

        # Apply Johnson's Rule to the virtual instance
        virtual_solution = johnsons_rule(virtual_instance)
        perm = virtual_solution.permutation

        # Evaluate on the ORIGINAL instance
        ms = compute_makespan(instance, perm)

        if ms < best_makespan:
            best_makespan = ms
            best_permutation = perm

    return FlowShopSolution(
        permutation=best_permutation,
        makespan=int(best_makespan),
    )


if __name__ == "__main__":
    # Test with the same 5×3 instance as NEH
    instance = FlowShopInstance(
        n=5, m=3,
        processing_times=np.array([
            [5, 9, 8, 10, 1],  # Machine 1
            [6, 3, 7, 2,  4],  # Machine 2
            [9, 7, 5, 4,  3],  # Machine 3
        ])
    )

    print("=== CDS Heuristic ===")
    sol = cds(instance)
    print(f"Permutation: {sol.permutation}")
    print(f"Makespan:    {sol.makespan}")

    # Compare with NEH on a random instance
    print("\n=== Random 20x5 Instance ===")
    rand_instance = FlowShopInstance.random(n=20, m=5, seed=42)
    sol_cds = cds(rand_instance)
    print(f"CDS  Makespan: {sol_cds.makespan}")

    # Import NEH for comparison
    from heuristics.neh import neh
    sol_neh = neh(rand_instance)
    print(f"NEH  Makespan: {sol_neh.makespan}")
    print(f"NEH improves:  {sol_cds.makespan - sol_neh.makespan} "
          f"({(sol_cds.makespan - sol_neh.makespan) / sol_cds.makespan * 100:.1f}%)")
