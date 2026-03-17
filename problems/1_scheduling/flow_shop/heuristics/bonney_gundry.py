"""
Bonney-Gundry Slope Index — Priority Index Heuristic for Fm | prmu | Cmax.

The Bonney-Gundry heuristic (1976) computes a slope index for each job based
on cumulative processing time relationships. Unlike Palmer's linear weights,
it uses the ratio of processing times on later vs. earlier machines to capture
the "processing time gradient" more accurately.

Algorithm:
    1. For each job j, compute:
       BG_j = sum_{k=1}^{m-1} (m - 2k) * (p_{k,j} - p_{k+1,j})
       This is equivalent to a weighted sum of successive differences.
    2. Sort jobs by decreasing BG_j.
    3. The sorted order is the heuristic solution.

Intuition:
    Similar to Palmer's, but measures how processing times change across
    successive machine pairs rather than just position-weighting. This
    captures the "slope" of processing times more precisely.

Notation: Fm | prmu | Cmax
Complexity: O(n*m + n*log(n)) — compute indices, then sort.
Quality: Comparable to Palmer; sometimes slightly better on instances
         with strong processing-time gradients.

Reference:
    Bonney, M.C. & Gundry, S.W. (1976). Solutions to the constrained
    flowshop sequencing problem. Operational Research Quarterly, 27(4),
    869-883. https://doi.org/10.2307/3009405
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from instance import FlowShopInstance, FlowShopSolution, compute_makespan


def bonney_gundry(instance: FlowShopInstance) -> FlowShopSolution:
    """Apply the Bonney-Gundry Slope Index heuristic.

    Args:
        instance: A FlowShopInstance with any number of machines.

    Returns:
        FlowShopSolution with the priority-ordered permutation and makespan.
    """
    p = instance.processing_times  # shape (m, n)
    m = instance.m
    n = instance.n

    indices: list[tuple[float, int]] = []
    for j in range(n):
        bg_j = 0.0
        for k in range(m - 1):
            weight = m - 2 * (k + 1)
            bg_j += weight * (int(p[k, j]) - int(p[k + 1, j]))
        indices.append((bg_j, j))

    # Sort by decreasing index (highest priority first)
    indices.sort(key=lambda x: (-x[0], x[1]))
    permutation = [j for _, j in indices]

    makespan = compute_makespan(instance, permutation)
    return FlowShopSolution(permutation=permutation, makespan=makespan)


if __name__ == "__main__":
    import numpy as np

    instance = FlowShopInstance(
        n=5, m=3,
        processing_times=np.array([
            [5, 9, 8, 10, 1],
            [6, 3, 7, 2, 4],
            [9, 7, 5, 4, 3],
        ]),
    )

    print("=== Bonney-Gundry Slope Index ===")
    sol = bonney_gundry(instance)
    print(f"Permutation: {sol.permutation}")
    print(f"Makespan:    {sol.makespan}")

    from heuristics.palmers_slope import palmers_slope

    sol_p = palmers_slope(instance)
    print(f"\nPalmer:  {sol_p.makespan}")
    print(f"BG:      {sol.makespan}")
