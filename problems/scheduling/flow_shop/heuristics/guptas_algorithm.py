"""
Gupta's Algorithm — Flow Shop Heuristic (1971)

A constructive heuristic for Fm | prmu | Cmax that generalizes Johnson's Rule
to m machines. Each job receives a priority index, and jobs are sorted by
decreasing priority.

Algorithm:
    1. For each job j, compute the Gupta index:
       - e_j = sign(min(p[0,j], p[m-1,j]))   (usually: sign favoring shorter
         first-machine processing, but the original uses a different convention)
       - Actually, Gupta defines:
           e_j = +1  if p[0,j] < p[m-1,j]
           e_j = -1  otherwise
       - s_j = e_j / min over i=0..m-2 of (p[i,j] + p[i+1,j])
    2. Sort jobs by decreasing s_j.
    3. The sorted order is the heuristic solution.

Intuition:
    The index combines two ideas:
    - The sign e_j captures whether a job "prefers" to go early (like Johnson's
      U-group for 2 machines) or late (V-group).
    - The denominator min(p[i,j] + p[i+1,j]) approximates the tightest
      "bottleneck" between consecutive machine pairs. Jobs with a tighter
      bottleneck need more careful positioning.

    By combining direction (sign) with bottleneck severity (min sum),
    Gupta creates a single priority score that works across m machines.

Notation: Fm | prmu | Cmax
Complexity: O(n*m + n*log(n)) — compute indices, then sort.
Quality: Generally between Palmer and CDS. Better than Palmer, sometimes
         competitive with CDS. Quick and simple.
Reference: Gupta, J.N.D. (1971). "A Functional Heuristic Algorithm for the
           Flow-Shop Scheduling Problem"
           Operations Research, 19(4):839-847.
"""

from __future__ import annotations
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from instance import FlowShopInstance, FlowShopSolution, compute_makespan


def guptas_algorithm(instance: FlowShopInstance) -> FlowShopSolution:
    """
    Apply Gupta's heuristic to the permutation flow shop.

    Args:
        instance: A FlowShopInstance with m >= 2 machines.

    Returns:
        FlowShopSolution with the Gupta-ordered permutation and makespan.
    """
    p = instance.processing_times  # shape (m, n)
    m = instance.m
    n = instance.n

    gupta_indices: list[tuple[float, int]] = []
    for j in range(n):
        # Sign: +1 if job "prefers" to go early (faster on first machine)
        e_j = 1 if p[0, j] < p[m - 1, j] else -1

        # Minimum sum of consecutive machine processing times
        min_sum = min(int(p[i, j]) + int(p[i + 1, j]) for i in range(m - 1))

        # Gupta index: direction / bottleneck severity
        # Avoid division by zero (though processing times should be > 0)
        s_j = e_j / max(min_sum, 1e-10)
        gupta_indices.append((s_j, j))

    # Sort by decreasing Gupta index
    gupta_indices.sort(key=lambda x: x[0], reverse=True)
    permutation = [j for _, j in gupta_indices]

    makespan = compute_makespan(instance, permutation)
    return FlowShopSolution(permutation=permutation, makespan=makespan)


if __name__ == "__main__":
    import numpy as np

    # Test on the standard 5×3 instance
    instance = FlowShopInstance(
        n=5, m=3,
        processing_times=np.array([
            [5, 9, 8, 10, 1],  # Machine 1
            [6, 3, 7, 2,  4],  # Machine 2
            [9, 7, 5, 4,  3],  # Machine 3
        ])
    )

    print("=== Gupta's Algorithm ===")
    sol = guptas_algorithm(instance)
    print(f"Permutation: {sol.permutation}")
    print(f"Makespan:    {sol.makespan}")

    # Show Gupta indices
    p = instance.processing_times
    m = 3
    print("\nGupta Index Details:")
    for j in range(5):
        e = 1 if p[0, j] < p[m - 1, j] else -1
        min_sum = min(int(p[i, j]) + int(p[i + 1, j]) for i in range(m - 1))
        s = e / max(min_sum, 1e-10)
        print(f"  Job {j}: M1={p[0,j]:2d}, M2={p[1,j]:2d}, M3={p[2,j]:2d}  "
              f"e={e:+d}, min_sum={min_sum}, s={s:+.4f}")

    # Compare with other heuristics on random instance
    print("\n=== Comparison on Random 20×5 ===")
    rand_instance = FlowShopInstance.random(n=20, m=5, seed=42)

    sol_gupta = guptas_algorithm(rand_instance)
    print(f"Gupta:   {sol_gupta.makespan}")

    from heuristics.palmers_slope import palmers_slope
    from heuristics.cds import cds
    from heuristics.neh import neh

    print(f"Palmer:  {palmers_slope(rand_instance).makespan}")
    print(f"CDS:     {cds(rand_instance).makespan}")
    print(f"NEH:     {neh(rand_instance).makespan}")
