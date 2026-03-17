"""
Johnson's Rule — Optimal Algorithm for F2 || Cmax

The optimal algorithm for the two-machine permutation flow shop makespan
problem. Runs in O(n log n) time.

Notation: F2 | | Cmax
Complexity: O(n log n)
Optimality: Guaranteed optimal for exactly 2 machines.
Reference: Johnson, S.M. (1954). "Optimal Two- and Three-Stage Production
           Schedules with Setup Times Included"
"""

from __future__ import annotations
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from instance import FlowShopInstance, FlowShopSolution, compute_makespan


def johnsons_rule(instance: FlowShopInstance) -> FlowShopSolution:
    """
    Apply Johnson's Rule to a 2-machine flow shop instance.

    Algorithm idea:
        Partition jobs into two sets based on which machine has the
        shorter processing time. Jobs where machine 1 is faster go
        first (sorted by machine 1 time ascending). Jobs where machine
        2 is faster go last (sorted by machine 2 time descending).

    Args:
        instance: A FlowShopInstance (must have m == 2).

    Returns:
        FlowShopSolution with the optimal permutation and makespan.

    Raises:
        ValueError: If instance does not have exactly 2 machines.
    """
    if instance.m != 2:
        raise ValueError(
            f"Johnson's Rule requires exactly 2 machines, got {instance.m}. "
            f"For m > 2, use the CDS heuristic which applies Johnson's Rule "
            f"to constructed 2-machine sub-problems."
        )

    p = instance.processing_times  # shape (2, n)

    # Step 1: Partition jobs into two groups
    group_u: list[int] = []  # Jobs where machine 1 is faster (or equal)
    group_v: list[int] = []  # Jobs where machine 2 is faster

    for j in range(instance.n):
        if p[0, j] <= p[1, j]:
            group_u.append(j)
        else:
            group_v.append(j)

    # Step 2: Sort U ascending by machine 1 time (feed machine 2 early)
    group_u.sort(key=lambda j: p[0, j])

    # Step 3: Sort V descending by machine 2 time (avoid M2 idle at end)
    group_v.sort(key=lambda j: p[1, j], reverse=True)

    # Step 4: Optimal sequence is U followed by V
    permutation = group_u + group_v

    makespan = compute_makespan(instance, permutation)
    return FlowShopSolution(permutation=permutation, makespan=makespan)


if __name__ == "__main__":
    # Example: 6 jobs, 2 machines (classic textbook example)
    # Jobs:     0    1    2    3    4    5
    # M1:       3    6    2    7    1    5
    # M2:       4    1    5    2    6    3

    import numpy as np

    instance = FlowShopInstance(
        n=6, m=2,
        processing_times=np.array([
            [3, 6, 2, 7, 1, 5],  # Machine 1
            [4, 1, 5, 2, 6, 3],  # Machine 2
        ])
    )

    solution = johnsons_rule(instance)
    print(f"Optimal permutation: {solution.permutation}")
    print(f"Optimal makespan:    {solution.makespan}")

    # Expected: Jobs where M1 <= M2: {0,2,4} sorted by M1 asc → [4,2,0]
    #           Jobs where M1 > M2:  {1,3,5} sorted by M2 desc → [5,3,1]
    #           Full sequence: [4, 2, 0, 5, 3, 1]
    #           Makespan: 25
