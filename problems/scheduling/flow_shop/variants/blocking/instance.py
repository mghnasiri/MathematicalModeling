"""
Blocking Flow Shop Instance & Solution Data Structures

In the blocking flow shop, there are no intermediate buffers between machines.
When a job finishes processing on machine i, it remains on that machine
(blocking it) until machine i+1 becomes available. This creates additional
dependencies between jobs compared to the standard PFSP.

The key difference from the standard PFSP is in the completion time recursion:
    Standard PFSP: C[i][k] = max(C[i-1][k], C[i][k-1]) + p[i][pi(k)]
    Blocking PFSP: C[i][k] = max(C[i-1][k], D[i][k-1]) + p[i][pi(k)]
    where D[i][k] = max(C[i][k], D[i+1][k])  (departure time)

The departure time D[i][k] accounts for the blocking: a job cannot leave
machine i until it can start on machine i+1, and the next job cannot start
on machine i until the current job departs.

Reference: Ronconi, D.P. (2004). "A Note on Constructive Heuristics for the
           Flowshop Problem with Blocking"
           International Journal of Production Economics, 87(1):39-48.
           DOI: 10.1016/S0925-5273(03)00065-3

           Grabowski, J. & Pempera, J. (2007). "The Permutation Flow Shop
           Problem with Blocking. A Tabu Search Approach"
           Omega, 35(3):302-311.
           DOI: 10.1016/j.omega.2005.07.004
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class BlockingFlowShopInstance:
    """
    A blocking permutation flow shop instance.

    Attributes:
        n: Number of jobs.
        m: Number of machines.
        processing_times: Matrix of shape (m, n) where processing_times[i][j]
                         is the processing time of job j on machine i.
    """
    n: int
    m: int
    processing_times: np.ndarray  # shape (m, n)

    def __post_init__(self):
        assert self.processing_times.shape == (self.m, self.n), (
            f"Expected shape ({self.m}, {self.n}), "
            f"got {self.processing_times.shape}"
        )

    @classmethod
    def random(cls, n: int, m: int, low: int = 1, high: int = 99,
               seed: int | None = None) -> BlockingFlowShopInstance:
        """Generate a random blocking flow shop instance."""
        rng = np.random.default_rng(seed)
        processing_times = rng.integers(low, high + 1, size=(m, n))
        return cls(n=n, m=m, processing_times=processing_times)


@dataclass
class BlockingFlowShopSolution:
    """
    A blocking flow shop solution.

    Attributes:
        permutation: Job processing order (list of job indices).
        makespan: The Cmax value of this solution.
    """
    permutation: list[int]
    makespan: int

    def __repr__(self) -> str:
        return (f"BlockingFlowShopSolution(makespan={self.makespan}, "
                f"permutation={self.permutation})")


def compute_makespan_blocking(
    instance: BlockingFlowShopInstance,
    permutation: list[int],
) -> int:
    """
    Compute the makespan of a permutation under the blocking constraint.

    In the blocking model, we track both completion times C[i][k] and
    departure times D[i][k]. A job departs machine i only when machine i+1
    is free, which may be later than its completion on machine i.

    The recursion is:
        C[i][k] = max(C[i-1][k], D[i][k-1]) + p[i][pi(k)]
        D[i][k] = max(C[i][k], D[i+1][k])  for i < m-1
        D[m-1][k] = C[m-1][k]               (no blocking on last machine)

    where D[i][k-1] represents when machine i becomes available (after the
    previous job departs).

    Args:
        instance: A BlockingFlowShopInstance.
        permutation: Job processing order.

    Returns:
        The makespan value.
    """
    if len(permutation) == 0:
        return 0

    n_jobs = len(permutation)
    m = instance.m
    p = instance.processing_times

    # Completion time and departure time matrices
    C = np.zeros((m, n_jobs), dtype=int)
    D = np.zeros((m, n_jobs), dtype=int)

    for k in range(n_jobs):
        job = permutation[k]

        # Compute completion times
        for i in range(m):
            if i == 0 and k == 0:
                C[i, k] = int(p[i, job])
            elif i == 0:
                C[i, k] = D[i, k - 1] + int(p[i, job])
            elif k == 0:
                C[i, k] = C[i - 1, k] + int(p[i, job])
            else:
                C[i, k] = max(C[i - 1, k], D[i, k - 1]) + int(p[i, job])

        # Compute departure times (backwards from last machine)
        D[m - 1, k] = C[m - 1, k]
        for i in range(m - 2, -1, -1):
            D[i, k] = max(C[i, k], D[i + 1, k])

    return int(C[m - 1, n_jobs - 1])


if __name__ == "__main__":
    # Example: 4 jobs, 3 machines
    instance = BlockingFlowShopInstance(
        n=4, m=3,
        processing_times=np.array([
            [3, 5, 2, 7],  # Machine 0
            [4, 2, 6, 1],  # Machine 1
            [2, 3, 4, 5],  # Machine 2
        ])
    )

    print("=== Blocking Flow Shop ===")
    print(f"Instance: {instance.n} jobs, {instance.m} machines")

    # Compare blocking vs standard makespan
    import sys
    import os
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', '..'
    ))
    from instance import FlowShopInstance, compute_makespan

    standard_instance = FlowShopInstance(
        n=instance.n, m=instance.m,
        processing_times=instance.processing_times
    )

    for perm in [[0, 1, 2, 3], [3, 0, 2, 1], [1, 0, 3, 2]]:
        ms_std = compute_makespan(standard_instance, perm)
        ms_blk = compute_makespan_blocking(instance, perm)
        print(f"\nPermutation {perm}:")
        print(f"  Standard makespan: {ms_std}")
        print(f"  Blocking makespan: {ms_blk}  (diff: +{ms_blk - ms_std})")
