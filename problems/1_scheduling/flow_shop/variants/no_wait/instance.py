"""
No-Wait Flow Shop Instance & Solution Data Structures

In the no-wait flow shop problem, each job must be processed on all machines
consecutively without any waiting time between machines. This means the start
time of a job on any machine is completely determined by its start time on the
first machine.

The key difference from the standard PFSP is that idle time on machines may
occur (machines wait for jobs), but jobs never wait between machines.

The no-wait constraint transforms the problem structure: makespan computation
depends on delay values d(j, k) between consecutive jobs, where d(j, k) is
the minimum delay between starting job j and starting job k such that no
machine overlap occurs.

Reference: Allahverdi, A. & Aldowaisan, T. (2004). "No-wait flowshops with
           bicriteria of makespan and maximum lateness"
           European Journal of Operational Research, 152(1):132-147.
           DOI: 10.1016/S0377-2217(02)00646-X

           Bertolissi, E. (2000). "Heuristic Algorithm for Scheduling in the
           No-Wait Flow-Shop"
           Journal of Materials Processing Technology, 107(1-3):459-465.
           DOI: 10.1016/S0924-0136(00)00720-2
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class NoWaitFlowShopInstance:
    """
    A no-wait permutation flow shop instance.

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
               seed: int | None = None) -> NoWaitFlowShopInstance:
        """Generate a random no-wait flow shop instance."""
        rng = np.random.default_rng(seed)
        processing_times = rng.integers(low, high + 1, size=(m, n))
        return cls(n=n, m=m, processing_times=processing_times)


@dataclass
class NoWaitFlowShopSolution:
    """
    A no-wait flow shop solution.

    Attributes:
        permutation: Job processing order (list of job indices).
        makespan: The Cmax value of this solution.
    """
    permutation: list[int]
    makespan: int

    def __repr__(self) -> str:
        return (f"NoWaitFlowShopSolution(makespan={self.makespan}, "
                f"permutation={self.permutation})")


def compute_delay(instance: NoWaitFlowShopInstance,
                  job_j: int, job_k: int) -> int:
    """
    Compute the minimum delay d(j, k) between starting job j and starting
    job k under the no-wait constraint.

    The delay ensures that on every machine, job k does not start before
    job j finishes. Due to the no-wait constraint, the start time of a job
    on machine i is fixed relative to its start on machine 0:
        start(job, i) = start(job, 0) + sum(p[0..i-1, job])

    Therefore:
        d(j, k) = max over i of [ sum(p[0..i, j]) - sum(p[0..i-1, k]) ]
    which simplifies to:
        d(j, k) = max over i of [ C_j[i] - S_k[i] ]
    where C_j[i] is the cumulative processing time of job j through machine i,
    and S_k[i] is the cumulative processing time of job k through machine i-1.

    Args:
        instance: A NoWaitFlowShopInstance.
        job_j: The preceding job index.
        job_k: The succeeding job index.

    Returns:
        Minimum delay (start time gap) between job j and job k.
    """
    m = instance.m
    p = instance.processing_times

    # Cumulative processing times
    cum_j = 0  # cumulative through machine i for job j
    cum_k = 0  # cumulative through machine i-1 for job k
    max_delay = 0

    for i in range(m):
        cum_j += int(p[i, job_j])
        # On machine i, job k cannot start until job j finishes
        # Delay needed = cum_j - cum_k (cum_k is sum of p[0..i-1, k])
        delay = cum_j - cum_k
        if delay > max_delay:
            max_delay = delay
        cum_k += int(p[i, job_k])

    return max_delay


def compute_delay_matrix(instance: NoWaitFlowShopInstance) -> np.ndarray:
    """
    Precompute the delay matrix D where D[j][k] = d(j, k).

    This matrix is asymmetric (D[j][k] != D[k][j] in general).
    The makespan of a permutation pi is:
        Cmax = sum_{i=0}^{n-2} D[pi[i]][pi[i+1]] + sum_{i=0}^{m-1} p[i, pi[n-1]]

    Args:
        instance: A NoWaitFlowShopInstance.

    Returns:
        np.ndarray of shape (n, n) with delay values.
    """
    n = instance.n
    D = np.zeros((n, n), dtype=int)

    for j in range(n):
        for k in range(n):
            if j != k:
                D[j, k] = compute_delay(instance, j, k)

    return D


def compute_makespan_nw(instance: NoWaitFlowShopInstance,
                        permutation: list[int],
                        delay_matrix: np.ndarray | None = None) -> int:
    """
    Compute the makespan of a permutation under the no-wait constraint.

    Cmax = sum of delays between consecutive jobs + total processing time
    of the last job across all machines.

    Args:
        instance: A NoWaitFlowShopInstance.
        permutation: Job processing order.
        delay_matrix: Precomputed delay matrix (optional, computed if None).

    Returns:
        The makespan value.
    """
    if len(permutation) == 0:
        return 0

    if delay_matrix is None:
        delay_matrix = compute_delay_matrix(instance)

    n_jobs = len(permutation)
    p = instance.processing_times

    # Start time of first job is 0
    # Start time of job at position k = sum of delays d(pi[0], pi[1]) + ... + d(pi[k-1], pi[k])
    total_delay = 0
    for k in range(n_jobs - 1):
        total_delay += delay_matrix[permutation[k], permutation[k + 1]]

    # Makespan = start time of last job + total processing time of last job
    last_job = permutation[-1]
    last_job_total_time = int(p[:, last_job].sum())

    return total_delay + last_job_total_time


if __name__ == "__main__":
    # Example: 4 jobs, 3 machines
    instance = NoWaitFlowShopInstance(
        n=4, m=3,
        processing_times=np.array([
            [3, 5, 2, 7],  # Machine 0
            [4, 2, 6, 1],  # Machine 1
            [2, 3, 4, 5],  # Machine 2
        ])
    )

    print("=== No-Wait Flow Shop ===")
    print(f"Instance: {instance.n} jobs, {instance.m} machines")

    # Compute delay matrix
    D = compute_delay_matrix(instance)
    print(f"\nDelay matrix D[j][k]:")
    for j in range(instance.n):
        row = " ".join(f"{D[j, k]:3d}" for k in range(instance.n))
        print(f"  Job {j}: [{row}]")

    # Evaluate a few permutations
    for perm in [[0, 1, 2, 3], [3, 0, 2, 1], [1, 0, 3, 2]]:
        ms = compute_makespan_nw(instance, perm, D)
        print(f"\nPermutation {perm}: makespan = {ms}")
