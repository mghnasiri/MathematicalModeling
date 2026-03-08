"""
Flow Shop Instance & Solution Data Structures

These are the shared data types used by all flow shop algorithms.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class FlowShopInstance:
    """
    A permutation flow shop instance.

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
    def from_file(cls, filepath: str) -> FlowShopInstance:
        """
        Load a Taillard-format instance.

        Format:
            n m
            p_11 p_12 ... p_1n   (machine 1)
            p_21 p_22 ... p_2n   (machine 2)
            ...
            p_m1 p_m2 ... p_mn   (machine m)
        """
        with open(filepath, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        n, m = map(int, lines[0].split())
        processing_times = np.zeros((m, n), dtype=int)

        for i in range(m):
            values = list(map(int, lines[1 + i].split()))
            processing_times[i] = values

        return cls(n=n, m=m, processing_times=processing_times)

    @classmethod
    def random(cls, n: int, m: int, low: int = 1, high: int = 99,
               seed: int | None = None) -> FlowShopInstance:
        """Generate a random instance (Taillard-style: U[low, high])."""
        rng = np.random.default_rng(seed)
        processing_times = rng.integers(low, high + 1, size=(m, n))
        return cls(n=n, m=m, processing_times=processing_times)


@dataclass
class FlowShopSolution:
    """
    A permutation flow shop solution.

    Attributes:
        permutation: Job processing order (list of job indices).
        makespan: The Cmax value of this solution.
        completion_times: Optional full completion time matrix (m x n_scheduled).
    """
    permutation: list[int]
    makespan: int
    completion_times: np.ndarray | None = None

    def __repr__(self) -> str:
        return (f"FlowShopSolution(makespan={self.makespan}, "
                f"permutation={self.permutation})")


def compute_makespan(instance: FlowShopInstance,
                     permutation: list[int]) -> int:
    """
    Compute the makespan (Cmax) of a permutation on the given instance.

    Uses the standard completion time recursion:
        C[i][k] = max(C[i-1][k], C[i][k-1]) + p[i][π(k)]

    Args:
        instance: The flow shop instance.
        permutation: Job order (list of job indices).

    Returns:
        The makespan value.
    """
    n_jobs = len(permutation)
    m = instance.m
    p = instance.processing_times

    # C[i] tracks completion time on machine i for the current job
    # We only need the previous job's completion times (space optimization)
    completion = np.zeros(m, dtype=int)

    for k in range(n_jobs):
        job = permutation[k]
        # Machine 0: just accumulates
        completion[0] += p[0, job]
        # Machines 1..m-1: max of (finished on prev machine, prev job on this machine)
        for i in range(1, m):
            completion[i] = max(completion[i - 1], completion[i]) + p[i, job]

    return int(completion[-1])


def compute_completion_times(instance: FlowShopInstance,
                             permutation: list[int]) -> np.ndarray:
    """
    Compute the full completion time matrix C[i][k] for a permutation.

    Returns:
        np.ndarray of shape (m, len(permutation)) with all completion times.
    """
    n_jobs = len(permutation)
    m = instance.m
    p = instance.processing_times
    C = np.zeros((m, n_jobs), dtype=int)

    for k in range(n_jobs):
        job = permutation[k]
        if k == 0 and 0 == 0:
            C[0, 0] = p[0, job]
        elif k > 0:
            C[0, k] = C[0, k - 1] + p[0, job]

        for i in range(1, m):
            prev_machine = C[i - 1, k]
            prev_job = C[i, k - 1] if k > 0 else 0
            C[i, k] = max(prev_machine, prev_job) + p[i, job]

    return C
