"""Batch Scheduling Problem instance and solution definitions.

Problem: n jobs on a single machine, each belonging to a job family.
Jobs in the same family can be batched together. A setup time is incurred
when switching between families. Minimize total weighted completion time.

Notation: 1 | batch, s_fam | Sigma(wj * Cj)
Complexity: NP-hard

References:
    Potts, C. N., & Kovalyov, M. Y. (2000). Scheduling with batching:
    A review. European Journal of Operational Research, 120(2), 228-249.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class BatchSchedulingInstance:
    """Batch scheduling problem instance.

    Attributes:
        n_jobs: Number of jobs.
        processing_times: Processing time for each job.
        weights: Weight (importance) of each job.
        families: Family index for each job.
        setup_time: Setup time between different families.
        n_families: Number of distinct families.
    """
    n_jobs: int
    processing_times: np.ndarray
    weights: np.ndarray
    families: np.ndarray
    setup_time: float
    n_families: int

    @classmethod
    def random(cls, n_jobs: int = 12, n_families: int = 3,
               setup_time: float = 5.0, seed: int = 42) -> BatchSchedulingInstance:
        """Generate a random batch scheduling instance.

        Args:
            n_jobs: Number of jobs.
            n_families: Number of job families.
            setup_time: Setup time between families.
            seed: Random seed.

        Returns:
            A random BatchSchedulingInstance.
        """
        rng = np.random.default_rng(seed)
        processing_times = rng.uniform(1.0, 10.0, size=n_jobs)
        weights = rng.uniform(1.0, 5.0, size=n_jobs)
        families = rng.integers(0, n_families, size=n_jobs)

        return cls(n_jobs=n_jobs, processing_times=processing_times,
                   weights=weights, families=families,
                   setup_time=setup_time, n_families=n_families)


@dataclass
class BatchSchedulingSolution:
    """Solution to a batch scheduling problem.

    Attributes:
        schedule: List of job indices in processing order.
        batches: List of lists, each inner list is a batch of job indices.
        completion_times: Completion time of each job.
        total_weighted_completion: Total weighted completion time.
    """
    schedule: list[int]
    batches: list[list[int]]
    completion_times: np.ndarray
    total_weighted_completion: float

    def __repr__(self) -> str:
        return (f"BatchSchedulingSolution(twc={self.total_weighted_completion:.2f}, "
                f"n_batches={len(self.batches)})")
