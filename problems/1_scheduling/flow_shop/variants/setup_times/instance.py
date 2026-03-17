"""
Sequence-Dependent Setup Times Flow Shop Instance & Solution Data Structures

In the SDST flow shop, a setup time s[i][j][k] is incurred on machine i when
switching from job j to job k. The setup time is sequence-dependent: it varies
based on both the preceding and succeeding jobs. Setup times are anticipatory
(setup on machine i can start as soon as the preceding job finishes on
machine i, even before the current job finishes on machine i-1).

The completion time recursion becomes:
    C[i][k] = max(C[i-1][pi(k)], C[i][pi(k-1)] + s[i][pi(k-1)][pi(k)])
              + p[i][pi(k)]

For the first job in the sequence, setup from a dummy initial state is used:
    C[i][pi(0)] = C[i-1][pi(0)] + s_initial[i][pi(0)] + p[i][pi(0)]

Notation: Fm | prmu, Ssd | Cmax
Reference: Ruiz, R., Maroto, C. & Alcaraz, J. (2005). "Solving the Flowshop
           Scheduling Problem with Sequence Dependent Setup Times Using
           Advanced Metaheuristics"
           European Journal of Operational Research, 165(1):34-54.
           DOI: 10.1016/j.ejor.2004.01.022

           Allahverdi, A., Ng, C.T., Cheng, T.C.E. & Kovalyov, M.Y. (2008).
           "A Survey of Scheduling Problems with Setup Times or Costs"
           European Journal of Operational Research, 187(3):985-1032.
           DOI: 10.1016/j.ejor.2006.06.060
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class SDSTFlowShopInstance:
    """
    A sequence-dependent setup times permutation flow shop instance.

    Attributes:
        n: Number of jobs.
        m: Number of machines.
        processing_times: Matrix of shape (m, n) where processing_times[i][j]
                         is the processing time of job j on machine i.
        setup_times: 3D array of shape (m, n+1, n) where setup_times[i][j][k]
                    is the setup time on machine i when transitioning from
                    job j to job k. Index j=n represents the initial setup
                    (no preceding job).
    """
    n: int
    m: int
    processing_times: np.ndarray  # shape (m, n)
    setup_times: np.ndarray  # shape (m, n+1, n) — last row of j-dim is initial

    def __post_init__(self):
        assert self.processing_times.shape == (self.m, self.n), (
            f"Processing times: expected shape ({self.m}, {self.n}), "
            f"got {self.processing_times.shape}"
        )
        assert self.setup_times.shape == (self.m, self.n + 1, self.n), (
            f"Setup times: expected shape ({self.m}, {self.n + 1}, {self.n}), "
            f"got {self.setup_times.shape}"
        )

    @classmethod
    def random(
        cls,
        n: int,
        m: int,
        p_low: int = 1,
        p_high: int = 99,
        s_low: int = 1,
        s_high: int = 49,
        seed: int | None = None,
    ) -> SDSTFlowShopInstance:
        """
        Generate a random SDST flow shop instance.

        Setup times are drawn from U[s_low, s_high], following the convention
        of Ruiz et al. (2005) where setup times are typically smaller than
        processing times.

        Args:
            n: Number of jobs.
            m: Number of machines.
            p_low: Lower bound for processing times.
            p_high: Upper bound for processing times.
            s_low: Lower bound for setup times.
            s_high: Upper bound for setup times.
            seed: Random seed for reproducibility.

        Returns:
            A random SDSTFlowShopInstance.
        """
        rng = np.random.default_rng(seed)
        processing_times = rng.integers(p_low, p_high + 1, size=(m, n))
        setup_times = rng.integers(s_low, s_high + 1, size=(m, n + 1, n))

        # Zero diagonal: no setup when a job follows itself (unused in PFSP
        # but consistent for general use)
        for i in range(m):
            for j in range(n):
                setup_times[i, j, j] = 0

        return cls(n=n, m=m, processing_times=processing_times,
                   setup_times=setup_times)


@dataclass
class SDSTFlowShopSolution:
    """
    A sequence-dependent setup times flow shop solution.

    Attributes:
        permutation: Job processing order (list of job indices).
        makespan: The Cmax value of this solution.
    """
    permutation: list[int]
    makespan: int

    def __repr__(self) -> str:
        return (f"SDSTFlowShopSolution(makespan={self.makespan}, "
                f"permutation={self.permutation})")


def compute_makespan_sdst(
    instance: SDSTFlowShopInstance,
    permutation: list[int],
) -> int:
    """
    Compute the makespan of a permutation with sequence-dependent setup times.

    The completion time recursion with anticipatory setups:
        C[i][k] = max(C[i-1][pi(k)],
                      C[i][pi(k-1)] + s[i][pi(k-1)][pi(k)]) + p[i][pi(k)]

    For the first job (k=0), the setup from the initial state is used:
        C[i][0] = max(C[i-1][pi(0)], s_initial) + s[i][n][pi(0)] + p[i][pi(0)]

    Note: Setups are anticipatory — the setup on machine i can begin as soon
    as the previous job completes on machine i, even if the current job hasn't
    yet finished on machine i-1.

    Args:
        instance: An SDSTFlowShopInstance.
        permutation: Job processing order.

    Returns:
        The makespan value.
    """
    if len(permutation) == 0:
        return 0

    n_jobs = len(permutation)
    m = instance.m
    p = instance.processing_times
    s = instance.setup_times

    # C[i] tracks completion time on machine i
    completion = np.zeros(m, dtype=int)

    for k in range(n_jobs):
        job = permutation[k]

        if k == 0:
            # First job: setup from initial state (index n)
            # Machine 0
            completion[0] = int(s[0, instance.n, job]) + int(p[0, job])
            # Machines 1..m-1
            for i in range(1, m):
                setup = int(s[i, instance.n, job])
                completion[i] = max(completion[i - 1], setup) + int(p[i, job])
        else:
            prev_job = permutation[k - 1]
            # Machine 0
            setup_0 = int(s[0, prev_job, job])
            completion[0] = completion[0] + setup_0 + int(p[0, job])
            # Machines 1..m-1
            for i in range(1, m):
                setup = int(s[i, prev_job, job])
                earliest = max(completion[i - 1], completion[i] + setup)
                completion[i] = earliest + int(p[i, job])

    return int(completion[-1])


if __name__ == "__main__":
    # Example: 4 jobs, 3 machines with SDST
    print("=== Sequence-Dependent Setup Times Flow Shop ===")

    instance = SDSTFlowShopInstance.random(n=4, m=3, seed=42)

    print(f"Instance: {instance.n} jobs, {instance.m} machines")
    print(f"\nProcessing times (m x n):")
    print(instance.processing_times)
    print(f"\nSetup times shape: {instance.setup_times.shape}")
    print(f"  (m={instance.m}, from={instance.n + 1} [incl. initial], to={instance.n})")

    # Evaluate a few permutations
    for perm in [[0, 1, 2, 3], [3, 0, 2, 1], [1, 0, 3, 2]]:
        ms = compute_makespan_sdst(instance, perm)
        print(f"\nPermutation {perm}: makespan = {ms}")
