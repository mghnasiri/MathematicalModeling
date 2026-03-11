"""
Distributed Permutation Flow Shop (DPFSP) — Instance and Solution.

Problem notation: DFm | prmu | Cmax

Multiple identical factories, each containing a flow shop with the same
number of machines. Jobs must be assigned to factories and sequenced
within each factory. Minimize overall makespan (max completion across
all factories).

Complexity: NP-hard (generalizes PFSP).

References:
    Naderi, B. & Ruiz, R. (2010). The distributed permutation flowshop
    scheduling problem. Computers & Operations Research, 37(4), 754-768.
    https://doi.org/10.1016/j.cor.2009.06.019
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class DPFSPInstance:
    """Distributed Permutation Flow Shop instance.

    Attributes:
        n: Number of jobs.
        m: Number of machines per factory.
        f: Number of factories.
        processing_times: Processing times, shape (n, m).
        name: Optional instance name.
    """

    n: int
    m: int
    f: int
    processing_times: np.ndarray
    name: str = ""

    def __post_init__(self):
        self.processing_times = np.asarray(self.processing_times, dtype=float)

    @classmethod
    def random(
        cls,
        n: int = 10,
        m: int = 3,
        f: int = 2,
        proc_range: tuple[int, int] = (1, 20),
        seed: int | None = None,
    ) -> DPFSPInstance:
        rng = np.random.default_rng(seed)
        proc = rng.integers(proc_range[0], proc_range[1] + 1, size=(n, m)).astype(float)
        return cls(n=n, m=m, f=f, processing_times=proc,
                   name=f"random_{n}x{m}x{f}")

    def factory_makespan(self, perm: list[int]) -> float:
        """Compute makespan for a single factory with given job permutation."""
        if not perm:
            return 0.0
        nj = len(perm)
        C = np.zeros((nj, self.m))
        for i, job in enumerate(perm):
            for k in range(self.m):
                prev_job = C[i - 1][k] if i > 0 else 0.0
                prev_machine = C[i][k - 1] if k > 0 else 0.0
                C[i][k] = max(prev_job, prev_machine) + self.processing_times[job][k]
        return float(C[-1][-1])

    def makespan(self, assignment: list[list[int]]) -> float:
        """Compute overall makespan across all factories.

        Args:
            assignment: List of f permutations, one per factory.

        Returns:
            Maximum factory makespan.
        """
        return max(
            (self.factory_makespan(perm) for perm in assignment),
            default=0.0
        )


@dataclass
class DPFSPSolution:
    """DPFSP solution.

    Attributes:
        assignment: Job permutations per factory.
        makespan: Overall makespan.
    """

    assignment: list[list[int]]
    makespan: float

    def __repr__(self) -> str:
        sizes = [len(a) for a in self.assignment]
        return f"DPFSPSolution(factories={sizes}, makespan={self.makespan:.1f})"


def validate_solution(
    instance: DPFSPInstance, solution: DPFSPSolution
) -> tuple[bool, list[str]]:
    errors = []
    if len(solution.assignment) != instance.f:
        errors.append(f"Expected {instance.f} factories, got {len(solution.assignment)}")
    all_jobs = []
    for perm in solution.assignment:
        all_jobs.extend(perm)
    if sorted(all_jobs) != list(range(instance.n)):
        errors.append("Assignment is not a valid partition of all jobs")
    actual = instance.makespan(solution.assignment)
    if abs(actual - solution.makespan) > 1e-4:
        errors.append(f"Reported makespan {solution.makespan:.2f} != actual {actual:.2f}")
    return len(errors) == 0, errors


def small_dpfsp_6x3x2() -> DPFSPInstance:
    return DPFSPInstance(
        n=6, m=3, f=2,
        processing_times=np.array([
            [5, 8, 3],
            [3, 6, 7],
            [8, 4, 5],
            [6, 7, 2],
            [4, 3, 9],
            [7, 5, 4],
        ], dtype=float),
        name="small_6x3x2",
    )


if __name__ == "__main__":
    inst = small_dpfsp_6x3x2()
    print(f"{inst.name}: n={inst.n}, m={inst.m}, f={inst.f}")
    assign = [list(range(3)), list(range(3, 6))]
    print(f"  Split evenly makespan: {inst.makespan(assign):.1f}")
