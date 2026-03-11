"""
Lot Streaming Flow Shop — Instance and Solution.

Problem notation: Fm | prmu, lot-streaming | Cmax

Each job can be split into sublots that are transferred between machines
independently. Allows overlapping of consecutive operations of the same
job on different machines, reducing makespan.

Complexity: NP-hard for m >= 3 with discrete sublots.

References:
    Trietsch, D. & Baker, K.R. (1993). Basic techniques for lot streaming.
    Operations Research, 41(6), 1065-1076.
    https://doi.org/10.1287/opre.41.6.1065

    Potts, C.N. & Baker, K.R. (2000). Flow shop scheduling with lot
    streaming. Operations Research Letters, 26(6), 297-303.
    https://doi.org/10.1016/S0167-6377(00)00011-0
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class LotStreamInstance:
    """Lot Streaming Flow Shop instance.

    Attributes:
        n: Number of jobs.
        m: Number of machines.
        processing_times: Processing times per job per machine, shape (n, m).
        num_sublots: Number of equal sublots per job.
        name: Optional instance name.
    """

    n: int
    m: int
    processing_times: np.ndarray
    num_sublots: int = 3
    name: str = ""

    def __post_init__(self):
        self.processing_times = np.asarray(self.processing_times, dtype=float)

    @classmethod
    def random(
        cls,
        n: int = 6,
        m: int = 3,
        num_sublots: int = 3,
        proc_range: tuple[int, int] = (5, 30),
        seed: int | None = None,
    ) -> LotStreamInstance:
        rng = np.random.default_rng(seed)
        proc = rng.integers(proc_range[0], proc_range[1] + 1, size=(n, m)).astype(float)
        return cls(n=n, m=m, processing_times=proc, num_sublots=num_sublots,
                   name=f"random_{n}x{m}")

    def makespan_no_streaming(self, perm: list[int]) -> float:
        """Standard flow shop makespan without lot streaming."""
        nj = len(perm)
        C = np.zeros((nj, self.m))
        for i, job in enumerate(perm):
            for k in range(self.m):
                prev_job = C[i - 1][k] if i > 0 else 0.0
                prev_machine = C[i][k - 1] if k > 0 else 0.0
                C[i][k] = max(prev_job, prev_machine) + self.processing_times[job][k]
        return float(C[-1][-1])

    def makespan_streaming(self, perm: list[int]) -> float:
        """Makespan with equal-sublot lot streaming.

        Each job is split into num_sublots equal sublots. Sublots are
        transferred between machines as soon as they complete.

        Args:
            perm: Job permutation (0-indexed).

        Returns:
            Makespan with lot streaming.
        """
        s = self.num_sublots
        nj = len(perm)
        # Sublot processing time for job j on machine k = p[j][k] / s
        # Track completion of each sublot
        # C[i][k][l] = completion of sublot l of job perm[i] on machine k
        C = np.zeros((nj, self.m, s))

        for i, job in enumerate(perm):
            for k in range(self.m):
                sublot_time = self.processing_times[job][k] / s
                for l in range(s):
                    earliest = 0.0
                    # After previous sublot of same job on same machine
                    if l > 0:
                        earliest = max(earliest, C[i][k][l - 1])
                    # After same sublot on previous machine
                    if k > 0:
                        earliest = max(earliest, C[i][k - 1][l])
                    # After previous job's sublot on same machine
                    if i > 0:
                        earliest = max(earliest, C[i - 1][k][l])
                    C[i][k][l] = earliest + sublot_time

        return float(C[-1][-1][-1])


@dataclass
class LotStreamSolution:
    """Lot Streaming Flow Shop solution.

    Attributes:
        permutation: Job processing order.
        makespan: Makespan with lot streaming.
        makespan_no_stream: Makespan without streaming (for comparison).
    """

    permutation: list[int]
    makespan: float
    makespan_no_stream: float

    def __repr__(self) -> str:
        return (f"LotStreamSolution(makespan={self.makespan:.1f}, "
                f"no_stream={self.makespan_no_stream:.1f})")


def validate_solution(
    instance: LotStreamInstance, solution: LotStreamSolution
) -> tuple[bool, list[str]]:
    errors = []
    if sorted(solution.permutation) != list(range(instance.n)):
        errors.append("Invalid permutation")
    actual = instance.makespan_streaming(solution.permutation)
    if abs(actual - solution.makespan) > 1e-4:
        errors.append(f"Makespan: {solution.makespan:.2f} != {actual:.2f}")
    return len(errors) == 0, errors


def small_ls_4x3() -> LotStreamInstance:
    return LotStreamInstance(
        n=4, m=3, num_sublots=3,
        processing_times=np.array([
            [12, 18, 9],
            [9, 15, 21],
            [24, 12, 6],
            [15, 21, 12],
        ], dtype=float),
        name="small_4x3",
    )


if __name__ == "__main__":
    inst = small_ls_4x3()
    perm = list(range(inst.n))
    print(f"No streaming: {inst.makespan_no_streaming(perm):.1f}")
    print(f"With streaming ({inst.num_sublots} sublots): {inst.makespan_streaming(perm):.1f}")
