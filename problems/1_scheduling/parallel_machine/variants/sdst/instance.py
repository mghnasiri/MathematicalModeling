"""
Parallel Machine with Sequence-Dependent Setup Times — Instance and Solution.

Problem notation: Rm | Ssd | Cmax

Extends unrelated parallel machines: setup time s_ij^k depends on the
preceding job i, the next job j, and the machine k. Objective: minimize
makespan.

Applications: semiconductor fabrication, printing (color changeovers),
injection molding, chemical processing with machine-specific changeovers.

Complexity: NP-hard (generalizes Rm||Cmax).

References:
    Allahverdi, A., Ng, C.T., Cheng, T.C.E. & Kovalyov, M.Y. (2008).
    A survey of scheduling problems with setup times or costs. European
    Journal of Operational Research, 187(3), 985-1032.
    https://doi.org/10.1016/j.ejor.2006.06.060

    Rabadi, G., Moraga, R.J. & Al-Salem, A. (2006). Heuristics for the
    unrelated parallel machine scheduling problem with setup times.
    Journal of Intelligent Manufacturing, 17(2), 199-207.
    https://doi.org/10.1007/s10845-005-6636-x
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class PMSDSTInstance:
    """Parallel Machine SDST instance.

    Attributes:
        n: Number of jobs.
        m: Number of machines.
        processing_times: Processing time p_jk, shape (n, m).
        setup_times: Setup time s_ijk, shape (n, n, m). s[i][j][k] = setup
            on machine k when job j follows job i. s[j][j][k] = initial
            setup for first job j on machine k.
        name: Optional instance name.
    """

    n: int
    m: int
    processing_times: np.ndarray
    setup_times: np.ndarray
    name: str = ""

    def __post_init__(self):
        self.processing_times = np.asarray(self.processing_times, dtype=float)
        self.setup_times = np.asarray(self.setup_times, dtype=float)

        if self.processing_times.shape != (self.n, self.m):
            raise ValueError(f"processing_times shape != ({self.n}, {self.m})")
        if self.setup_times.shape != (self.n, self.n, self.m):
            raise ValueError(f"setup_times shape != ({self.n}, {self.n}, {self.m})")

    @classmethod
    def random(
        cls,
        n: int = 10,
        m: int = 3,
        proc_range: tuple[int, int] = (10, 50),
        setup_range: tuple[int, int] = (1, 15),
        seed: int | None = None,
    ) -> PMSDSTInstance:
        """Generate a random instance.

        Args:
            n: Number of jobs.
            m: Number of machines.
            proc_range: Range for processing times.
            setup_range: Range for setup times.
            seed: Random seed.

        Returns:
            A random PMSDSTInstance.
        """
        rng = np.random.default_rng(seed)
        processing_times = rng.integers(
            proc_range[0], proc_range[1] + 1, size=(n, m)
        ).astype(float)
        setup_times = rng.integers(
            setup_range[0], setup_range[1] + 1, size=(n, n, m)
        ).astype(float)

        return cls(
            n=n, m=m,
            processing_times=processing_times,
            setup_times=setup_times,
            name=f"random_{n}_{m}",
        )

    def makespan(self, schedule: list[list[int]]) -> float:
        """Compute makespan for a schedule.

        Args:
            schedule: schedule[k] = ordered list of jobs on machine k.

        Returns:
            Makespan (Cmax).
        """
        cmax = 0.0
        for k in range(self.m):
            t = 0.0
            for idx, j in enumerate(schedule[k]):
                if idx == 0:
                    t += self.setup_times[j][j][k]
                else:
                    prev = schedule[k][idx - 1]
                    t += self.setup_times[prev][j][k]
                t += self.processing_times[j][k]
            cmax = max(cmax, t)
        return cmax


@dataclass
class PMSDSTSolution:
    """Solution to a PM-SDST instance.

    Attributes:
        schedule: schedule[k] = ordered list of jobs on machine k.
        makespan: Cmax value.
    """

    schedule: list[list[int]]
    makespan: float

    def __repr__(self) -> str:
        sizes = [len(s) for s in self.schedule]
        return f"PMSDSTSolution(makespan={self.makespan:.1f}, sizes={sizes})"


def validate_solution(
    instance: PMSDSTInstance, solution: PMSDSTSolution
) -> tuple[bool, list[str]]:
    """Validate a PM-SDST solution."""
    errors = []

    if len(solution.schedule) != instance.m:
        errors.append(f"Schedule has {len(solution.schedule)} machines, expected {instance.m}")
        return False, errors

    all_jobs = []
    for k, jobs in enumerate(solution.schedule):
        for j in jobs:
            if j < 0 or j >= instance.n:
                errors.append(f"Invalid job {j} on machine {k}")
            all_jobs.append(j)

    if sorted(all_jobs) != list(range(instance.n)):
        errors.append("Jobs are not a valid partition of {0, ..., n-1}")

    if errors:
        return False, errors

    actual = instance.makespan(solution.schedule)
    if abs(actual - solution.makespan) > 1e-4:
        errors.append(
            f"Reported makespan {solution.makespan:.2f} != actual {actual:.2f}"
        )

    return len(errors) == 0, errors


# ── Benchmark instances ──────────────────────────────────────────────────────


def small_pmsdst_4_2() -> PMSDSTInstance:
    """4 jobs, 2 machines with setup times."""
    return PMSDSTInstance(
        n=4, m=2,
        processing_times=np.array([
            [10, 15],
            [20, 12],
            [15, 18],
            [25, 10],
        ], dtype=float),
        setup_times=np.array([
            [[0, 0], [3, 5], [4, 2], [6, 3]],
            [[5, 3], [0, 0], [2, 4], [7, 2]],
            [[3, 4], [6, 2], [0, 0], [5, 6]],
            [[4, 5], [3, 3], [2, 4], [0, 0]],
        ], dtype=float),
        name="small_4_2",
    )


if __name__ == "__main__":
    inst = small_pmsdst_4_2()
    print(f"{inst.name}: n={inst.n}, m={inst.m}")
    schedule = [[0, 1], [2, 3]]
    print(f"  Schedule {schedule}: makespan={inst.makespan(schedule):.1f}")
