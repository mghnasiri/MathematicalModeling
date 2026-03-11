"""
Hybrid Flow Shop (HFS) — Instance and Solution.

Problem notation: HFm | prmu | Cmax

Extends the flow shop by having multiple identical parallel machines at each
stage. Each job must visit all stages in order, but at each stage can be
processed on any available machine.

Complexity: NP-hard even for 2 stages with 1 and 2 machines.

References:
    Linn, R. & Zhang, W. (1999). Hybrid flow shop scheduling: a survey.
    Computers & Industrial Engineering, 37(1-2), 57-61.
    https://doi.org/10.1016/S0360-8352(99)00023-6

    Ruiz, R. & Vázquez-Rodríguez, J.A. (2010). The hybrid flow shop
    scheduling problem. European Journal of Operational Research, 205(1),
    1-18. https://doi.org/10.1016/j.ejor.2009.09.024
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


@dataclass
class HFSInstance:
    """Hybrid Flow Shop instance.

    Attributes:
        n: Number of jobs.
        stages: Number of stages.
        machines_per_stage: Number of parallel machines at each stage, shape (stages,).
        processing_times: Processing times, shape (n, stages).
        name: Optional instance name.
    """

    n: int
    stages: int
    machines_per_stage: np.ndarray
    processing_times: np.ndarray
    name: str = ""

    def __post_init__(self):
        self.machines_per_stage = np.asarray(self.machines_per_stage, dtype=int)
        self.processing_times = np.asarray(self.processing_times, dtype=float)

    @classmethod
    def random(
        cls,
        n: int = 8,
        stages: int = 3,
        max_machines: int = 3,
        proc_range: tuple[int, int] = (1, 20),
        seed: int | None = None,
    ) -> HFSInstance:
        rng = np.random.default_rng(seed)
        machines = rng.integers(1, max_machines + 1, size=stages)
        proc = rng.integers(proc_range[0], proc_range[1] + 1, size=(n, stages)).astype(float)
        return cls(
            n=n, stages=stages, machines_per_stage=machines,
            processing_times=proc, name=f"random_{n}x{stages}"
        )

    def makespan(self, perm: list[int]) -> float:
        """Compute makespan for a job permutation using list scheduling at each stage.

        Jobs are assigned to stages in permutation order. At each stage,
        each job goes to the machine that becomes available earliest.

        Args:
            perm: Job permutation (0-indexed).

        Returns:
            Makespan value.
        """
        n = self.n
        s = self.stages

        # completion[j] = completion time of job j at the current stage
        completion = np.zeros(n)

        for stage in range(s):
            m = int(self.machines_per_stage[stage])
            machine_avail = np.zeros(m)

            for job in perm:
                # Earliest this job can start: after previous stage + earliest machine
                earliest_machine = int(np.argmin(machine_avail))
                start = max(completion[job], machine_avail[earliest_machine])
                end = start + self.processing_times[job][stage]
                completion[job] = end
                machine_avail[earliest_machine] = end

        return float(np.max(completion))


@dataclass
class HFSSolution:
    """HFS solution.

    Attributes:
        permutation: Job processing order.
        makespan: Makespan of the schedule.
    """

    permutation: list[int]
    makespan: float

    def __repr__(self) -> str:
        return f"HFSSolution(makespan={self.makespan:.1f})"


def validate_solution(
    instance: HFSInstance, solution: HFSSolution
) -> tuple[bool, list[str]]:
    errors = []
    if sorted(solution.permutation) != list(range(instance.n)):
        errors.append("Permutation is not valid")
    actual = instance.makespan(solution.permutation)
    if abs(actual - solution.makespan) > 1e-4:
        errors.append(f"Reported makespan {solution.makespan:.2f} != actual {actual:.2f}")
    return len(errors) == 0, errors


def small_hfs_4x3() -> HFSInstance:
    return HFSInstance(
        n=4,
        stages=3,
        machines_per_stage=np.array([2, 1, 2]),
        processing_times=np.array([
            [5, 8, 3],
            [3, 6, 7],
            [8, 4, 5],
            [6, 7, 2],
        ], dtype=float),
        name="small_4x3",
    )


if __name__ == "__main__":
    inst = small_hfs_4x3()
    print(f"{inst.name}: n={inst.n}, stages={inst.stages}")
    print(f"  Machines per stage: {inst.machines_per_stage}")
    print(f"  Identity perm makespan: {inst.makespan(list(range(inst.n))):.1f}")
