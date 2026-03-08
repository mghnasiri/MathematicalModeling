"""
Parallel Machine Instance & Solution Data Structures

Supports three machine environments:
- Identical (Pm): all machines have equal speed
- Uniform (Qm): machine i has speed s_i, processing time = p_j / s_i
- Unrelated (Rm): processing time p_ij depends on both job and machine

Primary objectives:
- Cmax (makespan): completion time of the last job
- Sum Cj (total completion time)
- Sum wjCj (total weighted completion time)

Reference: Pinedo, M.L. (2016). "Scheduling: Theory, Algorithms, and Systems"
           5th Edition, Springer.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class ParallelMachineInstance:
    """
    A parallel machine scheduling instance.

    Attributes:
        n: Number of jobs.
        m: Number of machines.
        processing_times: For identical/uniform machines, 1D array of shape (n,).
                         For unrelated machines, 2D array of shape (m, n).
        weights: Job weights, shape (n,). Default: all ones.
        speeds: Machine speeds for uniform machines, shape (m,).
                Default: all ones (identical machines).
        machine_type: One of "identical", "uniform", "unrelated".
    """
    n: int
    m: int
    processing_times: np.ndarray
    weights: np.ndarray | None = None
    speeds: np.ndarray | None = None
    machine_type: str = "identical"

    def __post_init__(self):
        if self.machine_type == "unrelated":
            assert self.processing_times.shape == (self.m, self.n), (
                f"Unrelated: expected shape ({self.m}, {self.n}), "
                f"got {self.processing_times.shape}"
            )
        else:
            assert self.processing_times.shape == (self.n,), (
                f"Expected shape ({self.n},), got {self.processing_times.shape}"
            )

        if self.weights is None:
            self.weights = np.ones(self.n, dtype=float)
        else:
            assert self.weights.shape == (self.n,)

        if self.speeds is None:
            self.speeds = np.ones(self.m, dtype=float)
        else:
            assert self.speeds.shape == (self.m,)

    @classmethod
    def random_identical(
        cls, n: int, m: int, low: int = 1, high: int = 99,
        seed: int | None = None,
    ) -> ParallelMachineInstance:
        """Generate a random identical parallel machine instance."""
        rng = np.random.default_rng(seed)
        processing_times = rng.integers(low, high + 1, size=n).astype(float)
        weights = rng.integers(1, 10, size=n).astype(float)
        return cls(n=n, m=m, processing_times=processing_times,
                   weights=weights, machine_type="identical")

    @classmethod
    def random_uniform(
        cls, n: int, m: int, low: int = 1, high: int = 99,
        seed: int | None = None,
    ) -> ParallelMachineInstance:
        """Generate a random uniform parallel machine instance."""
        rng = np.random.default_rng(seed)
        processing_times = rng.integers(low, high + 1, size=n).astype(float)
        speeds = rng.uniform(0.5, 2.0, size=m)
        return cls(n=n, m=m, processing_times=processing_times,
                   speeds=speeds, machine_type="uniform")

    @classmethod
    def random_unrelated(
        cls, n: int, m: int, low: int = 1, high: int = 99,
        seed: int | None = None,
    ) -> ParallelMachineInstance:
        """Generate a random unrelated parallel machine instance."""
        rng = np.random.default_rng(seed)
        processing_times = rng.integers(low, high + 1, size=(m, n)).astype(float)
        weights = rng.integers(1, 10, size=n).astype(float)
        return cls(n=n, m=m, processing_times=processing_times,
                   weights=weights, machine_type="unrelated")

    def get_processing_time(self, job: int, machine: int) -> float:
        """
        Get the processing time of a job on a machine.

        Args:
            job: Job index.
            machine: Machine index.

        Returns:
            Processing time (float).
        """
        if self.machine_type == "unrelated":
            return float(self.processing_times[machine, job])
        elif self.machine_type == "uniform":
            return float(self.processing_times[job]) / float(self.speeds[machine])
        else:  # identical
            return float(self.processing_times[job])


@dataclass
class ParallelMachineSolution:
    """
    A parallel machine scheduling solution.

    Attributes:
        assignment: List of m lists, where assignment[i] is the ordered list
                   of job indices assigned to machine i.
        makespan: The Cmax value of this solution.
        machine_loads: Total processing time on each machine.
    """
    assignment: list[list[int]]
    makespan: float
    machine_loads: list[float] | None = None

    def __repr__(self) -> str:
        return (f"ParallelMachineSolution(makespan={self.makespan:.1f}, "
                f"assignment={self.assignment})")


def compute_makespan(
    instance: ParallelMachineInstance,
    assignment: list[list[int]],
) -> float:
    """
    Compute the makespan of an assignment.

    Args:
        instance: A ParallelMachineInstance.
        assignment: List of m lists of job indices per machine.

    Returns:
        The makespan (maximum machine load).
    """
    loads = compute_machine_loads(instance, assignment)
    return max(loads) if loads else 0.0


def compute_machine_loads(
    instance: ParallelMachineInstance,
    assignment: list[list[int]],
) -> list[float]:
    """
    Compute the total processing time (load) on each machine.

    Args:
        instance: A ParallelMachineInstance.
        assignment: List of m lists of job indices per machine.

    Returns:
        List of loads, one per machine.
    """
    loads = []
    for i, jobs in enumerate(assignment):
        load = sum(instance.get_processing_time(j, i) for j in jobs)
        loads.append(load)
    return loads


def compute_total_completion_time(
    instance: ParallelMachineInstance,
    assignment: list[list[int]],
) -> float:
    """
    Compute the total (weighted) completion time of an assignment.

    Jobs on each machine are processed in the order given. The completion
    time of a job depends on its position in the machine's sequence.

    Args:
        instance: A ParallelMachineInstance.
        assignment: List of m lists of job indices per machine.

    Returns:
        Total weighted completion time.
    """
    total = 0.0
    for i, jobs in enumerate(assignment):
        time_on_machine = 0.0
        for job in jobs:
            time_on_machine += instance.get_processing_time(job, i)
            total += instance.weights[job] * time_on_machine
    return total


if __name__ == "__main__":
    print("=== Parallel Machine Instance ===")

    # Identical machines
    inst = ParallelMachineInstance.random_identical(n=10, m=3, seed=42)
    print(f"Identical: {inst.n} jobs, {inst.m} machines")
    print(f"Processing times: {inst.processing_times}")

    # Simple assignment: round-robin
    assignment = [[] for _ in range(inst.m)]
    for j in range(inst.n):
        assignment[j % inst.m].append(j)

    ms = compute_makespan(inst, assignment)
    loads = compute_machine_loads(inst, assignment)
    print(f"Round-robin: makespan={ms:.0f}, loads={[f'{l:.0f}' for l in loads]}")
