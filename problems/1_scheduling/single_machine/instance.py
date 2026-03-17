"""
Single Machine Scheduling — Problem Instance and Solution Definitions

Defines dataclasses for single machine scheduling problems with various
objectives. Supports processing times, weights, due dates, and release dates.

Notation: 1 | β | γ
    - 1: single machine environment
    - β: job constraints (due dates, weights, release dates, precedence)
    - γ: objective function (Cmax, ΣCj, ΣwjCj, Lmax, ΣTj, ΣwjTj, ΣUj)

Reference: Pinedo, M.L. (2016). "Scheduling: Theory, Algorithms, and Systems"
           5th Edition, Springer. Chapters 3-5.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class SingleMachineInstance:
    """
    Single machine scheduling problem instance.

    Attributes:
        n: Number of jobs.
        processing_times: Processing time of each job, shape (n,).
        weights: Weight of each job, shape (n,). Default: all ones.
        due_dates: Due date of each job, shape (n,). Default: None (no due dates).
        release_dates: Release date of each job, shape (n,). Default: None (all zero).
    """

    n: int
    processing_times: np.ndarray
    weights: np.ndarray | None = None
    due_dates: np.ndarray | None = None
    release_dates: np.ndarray | None = None

    def __post_init__(self):
        assert self.processing_times.shape == (self.n,), (
            f"processing_times shape {self.processing_times.shape} != ({self.n},)"
        )
        if self.weights is not None:
            assert self.weights.shape == (self.n,)
        if self.due_dates is not None:
            assert self.due_dates.shape == (self.n,)
        if self.release_dates is not None:
            assert self.release_dates.shape == (self.n,)

    @classmethod
    def random(
        cls,
        n: int,
        seed: int | None = None,
        with_weights: bool = True,
        with_due_dates: bool = True,
        with_release_dates: bool = False,
        tardiness_factor: float = 0.4,
        due_date_range: float = 0.6,
        p_low: int = 1,
        p_high: int = 100,
    ) -> SingleMachineInstance:
        """
        Generate a random single machine instance.

        Due dates follow the Pinedo convention:
            d_j ~ U[P(1 - TF - RDD/2), P(1 - TF + RDD/2)]
        where P = sum of processing times, TF = tardiness_factor,
        RDD = due_date_range.

        Args:
            n: Number of jobs.
            seed: Random seed.
            with_weights: Whether to generate weights.
            with_due_dates: Whether to generate due dates.
            with_release_dates: Whether to generate release dates.
            tardiness_factor: TF parameter for due date generation.
            due_date_range: RDD parameter for due date generation.
            p_low: Minimum processing time.
            p_high: Maximum processing time.

        Returns:
            A random SingleMachineInstance.
        """
        rng = np.random.default_rng(seed)

        processing_times = rng.integers(p_low, p_high + 1, size=n)

        weights = None
        if with_weights:
            weights = rng.integers(1, 11, size=n)

        due_dates = None
        if with_due_dates:
            total_p = processing_times.sum()
            d_low = max(0, total_p * (1 - tardiness_factor - due_date_range / 2))
            d_high = max(1, total_p * (1 - tardiness_factor + due_date_range / 2))
            due_dates = rng.integers(int(d_low), int(d_high) + 1, size=n)

        release_dates = None
        if with_release_dates:
            total_p = processing_times.sum()
            release_dates = rng.integers(0, int(total_p * 0.5) + 1, size=n)

        return cls(
            n=n,
            processing_times=processing_times,
            weights=weights,
            due_dates=due_dates,
            release_dates=release_dates,
        )

    @classmethod
    def from_arrays(
        cls,
        processing_times: list[int] | np.ndarray,
        weights: list[int] | np.ndarray | None = None,
        due_dates: list[int] | np.ndarray | None = None,
        release_dates: list[int] | np.ndarray | None = None,
    ) -> SingleMachineInstance:
        """Create instance from arrays (convenience constructor)."""
        p = np.asarray(processing_times)
        n = len(p)
        w = np.asarray(weights) if weights is not None else None
        d = np.asarray(due_dates) if due_dates is not None else None
        r = np.asarray(release_dates) if release_dates is not None else None
        return cls(n=n, processing_times=p, weights=w, due_dates=d, release_dates=r)


@dataclass
class SingleMachineSolution:
    """
    Solution to a single machine scheduling problem.

    Attributes:
        sequence: Job processing order (list of job indices).
        objective_value: Value of the objective function.
        objective_name: Name of the objective (e.g., "ΣwjCj", "Lmax").
    """

    sequence: list[int]
    objective_value: int | float
    objective_name: str = ""

    def __repr__(self) -> str:
        name = f" ({self.objective_name})" if self.objective_name else ""
        return (
            f"SingleMachineSolution(sequence={self.sequence}, "
            f"objective={self.objective_value}{name})"
        )


# ---------------------------------------------------------------------------
# Objective evaluation functions
# ---------------------------------------------------------------------------

def compute_completion_times(
    instance: SingleMachineInstance,
    sequence: list[int],
) -> np.ndarray:
    """
    Compute completion times for a given sequence.

    Args:
        instance: Problem instance.
        sequence: Job processing order.

    Returns:
        Array of completion times, shape (n,), indexed by job.
    """
    p = instance.processing_times
    C = np.zeros(instance.n)
    current_time = 0

    for job in sequence:
        if instance.release_dates is not None:
            current_time = max(current_time, instance.release_dates[job])
        current_time += p[job]
        C[job] = current_time

    return C


def compute_total_completion_time(
    instance: SingleMachineInstance,
    sequence: list[int],
) -> int:
    """Compute ΣCj for a given sequence."""
    C = compute_completion_times(instance, sequence)
    return int(C.sum())


def compute_weighted_completion_time(
    instance: SingleMachineInstance,
    sequence: list[int],
) -> int:
    """Compute ΣwjCj for a given sequence."""
    C = compute_completion_times(instance, sequence)
    w = instance.weights if instance.weights is not None else np.ones(instance.n)
    return int((w * C).sum())


def compute_makespan(
    instance: SingleMachineInstance,
    sequence: list[int],
) -> int:
    """Compute Cmax for a given sequence."""
    C = compute_completion_times(instance, sequence)
    return int(C.max())


def compute_lateness(
    instance: SingleMachineInstance,
    sequence: list[int],
) -> np.ndarray:
    """Compute lateness Lj = Cj - dj for each job."""
    assert instance.due_dates is not None, "Due dates required for lateness"
    C = compute_completion_times(instance, sequence)
    return C - instance.due_dates


def compute_maximum_lateness(
    instance: SingleMachineInstance,
    sequence: list[int],
) -> int:
    """Compute Lmax = max(Lj) for a given sequence."""
    L = compute_lateness(instance, sequence)
    return int(L.max())


def compute_tardiness(
    instance: SingleMachineInstance,
    sequence: list[int],
) -> np.ndarray:
    """Compute tardiness Tj = max(0, Cj - dj) for each job."""
    L = compute_lateness(instance, sequence)
    return np.maximum(L, 0)


def compute_total_tardiness(
    instance: SingleMachineInstance,
    sequence: list[int],
) -> int:
    """Compute ΣTj for a given sequence."""
    return int(compute_tardiness(instance, sequence).sum())


def compute_weighted_tardiness(
    instance: SingleMachineInstance,
    sequence: list[int],
) -> int:
    """Compute ΣwjTj for a given sequence."""
    T = compute_tardiness(instance, sequence)
    w = instance.weights if instance.weights is not None else np.ones(instance.n)
    return int((w * T).sum())


def compute_number_tardy(
    instance: SingleMachineInstance,
    sequence: list[int],
) -> int:
    """Compute ΣUj (number of tardy jobs) for a given sequence."""
    T = compute_tardiness(instance, sequence)
    return int((T > 0).sum())


if __name__ == "__main__":
    print("=== Single Machine Instance Demo ===\n")

    inst = SingleMachineInstance.from_arrays(
        processing_times=[3, 5, 2, 7, 4],
        weights=[2, 1, 3, 1, 2],
        due_dates=[10, 12, 8, 20, 15],
    )
    print(f"Jobs: {inst.n}")
    print(f"Processing times: {inst.processing_times}")
    print(f"Weights:          {inst.weights}")
    print(f"Due dates:        {inst.due_dates}")

    seq = list(range(inst.n))
    print(f"\nSequence: {seq}")
    print(f"  Completion times: {compute_completion_times(inst, seq)}")
    print(f"  ΣCj   = {compute_total_completion_time(inst, seq)}")
    print(f"  ΣwjCj = {compute_weighted_completion_time(inst, seq)}")
    print(f"  Lmax  = {compute_maximum_lateness(inst, seq)}")
    print(f"  ΣTj   = {compute_total_tardiness(inst, seq)}")
    print(f"  ΣwjTj = {compute_weighted_tardiness(inst, seq)}")
    print(f"  ΣUj   = {compute_number_tardy(inst, seq)}")
