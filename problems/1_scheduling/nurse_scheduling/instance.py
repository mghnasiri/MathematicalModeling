"""Nurse Scheduling Problem (NSP) / Staff Rostering.

Given n nurses and a planning horizon of d days with s shift types per day,
assign nurses to shifts to satisfy demand requirements while respecting
constraints (max shifts per nurse, no consecutive night shifts, etc.).

Objective: Minimize total under-coverage (unmet demand) or equivalently
maximize coverage.

Complexity: NP-hard in general (Ernst et al., 2004).

References:
    Ernst, A. T., Jiang, H., Krishnamoorthy, M., & Sier, D. (2004).
    Staff scheduling and rostering: A review of applications, methods
    and models. European Journal of Operational Research, 153(1), 3-27.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class NurseSchedulingInstance:
    """Nurse scheduling problem instance.

    Attributes:
        n_nurses: Number of nurses.
        n_days: Number of days in the planning horizon.
        n_shifts: Number of shift types per day (e.g., 3: morning/evening/night).
        demand: Required nurses per shift, shape (n_days, n_shifts).
        max_shifts: Maximum total shifts any nurse can work.
        max_consecutive: Maximum consecutive working days per nurse.
        shift_names: Names for each shift type.
    """

    n_nurses: int
    n_days: int
    n_shifts: int
    demand: np.ndarray
    max_shifts: int
    max_consecutive: int
    shift_names: list[str] = field(default_factory=lambda: ["Morning", "Evening", "Night"])

    @classmethod
    def random(cls, n_nurses: int = 10, n_days: int = 7, n_shifts: int = 3,
               max_shifts: int = 5, max_consecutive: int = 5,
               seed: int | None = None) -> NurseSchedulingInstance:
        """Generate a random nurse scheduling instance.

        Args:
            n_nurses: Number of nurses.
            n_days: Number of days.
            n_shifts: Number of shift types.
            max_shifts: Max shifts per nurse.
            max_consecutive: Max consecutive days.
            seed: Random seed.

        Returns:
            A random NurseSchedulingInstance.
        """
        rng = np.random.default_rng(seed)
        demand = rng.integers(1, max(2, n_nurses // 2), size=(n_days, n_shifts))
        names = ["Morning", "Evening", "Night"][:n_shifts]
        if n_shifts > 3:
            names = [f"Shift_{i}" for i in range(n_shifts)]
        return cls(n_nurses=n_nurses, n_days=n_days, n_shifts=n_shifts,
                   demand=demand, max_shifts=max_shifts,
                   max_consecutive=max_consecutive, shift_names=names)

    def count_violations(self, schedule: np.ndarray) -> dict[str, int]:
        """Count constraint violations in a schedule.

        Args:
            schedule: Binary array of shape (n_nurses, n_days, n_shifts).
                      schedule[i, d, s] = 1 if nurse i works shift s on day d.

        Returns:
            Dictionary with violation counts.
        """
        violations = {}

        # Max shifts per nurse
        total_shifts = schedule.sum(axis=(1, 2))  # per nurse
        over = int(np.sum(np.maximum(0, total_shifts - self.max_shifts)))
        violations["max_shifts"] = over

        # At most one shift per day per nurse
        daily = schedule.sum(axis=2)  # (n_nurses, n_days)
        multi = int(np.sum(np.maximum(0, daily - 1)))
        violations["multi_shift"] = multi

        # Max consecutive days
        consec_violations = 0
        for i in range(self.n_nurses):
            streak = 0
            for d in range(self.n_days):
                if daily[i, d] > 0:
                    streak += 1
                    if streak > self.max_consecutive:
                        consec_violations += 1
                else:
                    streak = 0
        violations["consecutive"] = consec_violations

        return violations


@dataclass
class NurseSchedulingSolution:
    """Solution to a nurse scheduling problem.

    Attributes:
        schedule: Binary array (n_nurses, n_days, n_shifts).
        under_coverage: Total unmet demand across all shifts.
        total_violations: Total constraint violations.
        objective: under_coverage (lower is better).
    """

    schedule: np.ndarray
    under_coverage: int
    total_violations: int
    objective: int

    def __repr__(self) -> str:
        return (f"NurseSchedulingSolution(under_coverage={self.under_coverage}, "
                f"violations={self.total_violations})")
