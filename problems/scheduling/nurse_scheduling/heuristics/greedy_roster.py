"""Greedy heuristic for Nurse Scheduling / Staff Rostering.

Processes each (day, shift) pair in order and assigns the most available
nurse (fewest shifts so far, respecting constraints).

Complexity: O(d * s * n) — iterating over days, shifts, and nurses.

References:
    Ernst, A. T., Jiang, H., Krishnamoorthy, M., & Sier, D. (2004).
    Staff scheduling and rostering: A review of applications, methods
    and models. European Journal of Operational Research, 153(1), 3-27.
"""
from __future__ import annotations

import sys
import os
import importlib.util
import numpy as np


def _load_parent(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_inst = _load_parent(
    "ns_instance",
    os.path.join(os.path.dirname(__file__), "..", "instance.py"),
)
NurseSchedulingInstance = _inst.NurseSchedulingInstance
NurseSchedulingSolution = _inst.NurseSchedulingSolution


def _is_feasible(schedule: np.ndarray, nurse: int, day: int,
                 instance: NurseSchedulingInstance) -> bool:
    """Check if assigning nurse to a shift on day is feasible.

    Args:
        schedule: Current schedule array.
        nurse: Nurse index.
        day: Day index.
        instance: Problem instance.

    Returns:
        True if assignment respects constraints.
    """
    # Already working a shift this day
    if schedule[nurse, day, :].sum() > 0:
        return False

    # Max total shifts
    if schedule[nurse].sum() >= instance.max_shifts:
        return False

    # Max consecutive days
    daily = schedule[nurse, :, :].sum(axis=1)
    streak = 0
    for dd in range(instance.n_days):
        working = daily[dd] > 0 if dd != day else True
        if working:
            streak += 1
            if streak > instance.max_consecutive:
                return False
        else:
            streak = 0

    return True


def greedy_roster(instance: NurseSchedulingInstance) -> NurseSchedulingSolution:
    """Greedy shift-by-shift nurse scheduling.

    For each (day, shift) pair, assigns the required number of nurses
    by selecting those with the fewest total shifts assigned so far,
    respecting feasibility constraints.

    Args:
        instance: A NurseSchedulingInstance.

    Returns:
        A NurseSchedulingSolution.
    """
    schedule = np.zeros((instance.n_nurses, instance.n_days, instance.n_shifts),
                        dtype=int)

    under_coverage = 0

    for d in range(instance.n_days):
        for s in range(instance.n_shifts):
            needed = int(instance.demand[d, s])
            # Get feasible nurses, sorted by total shifts (ascending)
            candidates = []
            for i in range(instance.n_nurses):
                if _is_feasible(schedule, i, d, instance):
                    total = int(schedule[i].sum())
                    candidates.append((total, i))
            candidates.sort()

            assigned = 0
            for _, nurse_idx in candidates:
                if assigned >= needed:
                    break
                # Re-check feasibility (nurse may have been assigned to
                # another shift on this day in a previous iteration)
                if schedule[nurse_idx, d, :].sum() == 0:
                    schedule[nurse_idx, d, s] = 1
                    assigned += 1

            under_coverage += max(0, needed - assigned)

    violations = instance.count_violations(schedule)
    total_viol = sum(violations.values())

    return NurseSchedulingSolution(
        schedule=schedule,
        under_coverage=under_coverage,
        total_violations=total_viol,
        objective=under_coverage,
    )


if __name__ == "__main__":
    inst = NurseSchedulingInstance.random(n_nurses=8, n_days=7, n_shifts=3,
                                          max_shifts=5, max_consecutive=5,
                                          seed=42)
    print(f"Instance: {inst.n_nurses} nurses, {inst.n_days} days, "
          f"{inst.n_shifts} shifts/day")
    print(f"Demand:\n{inst.demand}")

    sol = greedy_roster(inst)
    print(f"\nSolution: {sol}")
    print(f"Schedule shape: {sol.schedule.shape}")
    for i in range(inst.n_nurses):
        shifts = sol.schedule[i].sum()
        print(f"  Nurse {i}: {int(shifts)} shifts")
