"""Greedy Shift Fill heuristic for Workforce Scheduling.

Algorithm: For each shift, greedily assign available qualified employees
to fill skill demands, prioritizing lowest-cost assignments.

Complexity: O(n_shifts * n_employees * n_skills)

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
    "workforce_instance",
    os.path.join(os.path.dirname(__file__), "..", "instance.py"),
)
WorkforceInstance = _inst.WorkforceInstance
WorkforceSolution = _inst.WorkforceSolution


def greedy_shift_fill(instance: WorkforceInstance) -> WorkforceSolution:
    """Assign employees to shifts using a greedy cost-based approach.

    For each shift, identify unfilled skill demands and assign the
    cheapest available qualified employee until demands are met or
    no more employees are available.

    Args:
        instance: A WorkforceInstance.

    Returns:
        A WorkforceSolution with assignments, total cost, and uncovered demand.
    """
    assigned_employees: set[int] = set()
    assignments: dict[int, list[int]] = {}
    total_cost = 0.0
    total_uncovered = 0

    for j in range(instance.n_shifts):
        remaining_demand = instance.shift_requirements[j].copy()
        shift_employees: list[int] = []

        # Build candidate list: (cost, employee, skill) sorted by cost
        candidates = []
        for i in range(instance.n_employees):
            if i in assigned_employees:
                continue
            if not instance.availability[i, j]:
                continue
            # Find skills this employee can contribute to
            for k in range(instance.n_skills):
                if instance.employee_skills[i, k] and remaining_demand[k] > 0:
                    candidates.append((instance.cost[i, j], i, k))

        # Sort by cost (greedy cheapest first)
        candidates.sort(key=lambda x: x[0])

        # Assign greedily
        for cost, emp, skill in candidates:
            if emp in assigned_employees:
                continue
            # Check if this employee can still contribute to any unfilled demand
            contributes = False
            for k in range(instance.n_skills):
                if instance.employee_skills[emp, k] and remaining_demand[k] > 0:
                    contributes = True
                    break
            if not contributes:
                continue

            # Assign employee to shift
            shift_employees.append(emp)
            assigned_employees.add(emp)
            total_cost += cost

            # Reduce demand for all skills this employee has
            for k in range(instance.n_skills):
                if instance.employee_skills[emp, k] and remaining_demand[k] > 0:
                    remaining_demand[k] -= 1

            # Check if all demands met
            if remaining_demand.sum() == 0:
                break

        assignments[j] = shift_employees
        total_uncovered += int(remaining_demand.sum())

    return WorkforceSolution(
        assignments=assignments,
        total_cost=total_cost,
        uncovered_demand=total_uncovered,
    )


if __name__ == "__main__":
    inst = WorkforceInstance.random(n_employees=15, n_shifts=5, n_skills=3)
    sol = greedy_shift_fill(inst)
    print(f"Instance: {inst.n_employees} employees, {inst.n_shifts} shifts, "
          f"{inst.n_skills} skills")
    print(f"Solution: {sol}")
    for j, emps in sol.assignments.items():
        print(f"  Shift {j}: employees {emps}")
