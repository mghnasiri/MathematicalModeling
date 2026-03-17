"""Workforce Scheduling Problem — shift assignment with skills.

Problem: Assign employees to shifts such that each shift's skill demands
are met, respecting employee availability and skill sets. Each employee
can be assigned to at most one shift per period.

Notation: WS | skills, availability | min uncovered demand

Complexity: NP-hard (reduces from set cover).

References:
    Ernst, A. T., Jiang, H., Krishnamoorthy, M., & Sier, D. (2004).
    Staff scheduling and rostering: A review of applications, methods
    and models. European Journal of Operational Research, 153(1), 3-27.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


@dataclass
class WorkforceInstance:
    """Workforce scheduling problem instance.

    Args:
        n_employees: Number of employees.
        n_shifts: Number of shifts to fill.
        n_skills: Number of distinct skill types.
        employee_skills: Boolean matrix (n_employees x n_skills).
            employee_skills[i][k] is True if employee i has skill k.
        shift_requirements: Integer matrix (n_shifts x n_skills).
            shift_requirements[j][k] is number of employees with skill k
            needed for shift j.
        availability: Boolean matrix (n_employees x n_shifts).
            availability[i][j] is True if employee i can work shift j.
        cost: Cost matrix (n_employees x n_shifts).
            cost[i][j] is cost of assigning employee i to shift j.
    """
    n_employees: int
    n_shifts: int
    n_skills: int
    employee_skills: np.ndarray
    shift_requirements: np.ndarray
    availability: np.ndarray
    cost: np.ndarray

    @classmethod
    def random(cls, n_employees: int = 10, n_shifts: int = 5,
               n_skills: int = 3, seed: int = 42) -> WorkforceInstance:
        """Generate a random workforce scheduling instance.

        Args:
            n_employees: Number of employees.
            n_shifts: Number of shifts.
            n_skills: Number of skill types.
            seed: Random seed for reproducibility.

        Returns:
            A random WorkforceInstance.
        """
        rng = np.random.default_rng(seed)
        # Each employee has at least one skill
        employee_skills = rng.random((n_employees, n_skills)) > 0.5
        for i in range(n_employees):
            if not employee_skills[i].any():
                employee_skills[i, rng.integers(n_skills)] = True

        # Each shift requires 1-3 employees per skill (some skills may have 0)
        shift_requirements = np.zeros((n_shifts, n_skills), dtype=int)
        for j in range(n_shifts):
            n_required_skills = rng.integers(1, n_skills + 1)
            skills_needed = rng.choice(n_skills, size=n_required_skills, replace=False)
            for k in skills_needed:
                shift_requirements[j, k] = rng.integers(1, 3)

        # Availability: ~70% chance available
        availability = rng.random((n_employees, n_shifts)) > 0.3

        # Cost: uniform 1-10
        cost = rng.integers(1, 11, size=(n_employees, n_shifts)).astype(float)

        return cls(
            n_employees=n_employees,
            n_shifts=n_shifts,
            n_skills=n_skills,
            employee_skills=employee_skills,
            shift_requirements=shift_requirements,
            availability=availability,
            cost=cost,
        )


@dataclass
class WorkforceSolution:
    """Solution to a workforce scheduling problem.

    Args:
        assignments: Dict mapping shift index to list of assigned employee indices.
        total_cost: Total assignment cost.
        uncovered_demand: Total unfilled skill-shift demand slots.
    """
    assignments: dict[int, list[int]]
    total_cost: float
    uncovered_demand: int

    def __repr__(self) -> str:
        return (f"WorkforceSolution(cost={self.total_cost:.1f}, "
                f"uncovered={self.uncovered_demand}, "
                f"shifts_filled={len(self.assignments)})")
