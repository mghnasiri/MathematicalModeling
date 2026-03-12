"""Tests for Workforce Scheduling problem.

Tests: demand coverage, skill matching, no double-assignment,
edge cases, cost computation.
"""
from __future__ import annotations

import sys
import os
import importlib.util
import numpy as np
import pytest

def _load_mod(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_base = os.path.join(os.path.dirname(__file__), "..")

_instance_mod = _load_mod("wf_instance", os.path.join(_base, "instance.py"))
_greedy_mod = _load_mod("wf_greedy", os.path.join(_base, "heuristics", "greedy_shift_fill.py"))

WorkforceInstance = _instance_mod.WorkforceInstance
WorkforceSolution = _instance_mod.WorkforceSolution
greedy_shift_fill = _greedy_mod.greedy_shift_fill


class TestWorkforceInstance:
    """Test instance creation and random generation."""

    def test_random_instance_shape(self):
        inst = WorkforceInstance.random(n_employees=8, n_shifts=4, n_skills=2)
        assert inst.n_employees == 8
        assert inst.n_shifts == 4
        assert inst.n_skills == 2
        assert inst.employee_skills.shape == (8, 2)
        assert inst.shift_requirements.shape == (4, 2)
        assert inst.availability.shape == (8, 4)
        assert inst.cost.shape == (8, 4)

    def test_every_employee_has_skill(self):
        inst = WorkforceInstance.random(n_employees=20, n_skills=5, seed=99)
        for i in range(inst.n_employees):
            assert inst.employee_skills[i].any(), f"Employee {i} has no skills"

    def test_deterministic_random(self):
        inst1 = WorkforceInstance.random(seed=123)
        inst2 = WorkforceInstance.random(seed=123)
        np.testing.assert_array_equal(inst1.employee_skills, inst2.employee_skills)
        np.testing.assert_array_equal(inst1.cost, inst2.cost)


class TestGreedyShiftFill:
    """Test greedy shift fill algorithm."""

    def test_no_double_assignment(self):
        """Each employee assigned to at most one shift."""
        inst = WorkforceInstance.random(n_employees=10, n_shifts=5, seed=42)
        sol = greedy_shift_fill(inst)
        all_assigned = []
        for emps in sol.assignments.values():
            all_assigned.extend(emps)
        assert len(all_assigned) == len(set(all_assigned)), \
            "An employee is assigned to multiple shifts"

    def test_skill_matching(self):
        """Assigned employees have required skills."""
        inst = WorkforceInstance.random(n_employees=10, n_shifts=3, n_skills=2, seed=42)
        sol = greedy_shift_fill(inst)
        for j, emps in sol.assignments.items():
            for emp in emps:
                # Employee must have at least one skill needed by the shift
                has_relevant_skill = False
                for k in range(inst.n_skills):
                    if inst.employee_skills[emp, k] and inst.shift_requirements[j, k] > 0:
                        has_relevant_skill = True
                        break
                assert has_relevant_skill, \
                    f"Employee {emp} assigned to shift {j} without relevant skill"

    def test_availability_respected(self):
        """Only available employees are assigned."""
        inst = WorkforceInstance.random(n_employees=10, n_shifts=5, seed=42)
        sol = greedy_shift_fill(inst)
        for j, emps in sol.assignments.items():
            for emp in emps:
                assert inst.availability[emp, j], \
                    f"Employee {emp} assigned to shift {j} but not available"

    def test_cost_nonnegative(self):
        inst = WorkforceInstance.random(seed=42)
        sol = greedy_shift_fill(inst)
        assert sol.total_cost >= 0

    def test_uncovered_demand_nonnegative(self):
        inst = WorkforceInstance.random(seed=42)
        sol = greedy_shift_fill(inst)
        assert sol.uncovered_demand >= 0

    def test_full_coverage_easy_instance(self):
        """With many employees and few shifts, demand should be fully covered."""
        # 20 employees, 2 shifts, 1 skill, all available and qualified
        n_emp, n_shift, n_skill = 20, 2, 1
        inst = WorkforceInstance(
            n_employees=n_emp, n_shifts=n_shift, n_skills=n_skill,
            employee_skills=np.ones((n_emp, n_skill), dtype=bool),
            shift_requirements=np.array([[2], [2]], dtype=int),
            availability=np.ones((n_emp, n_shift), dtype=bool),
            cost=np.ones((n_emp, n_shift)),
        )
        sol = greedy_shift_fill(inst)
        assert sol.uncovered_demand == 0

    def test_single_employee_single_shift(self):
        inst = WorkforceInstance(
            n_employees=1, n_shifts=1, n_skills=1,
            employee_skills=np.array([[True]]),
            shift_requirements=np.array([[1]], dtype=int),
            availability=np.array([[True]]),
            cost=np.array([[5.0]]),
        )
        sol = greedy_shift_fill(inst)
        assert sol.uncovered_demand == 0
        assert sol.total_cost == 5.0
        assert sol.assignments[0] == [0]

    def test_solution_repr(self):
        inst = WorkforceInstance.random(seed=42)
        sol = greedy_shift_fill(inst)
        r = repr(sol)
        assert "WorkforceSolution" in r
        assert "cost=" in r

    def test_all_shifts_have_entry(self):
        """Every shift should have an entry in assignments."""
        inst = WorkforceInstance.random(n_shifts=4, seed=42)
        sol = greedy_shift_fill(inst)
        for j in range(inst.n_shifts):
            assert j in sol.assignments
