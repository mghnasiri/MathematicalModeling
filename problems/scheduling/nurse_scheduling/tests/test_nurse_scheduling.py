"""Tests for the Nurse Scheduling problem."""
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
_instance_mod = _load_mod("ns_instance", os.path.join(_base, "instance.py"))
_greedy_mod = _load_mod(
    "ns_greedy", os.path.join(_base, "heuristics", "greedy_roster.py")
)

NurseSchedulingInstance = _instance_mod.NurseSchedulingInstance
NurseSchedulingSolution = _instance_mod.NurseSchedulingSolution
greedy_roster = _greedy_mod.greedy_roster


class TestNurseSchedulingInstance:
    """Tests for instance construction."""

    def test_random_instance_shape(self):
        inst = NurseSchedulingInstance.random(n_nurses=6, n_days=5,
                                              n_shifts=3, seed=1)
        assert inst.n_nurses == 6
        assert inst.n_days == 5
        assert inst.n_shifts == 3
        assert inst.demand.shape == (5, 3)

    def test_demand_positive(self):
        inst = NurseSchedulingInstance.random(n_nurses=8, n_days=7, seed=2)
        assert np.all(inst.demand >= 1)

    def test_violation_counting(self):
        inst = NurseSchedulingInstance(
            n_nurses=2, n_days=3, n_shifts=2,
            demand=np.array([[1, 1], [1, 1], [1, 1]]),
            max_shifts=3, max_consecutive=2,
        )
        # No violations schedule
        sched = np.zeros((2, 3, 2), dtype=int)
        sched[0, 0, 0] = 1
        sched[1, 0, 1] = 1
        sched[0, 1, 0] = 1
        sched[1, 1, 1] = 1
        sched[0, 2, 0] = 1
        sched[1, 2, 1] = 1
        viol = inst.count_violations(sched)
        assert viol["multi_shift"] == 0


class TestGreedyRoster:
    """Tests for the greedy roster heuristic."""

    def test_schedule_shape(self):
        inst = NurseSchedulingInstance.random(n_nurses=6, n_days=5,
                                              n_shifts=3, seed=10)
        sol = greedy_roster(inst)
        assert sol.schedule.shape == (6, 5, 3)

    def test_binary_schedule(self):
        inst = NurseSchedulingInstance.random(n_nurses=6, n_days=5, seed=11)
        sol = greedy_roster(inst)
        assert np.all((sol.schedule == 0) | (sol.schedule == 1))

    def test_at_most_one_shift_per_day(self):
        inst = NurseSchedulingInstance.random(n_nurses=8, n_days=7, seed=12)
        sol = greedy_roster(inst)
        daily = sol.schedule.sum(axis=2)
        assert np.all(daily <= 1)

    def test_max_shifts_respected(self):
        inst = NurseSchedulingInstance.random(
            n_nurses=8, n_days=7, max_shifts=4, seed=13
        )
        sol = greedy_roster(inst)
        total_per_nurse = sol.schedule.sum(axis=(1, 2))
        assert np.all(total_per_nurse <= inst.max_shifts)

    def test_zero_undercoverage_with_surplus_nurses(self):
        """With many nurses and low demand, should have zero under-coverage."""
        inst = NurseSchedulingInstance(
            n_nurses=10, n_days=3, n_shifts=2,
            demand=np.array([[1, 1], [1, 1], [1, 1]]),
            max_shifts=3, max_consecutive=3,
        )
        sol = greedy_roster(inst)
        assert sol.under_coverage == 0

    def test_under_coverage_with_scarce_nurses(self):
        """With too few nurses, some demand must go unmet."""
        inst = NurseSchedulingInstance(
            n_nurses=1, n_days=3, n_shifts=2,
            demand=np.array([[2, 2], [2, 2], [2, 2]]),
            max_shifts=3, max_consecutive=3,
        )
        sol = greedy_roster(inst)
        assert sol.under_coverage > 0

    def test_objective_equals_undercoverage(self):
        inst = NurseSchedulingInstance.random(n_nurses=6, n_days=5, seed=14)
        sol = greedy_roster(inst)
        assert sol.objective == sol.under_coverage

    def test_no_violations(self):
        """Greedy should produce feasible schedules."""
        inst = NurseSchedulingInstance.random(
            n_nurses=10, n_days=5, max_shifts=5,
            max_consecutive=5, seed=15
        )
        sol = greedy_roster(inst)
        assert sol.total_violations == 0

    def test_single_day(self):
        inst = NurseSchedulingInstance(
            n_nurses=5, n_days=1, n_shifts=2,
            demand=np.array([[2, 2]]),
            max_shifts=1, max_consecutive=1,
        )
        sol = greedy_roster(inst)
        assert sol.under_coverage == 0
        assert sol.schedule.sum() == 4
