"""
Test suite for Linear Assignment Problem.

Tests cover:
- Instance creation and validation
- Hungarian (Kuhn-Munkres) algorithm
- Greedy heuristic
"""

from __future__ import annotations

import os
import sys
import pytest
import numpy as np
import importlib.util

_base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_module(name: str, filepath: str):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_inst_mod = _load_module("ap_inst_test", os.path.join(_base_dir, "instance.py"))
_hu_mod = _load_module("ap_hu_test", os.path.join(_base_dir, "exact", "hungarian.py"))
_gr_mod = _load_module("ap_gr_test", os.path.join(_base_dir, "heuristics", "greedy_assignment.py"))

AssignmentInstance = _inst_mod.AssignmentInstance
AssignmentSolution = _inst_mod.AssignmentSolution
validate_solution = _inst_mod.validate_solution
small_assignment_3 = _inst_mod.small_assignment_3
medium_assignment_5 = _inst_mod.medium_assignment_5

hungarian = _hu_mod.hungarian
greedy_assignment = _gr_mod.greedy_assignment


@pytest.fixture
def inst3():
    return small_assignment_3()


@pytest.fixture
def inst5():
    return medium_assignment_5()


@pytest.fixture
def random_inst():
    return AssignmentInstance.random(10, seed=42)


class TestAssignmentInstance:
    def test_create_basic(self, inst3):
        assert inst3.n == 3

    def test_random_instance(self):
        inst = AssignmentInstance.random(8, seed=123)
        assert inst.n == 8
        assert inst.cost_matrix.shape == (8, 8)

    def test_total_cost(self, inst3):
        cost = inst3.total_cost([1, 0, 2])
        assert abs(cost - (2 + 6 + 1)) < 1e-6

    def test_invalid_shape(self):
        with pytest.raises(ValueError):
            AssignmentInstance(n=3, cost_matrix=np.zeros((2, 3)))


class TestValidation:
    def test_valid_solution(self, inst3):
        sol = hungarian(inst3)
        valid, errors = validate_solution(inst3, sol)
        assert valid, errors

    def test_not_permutation(self, inst3):
        sol = AssignmentSolution(assignment=[0, 0, 1], cost=0.0)
        valid, errors = validate_solution(inst3, sol)
        assert not valid


class TestHungarian:
    def test_small_3(self, inst3):
        sol = hungarian(inst3)
        valid, errors = validate_solution(inst3, sol)
        assert valid, errors
        # Optimal: 0->1(2), 1->2(3), 2->0(5) = 10
        # OR: 0->2(7), 1->0(6), 2->2(1) = 14
        # OR: 0->1(2), 1->0(6), 2->2(1) = 9
        assert sol.cost <= 10.0

    def test_medium_5(self, inst5):
        sol = hungarian(inst5)
        valid, errors = validate_solution(inst5, sol)
        assert valid, errors
        assert sol.cost <= 15.0

    def test_identity_assignment(self):
        costs = np.eye(3) * 100 + (1 - np.eye(3)) * 1
        inst = AssignmentInstance(n=3, cost_matrix=costs)
        sol = hungarian(inst)
        # Should assign to off-diagonal: cost = 3
        assert abs(sol.cost - 3.0) < 1e-6

    def test_optimal_permutation(self, inst3):
        sol = hungarian(inst3)
        assert sorted(sol.assignment) == [0, 1, 2]

    def test_random_instance(self, random_inst):
        sol = hungarian(random_inst)
        valid, errors = validate_solution(random_inst, sol)
        assert valid, errors

    def test_beats_greedy(self, random_inst):
        h = hungarian(random_inst)
        g = greedy_assignment(random_inst)
        assert h.cost <= g.cost + 1e-6


class TestGreedyAssignment:
    def test_feasible(self, inst3):
        sol = greedy_assignment(inst3)
        valid, errors = validate_solution(inst3, sol)
        assert valid, errors

    def test_random_instance(self, random_inst):
        sol = greedy_assignment(random_inst)
        valid, errors = validate_solution(random_inst, sol)
        assert valid, errors


class TestCrossMethod:
    def test_hungarian_optimal(self, inst5):
        h = hungarian(inst5)
        g = greedy_assignment(inst5)
        assert h.cost <= g.cost + 1e-6

    def test_both_valid(self, random_inst):
        for method in [hungarian, greedy_assignment]:
            sol = method(random_inst)
            valid, errors = validate_solution(random_inst, sol)
            assert valid, errors
