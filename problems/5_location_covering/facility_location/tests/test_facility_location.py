"""
Test suite for Uncapacitated Facility Location Problem (UFLP).

Tests cover:
- Instance creation and validation
- Greedy add/drop heuristics
- Simulated annealing
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


_inst_mod = _load_module("fl_inst_test", os.path.join(_base_dir, "instance.py"))
_gr_mod = _load_module("fl_gr_test", os.path.join(_base_dir, "heuristics", "greedy_facility.py"))
_sa_mod = _load_module("fl_sa_test", os.path.join(_base_dir, "metaheuristics", "simulated_annealing.py"))

FacilityLocationInstance = _inst_mod.FacilityLocationInstance
FacilityLocationSolution = _inst_mod.FacilityLocationSolution
validate_solution = _inst_mod.validate_solution
small_uflp_3_5 = _inst_mod.small_uflp_3_5
medium_uflp_5_10 = _inst_mod.medium_uflp_5_10

greedy_add = _gr_mod.greedy_add
greedy_drop = _gr_mod.greedy_drop
simulated_annealing = _sa_mod.simulated_annealing


@pytest.fixture
def inst_small():
    return small_uflp_3_5()


@pytest.fixture
def inst_medium():
    return medium_uflp_5_10()


@pytest.fixture
def random_inst():
    return FacilityLocationInstance.random(8, 15, seed=42)


class TestFacilityLocationInstance:
    def test_create_basic(self, inst_small):
        assert inst_small.m == 3
        assert inst_small.n == 5

    def test_random_instance(self):
        inst = FacilityLocationInstance.random(5, 10, seed=123)
        assert inst.m == 5
        assert inst.n == 10

    def test_total_cost(self, inst_small):
        # Open all, assign to nearest
        cost = inst_small.total_cost([0, 1, 2], [0, 1, 1, 2, 2])
        assert cost > 0

    def test_invalid_shapes(self):
        with pytest.raises(ValueError):
            FacilityLocationInstance(
                m=2, n=3,
                fixed_costs=np.array([1.0]),
                assignment_costs=np.zeros((2, 3)))


class TestValidation:
    def test_valid_solution(self, inst_small):
        sol = greedy_add(inst_small)
        valid, errors = validate_solution(inst_small, sol)
        assert valid, errors

    def test_customer_to_closed_facility(self, inst_small):
        sol = FacilityLocationSolution(
            open_facilities=[0],
            assignments=[0, 1, 0, 0, 0],  # Customer 1 to closed fac 1
            cost=0.0)
        valid, errors = validate_solution(inst_small, sol)
        assert not valid


class TestGreedyAdd:
    def test_feasible(self, inst_small):
        sol = greedy_add(inst_small)
        valid, errors = validate_solution(inst_small, sol)
        assert valid, errors

    def test_at_least_one_open(self, inst_small):
        sol = greedy_add(inst_small)
        assert len(sol.open_facilities) >= 1

    def test_random_instance(self, random_inst):
        sol = greedy_add(random_inst)
        valid, errors = validate_solution(random_inst, sol)
        assert valid, errors


class TestGreedyDrop:
    def test_feasible(self, inst_small):
        sol = greedy_drop(inst_small)
        valid, errors = validate_solution(inst_small, sol)
        assert valid, errors

    def test_random_instance(self, random_inst):
        sol = greedy_drop(random_inst)
        valid, errors = validate_solution(random_inst, sol)
        assert valid, errors


class TestSimulatedAnnealing:
    def test_feasible(self, inst_small):
        sol = simulated_annealing(inst_small, max_iterations=5000, seed=42)
        valid, errors = validate_solution(inst_small, sol)
        assert valid, errors

    def test_deterministic(self, inst_small):
        s1 = simulated_annealing(inst_small, max_iterations=3000, seed=42)
        s2 = simulated_annealing(inst_small, max_iterations=3000, seed=42)
        assert abs(s1.cost - s2.cost) < 1e-10

    def test_competitive(self, inst_small):
        ga_sol = greedy_add(inst_small)
        sa_sol = simulated_annealing(inst_small, max_iterations=10000, seed=42)
        assert sa_sol.cost <= ga_sol.cost * 1.3

    def test_random_instance(self, random_inst):
        sol = simulated_annealing(random_inst, max_iterations=5000, seed=42)
        valid, errors = validate_solution(random_inst, sol)
        assert valid, errors


class TestCrossMethod:
    def test_all_valid(self, inst_medium):
        methods = [
            greedy_add,
            greedy_drop,
            lambda i: simulated_annealing(i, max_iterations=3000, seed=42),
        ]
        for method in methods:
            sol = method(inst_medium)
            valid, errors = validate_solution(inst_medium, sol)
            assert valid, errors
