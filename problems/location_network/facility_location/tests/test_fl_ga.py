"""Tests for Genetic Algorithm on Facility Location (UFLP)."""

from __future__ import annotations

import sys
import os
import importlib.util

import numpy as np
import pytest

_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_mod(name, filepath):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_inst = _load_mod("fl_instance_test_ga", os.path.join(_parent_dir, "instance.py"))
FacilityLocationInstance = _inst.FacilityLocationInstance
FacilityLocationSolution = _inst.FacilityLocationSolution
validate_solution = _inst.validate_solution
small_uflp_3_5 = _inst.small_uflp_3_5
medium_uflp_5_10 = _inst.medium_uflp_5_10

_ga = _load_mod(
    "fl_ga_test",
    os.path.join(_parent_dir, "metaheuristics", "genetic_algorithm.py"),
)
genetic_algorithm = _ga.genetic_algorithm

_greedy = _load_mod(
    "fl_greedy_test_ga",
    os.path.join(_parent_dir, "heuristics", "greedy_facility.py"),
)
greedy_add = _greedy.greedy_add


class TestFLGAValidity:
    """Test that GA produces valid solutions."""

    def test_small_valid(self):
        inst = small_uflp_3_5()
        sol = genetic_algorithm(inst, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Invalid solution: {errors}"

    def test_medium_valid(self):
        inst = medium_uflp_5_10()
        sol = genetic_algorithm(inst, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Invalid solution: {errors}"

    def test_random_instance_valid(self):
        inst = FacilityLocationInstance.random(m=8, n=15, seed=123)
        sol = genetic_algorithm(inst, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Invalid solution: {errors}"

    def test_at_least_one_open(self):
        inst = small_uflp_3_5()
        sol = genetic_algorithm(inst, seed=42)
        assert len(sol.open_facilities) >= 1


class TestFLGAQuality:
    """Test solution quality."""

    def test_ga_not_worse_than_greedy(self):
        inst = small_uflp_3_5()
        greedy_sol = greedy_add(inst)
        ga_sol = genetic_algorithm(inst, generations=300, seed=42)
        # GA should be competitive with greedy
        assert ga_sol.cost <= greedy_sol.cost * 1.2

    def test_cost_matches_computed(self):
        inst = small_uflp_3_5()
        sol = genetic_algorithm(inst, seed=42)
        actual_cost = inst.total_cost(sol.open_facilities, sol.assignments)
        assert abs(sol.cost - actual_cost) < 1e-4

    def test_all_customers_assigned(self):
        inst = medium_uflp_5_10()
        sol = genetic_algorithm(inst, seed=42)
        assert len(sol.assignments) == inst.n


class TestFLGADeterminism:
    """Test deterministic behavior with seed."""

    def test_same_seed_same_result(self):
        inst = small_uflp_3_5()
        sol1 = genetic_algorithm(inst, seed=42)
        sol2 = genetic_algorithm(inst, seed=42)
        assert abs(sol1.cost - sol2.cost) < 1e-6

    def test_different_seed_both_valid(self):
        inst = small_uflp_3_5()
        sol1 = genetic_algorithm(inst, seed=1)
        sol2 = genetic_algorithm(inst, seed=999)
        valid1, _ = validate_solution(inst, sol1)
        valid2, _ = validate_solution(inst, sol2)
        assert valid1 and valid2


class TestFLGAEdgeCases:
    """Test edge cases."""

    def test_single_facility(self):
        inst = FacilityLocationInstance(
            m=1, n=3,
            fixed_costs=np.array([50.0]),
            assignment_costs=np.array([[10.0, 20.0, 30.0]]),
            name="single_fac",
        )
        sol = genetic_algorithm(inst, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Invalid: {errors}"
        assert sol.open_facilities == [0]

    def test_time_limit(self):
        inst = FacilityLocationInstance.random(m=8, n=15, seed=42)
        sol = genetic_algorithm(
            inst, generations=1000000, time_limit=0.5, seed=42
        )
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Invalid: {errors}"
