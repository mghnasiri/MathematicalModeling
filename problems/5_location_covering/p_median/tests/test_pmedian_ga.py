"""Tests for Genetic Algorithm on p-Median Problem."""

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


_inst = _load_mod("pm_instance_test_ga", os.path.join(_parent_dir, "instance.py"))
PMedianInstance = _inst.PMedianInstance
PMedianSolution = _inst.PMedianSolution
validate_solution = _inst.validate_solution
small_pmedian_6_2 = _inst.small_pmedian_6_2

_ga = _load_mod(
    "pm_ga_test",
    os.path.join(_parent_dir, "metaheuristics", "genetic_algorithm.py"),
)
genetic_algorithm = _ga.genetic_algorithm

_greedy = _load_mod(
    "pm_greedy_test_ga",
    os.path.join(_parent_dir, "heuristics", "greedy_pmedian.py"),
)
greedy_pmedian = _greedy.greedy_pmedian


class TestPMedianGAValidity:
    """Test that GA produces valid solutions."""

    def test_small_valid(self):
        inst = small_pmedian_6_2()
        sol = genetic_algorithm(inst, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Invalid solution: {errors}"

    def test_random_instance_valid(self):
        inst = PMedianInstance.random(n=10, m=10, p=3, seed=42)
        sol = genetic_algorithm(inst, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Invalid solution: {errors}"

    def test_correct_number_open(self):
        inst = small_pmedian_6_2()
        sol = genetic_algorithm(inst, seed=42)
        assert len(sol.open_facilities) == inst.p


class TestPMedianGAQuality:
    """Test solution quality."""

    def test_ga_competitive_with_greedy(self):
        inst = small_pmedian_6_2()
        greedy_sol = greedy_pmedian(inst)
        ga_sol = genetic_algorithm(inst, generations=300, seed=42)
        assert ga_sol.cost <= greedy_sol.cost * 1.1

    def test_cost_matches_computed(self):
        inst = small_pmedian_6_2()
        sol = genetic_algorithm(inst, seed=42)
        actual_cost = inst.total_cost(sol.open_facilities, sol.assignments)
        assert abs(sol.cost - actual_cost) < 1e-4


class TestPMedianGADeterminism:
    """Test deterministic behavior with seed."""

    def test_same_seed_same_result(self):
        inst = small_pmedian_6_2()
        sol1 = genetic_algorithm(inst, seed=42)
        sol2 = genetic_algorithm(inst, seed=42)
        assert abs(sol1.cost - sol2.cost) < 1e-6

    def test_different_seed_both_valid(self):
        inst = small_pmedian_6_2()
        sol1 = genetic_algorithm(inst, seed=1)
        sol2 = genetic_algorithm(inst, seed=999)
        valid1, _ = validate_solution(inst, sol1)
        valid2, _ = validate_solution(inst, sol2)
        assert valid1 and valid2


class TestPMedianGAEdgeCases:
    """Test edge cases."""

    def test_p_equals_1(self):
        inst = PMedianInstance.random(n=5, m=5, p=1, seed=42)
        sol = genetic_algorithm(inst, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Invalid: {errors}"
        assert len(sol.open_facilities) == 1

    def test_time_limit(self):
        inst = PMedianInstance.random(n=10, m=10, p=3, seed=42)
        sol = genetic_algorithm(
            inst, generations=1000000, time_limit=0.5, seed=42
        )
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Invalid: {errors}"
