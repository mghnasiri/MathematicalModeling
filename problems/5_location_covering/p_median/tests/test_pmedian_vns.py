"""Tests for Variable Neighborhood Search on p-Median."""

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


_inst = _load_mod("pmedian_instance_test_vns", os.path.join(_parent_dir, "instance.py"))
PMedianInstance = _inst.PMedianInstance
PMedianSolution = _inst.PMedianSolution
validate_solution = _inst.validate_solution
small_pmedian_6_2 = _inst.small_pmedian_6_2

_vns = _load_mod(
    "pmedian_vns_test",
    os.path.join(_parent_dir, "metaheuristics", "vns.py"),
)
vns = _vns.vns

_greedy = _load_mod(
    "pm_greedy_test_vns",
    os.path.join(_parent_dir, "heuristics", "greedy_pmedian.py"),
)
greedy_pmedian = _greedy.greedy_pmedian


class TestPMedianVNSValidity:
    """Test that VNS produces valid solutions."""

    def test_small_valid(self):
        inst = small_pmedian_6_2()
        sol = vns(inst, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Invalid solution: {errors}"

    def test_random_valid(self):
        inst = PMedianInstance.random(n=10, m=10, p=3, seed=42)
        sol = vns(inst, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Invalid solution: {errors}"

    def test_correct_p(self):
        inst = small_pmedian_6_2()
        sol = vns(inst, seed=42)
        assert len(sol.open_facilities) == inst.p


class TestPMedianVNSQuality:
    """Test solution quality."""

    def test_vns_competitive_with_greedy(self):
        inst = PMedianInstance.random(n=12, m=12, p=3, seed=42)
        greedy_sol = greedy_pmedian(inst)
        vns_sol = vns(inst, max_iterations=200, seed=42)
        assert vns_sol.cost <= greedy_sol.cost + 1e-6

    def test_cost_matches(self):
        inst = small_pmedian_6_2()
        sol = vns(inst, seed=42)
        actual = inst.total_cost(sol.open_facilities, sol.assignments)
        assert abs(sol.cost - actual) < 1e-4


class TestPMedianVNSDeterminism:
    """Test deterministic behavior with seed."""

    def test_same_seed_same_result(self):
        inst = small_pmedian_6_2()
        sol1 = vns(inst, max_iterations=100, seed=42)
        sol2 = vns(inst, max_iterations=100, seed=42)
        assert abs(sol1.cost - sol2.cost) < 1e-6

    def test_different_seed_both_valid(self):
        inst = PMedianInstance.random(n=8, m=8, p=2, seed=42)
        sol1 = vns(inst, seed=1)
        sol2 = vns(inst, seed=999)
        valid1, _ = validate_solution(inst, sol1)
        valid2, _ = validate_solution(inst, sol2)
        assert valid1 and valid2


class TestPMedianVNSEdgeCases:
    """Test edge cases."""

    def test_p_equals_m(self):
        inst = PMedianInstance.random(n=5, m=3, p=3, seed=42)
        sol = vns(inst, seed=42)
        assert len(sol.open_facilities) == 3
        assert set(sol.open_facilities) == {0, 1, 2}

    def test_time_limit(self):
        inst = PMedianInstance.random(n=15, m=15, p=4, seed=42)
        sol = vns(inst, max_iterations=1000000, time_limit=0.5, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Invalid: {errors}"
