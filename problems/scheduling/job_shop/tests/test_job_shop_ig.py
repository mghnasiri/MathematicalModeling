"""Tests for Iterated Greedy on Job Shop Scheduling."""

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


_inst = _load_mod("jsp_instance_test_ig", os.path.join(_parent_dir, "instance.py"))
JobShopInstance = _inst.JobShopInstance
JobShopSolution = _inst.JobShopSolution
validate_solution = _inst.validate_solution
ft06 = _inst.ft06

_ig = _load_mod(
    "jsp_ig_test",
    os.path.join(_parent_dir, "metaheuristics", "iterated_greedy.py"),
)
iterated_greedy = _ig.iterated_greedy

_disp = _load_mod(
    "jsp_disp_test_ig",
    os.path.join(_parent_dir, "heuristics", "dispatching_rules.py"),
)
dispatching_rule = _disp.dispatching_rule


class TestJSPIGValidity:
    """Test that IG produces valid solutions."""

    def test_ft06_valid(self):
        inst = ft06()
        sol = iterated_greedy(inst, seed=42)
        valid, errors = validate_solution(inst, sol.start_times)
        assert valid, f"Invalid solution: {errors}"

    def test_random_valid(self):
        inst = JobShopInstance.random(n=6, m=4, seed=42)
        sol = iterated_greedy(inst, seed=42)
        valid, errors = validate_solution(inst, sol.start_times)
        assert valid, f"Invalid solution: {errors}"

    def test_all_operations_scheduled(self):
        inst = ft06()
        sol = iterated_greedy(inst, seed=42)
        for j in range(inst.n):
            for k in range(len(inst.jobs[j])):
                assert (j, k) in sol.start_times


class TestJSPIGQuality:
    """Test solution quality."""

    def test_ft06_reasonable(self):
        inst = ft06()
        sol = iterated_greedy(inst, max_iterations=500, seed=42)
        assert sol.makespan >= 55  # Optimal = 55
        assert sol.makespan <= 80  # Should be reasonable

    def test_ig_competitive_with_dispatching(self):
        inst = JobShopInstance.random(n=6, m=4, seed=42)
        disp_sol = dispatching_rule(inst, rule="spt")
        ig_sol = iterated_greedy(inst, max_iterations=500, seed=42)
        assert ig_sol.makespan <= disp_sol.makespan * 1.1


class TestJSPIGDeterminism:
    """Test deterministic behavior with seed."""

    def test_same_seed_same_result(self):
        inst = ft06()
        sol1 = iterated_greedy(inst, max_iterations=200, seed=42)
        sol2 = iterated_greedy(inst, max_iterations=200, seed=42)
        assert sol1.makespan == sol2.makespan

    def test_different_seed_both_valid(self):
        inst = JobShopInstance.random(n=5, m=3, seed=42)
        sol1 = iterated_greedy(inst, seed=1)
        sol2 = iterated_greedy(inst, seed=999)
        valid1, _ = validate_solution(inst, sol1.start_times)
        valid2, _ = validate_solution(inst, sol2.start_times)
        assert valid1 and valid2


class TestJSPIGEdgeCases:
    """Test edge cases."""

    def test_time_limit(self):
        inst = JobShopInstance.random(n=8, m=5, seed=42)
        sol = iterated_greedy(
            inst, max_iterations=1000000, time_limit=0.5, seed=42
        )
        valid, errors = validate_solution(inst, sol.start_times)
        assert valid, f"Invalid: {errors}"
