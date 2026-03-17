"""Tests for Variable Neighborhood Search on Flexible Job Shop Scheduling."""

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


_inst = _load_mod("fjsp_instance_test_vns", os.path.join(_parent_dir, "instance.py"))
FlexibleJobShopInstance = _inst.FlexibleJobShopInstance
FlexibleJobShopSolution = _inst.FlexibleJobShopSolution
validate_solution = _inst.validate_solution

_vns = _load_mod(
    "fjsp_vns_test",
    os.path.join(_parent_dir, "metaheuristics", "vns.py"),
)
vns = _vns.vns

_disp = _load_mod(
    "fjsp_disp_test_vns",
    os.path.join(_parent_dir, "heuristics", "dispatching_rules.py"),
)
dispatching_rule = _disp.dispatching_rule


class TestFJSPVNSValidity:
    """Test that VNS produces valid solutions."""

    def test_random_valid(self):
        inst = FlexibleJobShopInstance.random(n=4, m=3, seed=42)
        sol = vns(inst, seed=42)
        valid, errors = validate_solution(inst, sol.assignments, sol.start_times)
        assert valid, f"Invalid solution: {errors}"

    def test_total_fjsp_valid(self):
        inst = FlexibleJobShopInstance.random_total(n=4, m=3, seed=42)
        sol = vns(inst, seed=42)
        valid, errors = validate_solution(inst, sol.assignments, sol.start_times)
        assert valid, f"Invalid solution: {errors}"

    def test_all_operations_assigned(self):
        inst = FlexibleJobShopInstance.random(n=4, m=3, seed=42)
        sol = vns(inst, seed=42)
        for j in range(inst.n):
            for k in range(len(inst.jobs[j])):
                assert (j, k) in sol.assignments
                assert (j, k) in sol.start_times


class TestFJSPVNSQuality:
    """Test solution quality."""

    def test_vns_competitive_with_dispatching(self):
        inst = FlexibleJobShopInstance.random(n=5, m=3, seed=42)
        disp_sol = dispatching_rule(inst, priority_rule="spt")
        vns_sol = vns(inst, max_iterations=200, seed=42)
        assert vns_sol.makespan <= disp_sol.makespan * 1.05


class TestFJSPVNSDeterminism:
    """Test deterministic behavior with seed."""

    def test_same_seed_same_result(self):
        inst = FlexibleJobShopInstance.random(n=4, m=3, seed=42)
        sol1 = vns(inst, max_iterations=100, seed=42)
        sol2 = vns(inst, max_iterations=100, seed=42)
        assert sol1.makespan == sol2.makespan

    def test_different_seed_both_valid(self):
        inst = FlexibleJobShopInstance.random(n=4, m=3, seed=42)
        sol1 = vns(inst, seed=1)
        sol2 = vns(inst, seed=999)
        valid1, _ = validate_solution(inst, sol1.assignments, sol1.start_times)
        valid2, _ = validate_solution(inst, sol2.assignments, sol2.start_times)
        assert valid1 and valid2


class TestFJSPVNSEdgeCases:
    """Test edge cases."""

    def test_time_limit(self):
        inst = FlexibleJobShopInstance.random(n=6, m=4, seed=42)
        sol = vns(inst, max_iterations=1000000, time_limit=0.5, seed=42)
        valid, errors = validate_solution(inst, sol.assignments, sol.start_times)
        assert valid, f"Invalid: {errors}"
