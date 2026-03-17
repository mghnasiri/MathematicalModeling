"""Tests for Local Search on RCPSP."""

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


_inst = _load_mod("rcpsp_instance_test_ls", os.path.join(_parent_dir, "instance.py"))
RCPSPInstance = _inst.RCPSPInstance
RCPSPSolution = _inst.RCPSPSolution
validate_solution = _inst.validate_solution

_ls = _load_mod(
    "rcpsp_ls_test",
    os.path.join(_parent_dir, "metaheuristics", "local_search.py"),
)
local_search = _ls.local_search

_sgs = _load_mod(
    "rcpsp_sgs_test_ls",
    os.path.join(_parent_dir, "heuristics", "serial_sgs.py"),
)
serial_sgs = _sgs.serial_sgs


class TestRCPSPLSValidity:
    """Test that LS produces valid solutions."""

    def test_random_valid(self):
        inst = RCPSPInstance.random(n=8, seed=42)
        sol = local_search(inst, seed=42)
        valid, errors = validate_solution(inst, sol.start_times)
        assert valid, f"Invalid solution: {errors}"

    def test_larger_random_valid(self):
        inst = RCPSPInstance.random(n=12, seed=123)
        sol = local_search(inst, seed=42)
        valid, errors = validate_solution(inst, sol.start_times)
        assert valid, f"Invalid solution: {errors}"

    def test_makespan_above_critical_path(self):
        inst = RCPSPInstance.random(n=8, seed=42)
        sol = local_search(inst, seed=42)
        assert sol.makespan >= inst.critical_path_length()


class TestRCPSPLSQuality:
    """Test solution quality."""

    def test_ls_competitive_with_sgs(self):
        inst = RCPSPInstance.random(n=10, seed=42)
        sgs_sol = serial_sgs(inst, priority_rule="lft")
        ls_sol = local_search(inst, max_iterations=500, seed=42)
        assert ls_sol.makespan <= sgs_sol.makespan + 1

    def test_swap_neighborhood(self):
        inst = RCPSPInstance.random(n=8, seed=42)
        sol = local_search(inst, neighborhood="swap", seed=42)
        valid, errors = validate_solution(inst, sol.start_times)
        assert valid, f"Invalid: {errors}"

    def test_shift_neighborhood(self):
        inst = RCPSPInstance.random(n=8, seed=42)
        sol = local_search(inst, neighborhood="shift", seed=42)
        valid, errors = validate_solution(inst, sol.start_times)
        assert valid, f"Invalid: {errors}"


class TestRCPSPLSDeterminism:
    """Test deterministic behavior with seed."""

    def test_same_seed_same_result(self):
        inst = RCPSPInstance.random(n=8, seed=42)
        sol1 = local_search(inst, max_iterations=200, seed=42)
        sol2 = local_search(inst, max_iterations=200, seed=42)
        assert sol1.makespan == sol2.makespan

    def test_different_seed_both_valid(self):
        inst = RCPSPInstance.random(n=8, seed=42)
        sol1 = local_search(inst, seed=1)
        sol2 = local_search(inst, seed=999)
        valid1, _ = validate_solution(inst, sol1.start_times)
        valid2, _ = validate_solution(inst, sol2.start_times)
        assert valid1 and valid2


class TestRCPSPLSEdgeCases:
    """Test edge cases."""

    def test_time_limit(self):
        inst = RCPSPInstance.random(n=12, seed=42)
        sol = local_search(
            inst, max_iterations=1000000, time_limit=0.5, seed=42
        )
        valid, errors = validate_solution(inst, sol.start_times)
        assert valid, f"Invalid: {errors}"
