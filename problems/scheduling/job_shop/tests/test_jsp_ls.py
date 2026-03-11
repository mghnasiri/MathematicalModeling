"""Tests for Local Search on Job Shop Scheduling."""

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


_inst = _load_mod("jsp_instance_test_ls", os.path.join(_parent_dir, "instance.py"))
JobShopInstance = _inst.JobShopInstance
JobShopSolution = _inst.JobShopSolution
validate_solution = _inst.validate_solution
ft06 = _inst.ft06

_ls = _load_mod(
    "jsp_ls_test",
    os.path.join(_parent_dir, "metaheuristics", "local_search.py"),
)
local_search = _ls.local_search

_disp = _load_mod(
    "jsp_disp_test_ls",
    os.path.join(_parent_dir, "heuristics", "dispatching_rules.py"),
)
dispatching_rule = _disp.dispatching_rule


class TestJSPLSValidity:
    def test_feasible_schedule(self):
        inst = JobShopInstance.random(n=5, m=3, seed=42)
        sol = local_search(inst, seed=42)
        valid, errors = validate_solution(inst, sol.start_times)
        assert valid, f"Validation errors: {errors}"

    def test_makespan_positive(self):
        inst = JobShopInstance.random(n=5, m=3, seed=42)
        sol = local_search(inst, seed=42)
        assert sol.makespan > 0


class TestJSPLSQuality:
    def test_competitive_with_dispatching(self):
        inst = JobShopInstance.random(n=6, m=4, seed=42)
        disp_sol = dispatching_rule(inst, rule="spt")
        ls_sol = local_search(inst, max_iterations=200, seed=42)
        assert ls_sol.makespan <= disp_sol.makespan

    def test_ft06_reasonable(self):
        inst = ft06()
        sol = local_search(inst, max_iterations=500, seed=42)
        assert sol.makespan <= 70  # optimal is 55, should be close


class TestJSPLSDeterminism:
    def test_same_seed(self):
        inst = JobShopInstance.random(n=5, m=3, seed=42)
        sol1 = local_search(inst, max_iterations=100, seed=42)
        sol2 = local_search(inst, max_iterations=100, seed=42)
        assert sol1.makespan == sol2.makespan

    def test_different_seed_both_valid(self):
        inst = JobShopInstance.random(n=5, m=3, seed=42)
        sol1 = local_search(inst, seed=1)
        sol2 = local_search(inst, seed=999)
        valid1, _ = validate_solution(inst, sol1.start_times)
        valid2, _ = validate_solution(inst, sol2.start_times)
        assert valid1
        assert valid2


class TestJSPLSEdgeCases:
    def test_time_limit(self):
        inst = JobShopInstance.random(n=6, m=4, seed=42)
        sol = local_search(inst, max_iterations=1000000, time_limit=0.5, seed=42)
        valid, errors = validate_solution(inst, sol.start_times)
        assert valid, f"Validation errors: {errors}"
