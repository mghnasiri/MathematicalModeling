"""Tests for Iterated Greedy on Flexible Job Shop Scheduling."""

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


_inst = _load_mod("fjsp_instance_test_ig", os.path.join(_parent_dir, "instance.py"))
FlexibleJobShopInstance = _inst.FlexibleJobShopInstance
validate_solution = _inst.validate_solution

_ig = _load_mod(
    "fjsp_ig_test",
    os.path.join(_parent_dir, "metaheuristics", "iterated_greedy.py"),
)
iterated_greedy = _ig.iterated_greedy

_disp = _load_mod(
    "fjsp_disp_test_ig",
    os.path.join(_parent_dir, "heuristics", "dispatching_rules.py"),
)
dispatching_rule = _disp.dispatching_rule


class TestFJSPIGValidity:
    def test_random_valid(self):
        inst = FlexibleJobShopInstance.random(n=4, m=3, seed=42)
        sol = iterated_greedy(inst, seed=42)
        valid, errors = validate_solution(inst, sol.assignments, sol.start_times)
        assert valid, f"Invalid: {errors}"

    def test_total_fjsp_valid(self):
        inst = FlexibleJobShopInstance.random_total(n=4, m=3, seed=42)
        sol = iterated_greedy(inst, seed=42)
        valid, errors = validate_solution(inst, sol.assignments, sol.start_times)
        assert valid, f"Invalid: {errors}"


class TestFJSPIGQuality:
    def test_ig_competitive(self):
        inst = FlexibleJobShopInstance.random(n=5, m=3, seed=42)
        disp_sol = dispatching_rule(inst, priority_rule="spt")
        ig_sol = iterated_greedy(inst, max_iterations=500, seed=42)
        assert ig_sol.makespan <= disp_sol.makespan * 1.05


class TestFJSPIGDeterminism:
    def test_same_seed(self):
        inst = FlexibleJobShopInstance.random(n=4, m=3, seed=42)
        sol1 = iterated_greedy(inst, max_iterations=100, seed=42)
        sol2 = iterated_greedy(inst, max_iterations=100, seed=42)
        assert sol1.makespan == sol2.makespan

    def test_different_seed_both_valid(self):
        inst = FlexibleJobShopInstance.random(n=4, m=3, seed=42)
        sol1 = iterated_greedy(inst, seed=1)
        sol2 = iterated_greedy(inst, seed=999)
        v1, _ = validate_solution(inst, sol1.assignments, sol1.start_times)
        v2, _ = validate_solution(inst, sol2.assignments, sol2.start_times)
        assert v1 and v2


class TestFJSPIGEdgeCases:
    def test_time_limit(self):
        inst = FlexibleJobShopInstance.random(n=6, m=4, seed=42)
        sol = iterated_greedy(inst, max_iterations=1000000, time_limit=0.5, seed=42)
        valid, errors = validate_solution(inst, sol.assignments, sol.start_times)
        assert valid, f"Invalid: {errors}"
