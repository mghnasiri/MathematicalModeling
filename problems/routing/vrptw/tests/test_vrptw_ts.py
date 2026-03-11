"""Tests for Tabu Search on VRPTW."""

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


_inst = _load_mod("vrptw_instance_test_ts", os.path.join(_parent_dir, "instance.py"))
VRPTWInstance = _inst.VRPTWInstance
VRPTWSolution = _inst.VRPTWSolution
validate_solution = _inst.validate_solution
solomon_c101_mini = _inst.solomon_c101_mini
tight_tw5 = _inst.tight_tw5

_ts = _load_mod(
    "vrptw_ts_test",
    os.path.join(_parent_dir, "metaheuristics", "tabu_search.py"),
)
tabu_search = _ts.tabu_search


class TestVRPTWTSValidity:
    """Test that TS produces valid solutions."""

    def test_solomon_mini_valid(self):
        inst = solomon_c101_mini()
        sol = tabu_search(inst, max_iterations=200, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Invalid solution: {errors}"

    def test_tight_tw5_valid(self):
        inst = tight_tw5()
        sol = tabu_search(inst, max_iterations=200, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Invalid solution: {errors}"

    def test_random_instance_valid(self):
        inst = VRPTWInstance.random(n=8, seed=42)
        sol = tabu_search(inst, max_iterations=200, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Invalid solution: {errors}"

    def test_all_customers_served(self):
        inst = solomon_c101_mini()
        sol = tabu_search(inst, max_iterations=200, seed=42)
        served = set()
        for route in sol.routes:
            served.update(route)
        assert served == set(range(1, inst.n + 1))


class TestVRPTWTSQuality:
    """Test solution quality."""

    def test_distance_positive(self):
        inst = solomon_c101_mini()
        sol = tabu_search(inst, max_iterations=200, seed=42)
        assert sol.distance > 0

    def test_distance_matches_routes(self):
        inst = solomon_c101_mini()
        sol = tabu_search(inst, max_iterations=200, seed=42)
        actual = inst.total_distance(sol.routes)
        assert abs(sol.distance - actual) < 1e-6


class TestVRPTWTSDeterminism:
    """Test deterministic behavior with seed."""

    def test_same_seed_same_result(self):
        inst = tight_tw5()
        sol1 = tabu_search(inst, max_iterations=200, seed=42)
        sol2 = tabu_search(inst, max_iterations=200, seed=42)
        assert abs(sol1.distance - sol2.distance) < 1e-6

    def test_different_seed_both_valid(self):
        inst = tight_tw5()
        sol1 = tabu_search(inst, max_iterations=200, seed=1)
        sol2 = tabu_search(inst, max_iterations=200, seed=999)
        valid1, _ = validate_solution(inst, sol1)
        valid2, _ = validate_solution(inst, sol2)
        assert valid1 and valid2


class TestVRPTWTSEdgeCases:
    """Test edge cases."""

    def test_time_limit(self):
        inst = VRPTWInstance.random(n=8, seed=42)
        sol = tabu_search(
            inst, max_iterations=1000000, time_limit=0.5, seed=42
        )
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Invalid: {errors}"
