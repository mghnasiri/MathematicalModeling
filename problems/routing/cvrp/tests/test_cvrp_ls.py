"""Tests for Local Search on CVRP."""

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


_inst = _load_mod("cvrp_instance_test_ls", os.path.join(_parent_dir, "instance.py"))
CVRPInstance = _inst.CVRPInstance
CVRPSolution = _inst.CVRPSolution
validate_solution = _inst.validate_solution
small6 = _inst.small6
christofides1 = _inst.christofides1
medium12 = _inst.medium12

_ls = _load_mod(
    "cvrp_ls_test",
    os.path.join(_parent_dir, "metaheuristics", "local_search.py"),
)
local_search = _ls.local_search

_cw = _load_mod(
    "cvrp_cw_test_ls",
    os.path.join(_parent_dir, "heuristics", "clarke_wright.py"),
)
clarke_wright_savings = _cw.clarke_wright_savings


class TestCVRPLSValidity:
    """Test that LS produces valid solutions."""

    def test_small6_valid(self):
        inst = small6()
        sol = local_search(inst, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Invalid solution: {errors}"

    def test_christofides1_valid(self):
        inst = christofides1()
        sol = local_search(inst, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Invalid solution: {errors}"

    def test_random_valid(self):
        inst = CVRPInstance.random(n=10, seed=42)
        sol = local_search(inst, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Invalid solution: {errors}"

    def test_distance_matches(self):
        inst = small6()
        sol = local_search(inst, seed=42)
        actual = inst.total_distance(sol.routes)
        assert abs(sol.distance - actual) < 1e-6


class TestCVRPLSQuality:
    """Test solution quality."""

    def test_ls_competitive_with_cw(self):
        inst = small6()
        cw_sol = clarke_wright_savings(inst)
        ls_sol = local_search(inst, max_iterations=200, seed=42)
        assert ls_sol.distance <= cw_sol.distance * 1.05


class TestCVRPLSDeterminism:
    """Test deterministic behavior with seed."""

    def test_same_seed_same_result(self):
        inst = small6()
        sol1 = local_search(inst, max_iterations=100, seed=42)
        sol2 = local_search(inst, max_iterations=100, seed=42)
        assert abs(sol1.distance - sol2.distance) < 1e-6

    def test_different_seed_both_valid(self):
        inst = christofides1()
        sol1 = local_search(inst, max_iterations=50, seed=1)
        sol2 = local_search(inst, max_iterations=50, seed=999)
        valid1, _ = validate_solution(inst, sol1)
        valid2, _ = validate_solution(inst, sol2)
        assert valid1 and valid2


class TestCVRPLSEdgeCases:
    """Test edge cases."""

    def test_single_customer(self):
        dist = [[0, 10], [10, 0]]
        inst = CVRPInstance(
            n=1, capacity=100.0,
            demands=np.array([5.0]),
            distance_matrix=np.array(dist, dtype=float),
        )
        sol = local_search(inst, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Invalid: {errors}"

    def test_time_limit(self):
        inst = CVRPInstance.random(n=12, seed=42)
        sol = local_search(inst, max_iterations=1000000, time_limit=0.5, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Invalid: {errors}"
