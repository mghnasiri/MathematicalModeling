"""Tests for Variable Neighborhood Search on CVRP."""

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


_inst = _load_mod("cvrp_instance_test_vns", os.path.join(_parent_dir, "instance.py"))
CVRPInstance = _inst.CVRPInstance
CVRPSolution = _inst.CVRPSolution
validate_solution = _inst.validate_solution
small6 = _inst.small6
christofides1 = _inst.christofides1
medium12 = _inst.medium12

_vns = _load_mod(
    "cvrp_vns_test",
    os.path.join(_parent_dir, "metaheuristics", "vns.py"),
)
vns = _vns.vns

_cw = _load_mod(
    "cvrp_cw_test_vns",
    os.path.join(_parent_dir, "heuristics", "clarke_wright.py"),
)
clarke_wright_savings = _cw.clarke_wright_savings


class TestCVRPVNSValidity:
    """Test that VNS produces valid solutions."""

    def test_small6_valid(self):
        inst = small6()
        sol = vns(inst, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Invalid solution: {errors}"

    def test_christofides1_valid(self):
        inst = christofides1()
        sol = vns(inst, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Invalid solution: {errors}"

    def test_random_instance_valid(self):
        inst = CVRPInstance.random(n=10, seed=42)
        sol = vns(inst, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Invalid solution: {errors}"

    def test_distance_matches(self):
        inst = small6()
        sol = vns(inst, seed=42)
        actual = inst.total_distance(sol.routes)
        assert abs(sol.distance - actual) < 1e-6


class TestCVRPVNSQuality:
    """Test solution quality."""

    def test_vns_competitive_with_cw(self):
        inst = small6()
        cw_sol = clarke_wright_savings(inst)
        vns_sol = vns(inst, max_iterations=200, seed=42)
        assert vns_sol.distance <= cw_sol.distance * 1.05

    def test_medium12_reasonable(self):
        inst = medium12()
        sol = vns(inst, max_iterations=200, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Invalid: {errors}"


class TestCVRPVNSDeterminism:
    """Test deterministic behavior with seed."""

    def test_same_seed_same_result(self):
        inst = small6()
        sol1 = vns(inst, max_iterations=100, seed=42)
        sol2 = vns(inst, max_iterations=100, seed=42)
        assert abs(sol1.distance - sol2.distance) < 1e-6

    def test_different_seed_both_valid(self):
        inst = christofides1()
        sol1 = vns(inst, max_iterations=50, seed=1)
        sol2 = vns(inst, max_iterations=50, seed=999)
        valid1, _ = validate_solution(inst, sol1)
        valid2, _ = validate_solution(inst, sol2)
        assert valid1 and valid2


class TestCVRPVNSEdgeCases:
    """Test edge cases."""

    def test_single_customer(self):
        dist = [[0, 10], [10, 0]]
        inst = CVRPInstance(
            n=1, capacity=100.0,
            demands=np.array([5.0]),
            distance_matrix=np.array(dist, dtype=float),
        )
        sol = vns(inst, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Invalid: {errors}"

    def test_time_limit(self):
        inst = CVRPInstance.random(n=12, seed=42)
        sol = vns(inst, max_iterations=1000000, time_limit=0.5, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Invalid: {errors}"
