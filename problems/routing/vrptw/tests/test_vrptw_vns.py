"""Tests for Variable Neighborhood Search on VRPTW."""

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


_inst = _load_mod("vrptw_instance_test_vns", os.path.join(_parent_dir, "instance.py"))
VRPTWInstance = _inst.VRPTWInstance
VRPTWSolution = _inst.VRPTWSolution
validate_solution = _inst.validate_solution
solomon_c101_mini = _inst.solomon_c101_mini
tight_tw5 = _inst.tight_tw5

_vns = _load_mod(
    "vrptw_vns_test",
    os.path.join(_parent_dir, "metaheuristics", "vns.py"),
)
vns = _vns.vns

_sol = _load_mod(
    "vrptw_solomon_test_vns",
    os.path.join(_parent_dir, "heuristics", "solomon_insertion.py"),
)
solomon_insertion = _sol.solomon_insertion


class TestVRPTWVNSValidity:
    """Test that VNS produces valid solutions."""

    def test_solomon_mini_valid(self):
        inst = solomon_c101_mini()
        sol = vns(inst, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Invalid solution: {errors}"

    def test_tight_tw_valid(self):
        inst = tight_tw5()
        sol = vns(inst, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Invalid solution: {errors}"

    def test_random_valid(self):
        inst = VRPTWInstance.random(n=8, seed=42)
        sol = vns(inst, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Invalid solution: {errors}"

    def test_distance_matches(self):
        inst = solomon_c101_mini()
        sol = vns(inst, seed=42)
        actual = inst.total_distance(sol.routes)
        assert abs(sol.distance - actual) < 1e-6


class TestVRPTWVNSQuality:
    """Test solution quality."""

    def test_vns_competitive_with_solomon(self):
        inst = solomon_c101_mini()
        sol_sol = solomon_insertion(inst)
        vns_sol = vns(inst, max_iterations=200, seed=42)
        assert vns_sol.distance <= sol_sol.distance * 1.05


class TestVRPTWVNSDeterminism:
    """Test deterministic behavior with seed."""

    def test_same_seed_same_result(self):
        inst = solomon_c101_mini()
        sol1 = vns(inst, max_iterations=100, seed=42)
        sol2 = vns(inst, max_iterations=100, seed=42)
        assert abs(sol1.distance - sol2.distance) < 1e-6

    def test_different_seed_both_valid(self):
        inst = tight_tw5()
        sol1 = vns(inst, seed=1)
        sol2 = vns(inst, seed=999)
        valid1, _ = validate_solution(inst, sol1)
        valid2, _ = validate_solution(inst, sol2)
        assert valid1 and valid2


class TestVRPTWVNSEdgeCases:
    """Test edge cases."""

    def test_time_limit(self):
        inst = VRPTWInstance.random(n=10, seed=42)
        sol = vns(inst, max_iterations=1000000, time_limit=0.5, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Invalid: {errors}"
