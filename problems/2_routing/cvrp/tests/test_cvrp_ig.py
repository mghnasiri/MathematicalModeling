"""Tests for Iterated Greedy on CVRP."""

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


_inst = _load_mod("cvrp_instance_test_ig", os.path.join(_parent_dir, "instance.py"))
CVRPInstance = _inst.CVRPInstance
CVRPSolution = _inst.CVRPSolution
validate_solution = _inst.validate_solution
small6 = _inst.small6
christofides1 = _inst.christofides1

_ig = _load_mod(
    "cvrp_ig_test",
    os.path.join(_parent_dir, "metaheuristics", "iterated_greedy.py"),
)
iterated_greedy = _ig.iterated_greedy

_cw = _load_mod(
    "cvrp_cw_test_ig",
    os.path.join(_parent_dir, "heuristics", "clarke_wright.py"),
)
clarke_wright_savings = _cw.clarke_wright_savings


class TestCVRPIGValidity:
    def test_small6_valid(self):
        inst = small6()
        sol = iterated_greedy(inst, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Invalid: {errors}"

    def test_christofides1_valid(self):
        inst = christofides1()
        sol = iterated_greedy(inst, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Invalid: {errors}"

    def test_random_valid(self):
        inst = CVRPInstance.random(n=10, seed=42)
        sol = iterated_greedy(inst, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Invalid: {errors}"

    def test_distance_matches(self):
        inst = small6()
        sol = iterated_greedy(inst, seed=42)
        actual = inst.total_distance(sol.routes)
        assert abs(sol.distance - actual) < 1e-6


class TestCVRPIGQuality:
    def test_ig_competitive_with_cw(self):
        inst = small6()
        cw_sol = clarke_wright_savings(inst)
        ig_sol = iterated_greedy(inst, max_iterations=500, seed=42)
        assert ig_sol.distance <= cw_sol.distance * 1.05


class TestCVRPIGDeterminism:
    def test_same_seed(self):
        inst = small6()
        sol1 = iterated_greedy(inst, max_iterations=200, seed=42)
        sol2 = iterated_greedy(inst, max_iterations=200, seed=42)
        assert abs(sol1.distance - sol2.distance) < 1e-6

    def test_different_seed_both_valid(self):
        inst = CVRPInstance.random(n=8, seed=42)
        sol1 = iterated_greedy(inst, seed=1)
        sol2 = iterated_greedy(inst, seed=999)
        v1, _ = validate_solution(inst, sol1)
        v2, _ = validate_solution(inst, sol2)
        assert v1 and v2


class TestCVRPIGEdgeCases:
    def test_time_limit(self):
        inst = CVRPInstance.random(n=12, seed=42)
        sol = iterated_greedy(inst, max_iterations=1000000, time_limit=0.5, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Invalid: {errors}"
