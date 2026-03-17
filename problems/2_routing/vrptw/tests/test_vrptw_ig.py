"""Tests for Iterated Greedy on VRPTW."""

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


_inst = _load_mod("vrptw_instance_test_ig", os.path.join(_parent_dir, "instance.py"))
VRPTWInstance = _inst.VRPTWInstance
VRPTWSolution = _inst.VRPTWSolution
validate_solution = _inst.validate_solution

_ig = _load_mod(
    "vrptw_ig_test",
    os.path.join(_parent_dir, "metaheuristics", "iterated_greedy.py"),
)
iterated_greedy = _ig.iterated_greedy

_si = _load_mod(
    "vrptw_si_test_ig",
    os.path.join(_parent_dir, "heuristics", "solomon_insertion.py"),
)
solomon_insertion = _si.solomon_insertion


class TestVRPTWIGValidity:
    def test_all_customers_visited(self):
        inst = VRPTWInstance.random(n=8, seed=42)
        sol = iterated_greedy(inst, seed=42)
        all_cust = sorted(c for r in sol.routes for c in r)
        assert all_cust == list(range(1, inst.n + 1))

    def test_feasible_solution(self):
        inst = VRPTWInstance.random(n=8, seed=42)
        sol = iterated_greedy(inst, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Validation errors: {errors}"

    def test_distance_matches(self):
        inst = VRPTWInstance.random(n=8, seed=42)
        sol = iterated_greedy(inst, seed=42)
        actual = inst.total_distance(sol.routes)
        assert abs(sol.distance - actual) < 1e-6


class TestVRPTWIGQuality:
    def test_competitive_with_solomon(self):
        inst = VRPTWInstance.random(n=10, seed=42)
        si_sol = solomon_insertion(inst)
        ig_sol = iterated_greedy(inst, max_iterations=500, seed=42)
        assert ig_sol.distance <= si_sol.distance + 1e-6


class TestVRPTWIGDeterminism:
    def test_same_seed(self):
        inst = VRPTWInstance.random(n=8, seed=42)
        sol1 = iterated_greedy(inst, max_iterations=200, seed=42)
        sol2 = iterated_greedy(inst, max_iterations=200, seed=42)
        assert abs(sol1.distance - sol2.distance) < 1e-6

    def test_different_seed_both_valid(self):
        inst = VRPTWInstance.random(n=8, seed=42)
        sol1 = iterated_greedy(inst, max_iterations=200, seed=1)
        sol2 = iterated_greedy(inst, max_iterations=200, seed=999)
        valid1, _ = validate_solution(inst, sol1)
        valid2, _ = validate_solution(inst, sol2)
        assert valid1
        assert valid2


class TestVRPTWIGEdgeCases:
    def test_time_limit(self):
        inst = VRPTWInstance.random(n=12, seed=42)
        sol = iterated_greedy(inst, max_iterations=1000000, time_limit=0.5, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Validation errors: {errors}"
