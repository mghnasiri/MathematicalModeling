"""Tests for Iterated Greedy on Facility Location."""

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


_inst = _load_mod("fl_instance_test_ig", os.path.join(_parent_dir, "instance.py"))
FacilityLocationInstance = _inst.FacilityLocationInstance
validate_solution = _inst.validate_solution

_ig = _load_mod(
    "fl_ig_test",
    os.path.join(_parent_dir, "metaheuristics", "iterated_greedy.py"),
)
iterated_greedy = _ig.iterated_greedy

_gr = _load_mod(
    "fl_gr_test_ig",
    os.path.join(_parent_dir, "heuristics", "greedy_facility.py"),
)
greedy_add = _gr.greedy_add


class TestFLIGValidity:
    def test_valid_solution(self):
        inst = FacilityLocationInstance.random(m=5, n=10, seed=42)
        sol = iterated_greedy(inst, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Validation errors: {errors}"

    def test_at_least_one_open(self):
        inst = FacilityLocationInstance.random(m=5, n=10, seed=42)
        sol = iterated_greedy(inst, seed=42)
        assert len(sol.open_facilities) >= 1


class TestFLIGQuality:
    def test_competitive_with_greedy(self):
        inst = FacilityLocationInstance.random(m=8, n=15, seed=42)
        gr_sol = greedy_add(inst)
        ig_sol = iterated_greedy(inst, max_iterations=500, seed=42)
        assert ig_sol.cost <= gr_sol.cost + 1e-6


class TestFLIGDeterminism:
    def test_same_seed(self):
        inst = FacilityLocationInstance.random(m=5, n=10, seed=42)
        sol1 = iterated_greedy(inst, max_iterations=200, seed=42)
        sol2 = iterated_greedy(inst, max_iterations=200, seed=42)
        assert abs(sol1.cost - sol2.cost) < 1e-6


class TestFLIGEdgeCases:
    def test_single_facility(self):
        inst = FacilityLocationInstance.random(m=1, n=5, seed=42)
        sol = iterated_greedy(inst, seed=42)
        assert sol.open_facilities == [0]

    def test_time_limit(self):
        inst = FacilityLocationInstance.random(m=8, n=15, seed=42)
        sol = iterated_greedy(inst, max_iterations=1000000, time_limit=0.5, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Validation errors: {errors}"
