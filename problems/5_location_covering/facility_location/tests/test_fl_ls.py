"""Tests for Local Search on Facility Location."""

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


_inst = _load_mod("fl_instance_test_ls", os.path.join(_parent_dir, "instance.py"))
FacilityLocationInstance = _inst.FacilityLocationInstance
FacilityLocationSolution = _inst.FacilityLocationSolution
validate_solution = _inst.validate_solution

_ls = _load_mod(
    "fl_ls_test",
    os.path.join(_parent_dir, "metaheuristics", "local_search.py"),
)
local_search = _ls.local_search

_gr = _load_mod(
    "fl_gr_test_ls",
    os.path.join(_parent_dir, "heuristics", "greedy_facility.py"),
)
greedy_add = _gr.greedy_add


class TestFLLSValidity:
    def test_valid_solution(self):
        inst = FacilityLocationInstance.random(m=5, n=10, seed=42)
        sol = local_search(inst, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Validation errors: {errors}"

    def test_at_least_one_open(self):
        inst = FacilityLocationInstance.random(m=5, n=10, seed=42)
        sol = local_search(inst, seed=42)
        assert len(sol.open_facilities) >= 1

    def test_all_customers_assigned(self):
        inst = FacilityLocationInstance.random(m=5, n=10, seed=42)
        sol = local_search(inst, seed=42)
        assert len(sol.assignments) == inst.n


class TestFLLSQuality:
    def test_competitive_with_greedy(self):
        inst = FacilityLocationInstance.random(m=8, n=15, seed=42)
        gr_sol = greedy_add(inst)
        ls_sol = local_search(inst, max_iterations=200, seed=42)
        assert ls_sol.cost <= gr_sol.cost + 1e-6


class TestFLLSDeterminism:
    def test_same_seed(self):
        inst = FacilityLocationInstance.random(m=5, n=10, seed=42)
        sol1 = local_search(inst, max_iterations=100, seed=42)
        sol2 = local_search(inst, max_iterations=100, seed=42)
        assert abs(sol1.cost - sol2.cost) < 1e-6


class TestFLLSEdgeCases:
    def test_single_facility(self):
        inst = FacilityLocationInstance.random(m=1, n=5, seed=42)
        sol = local_search(inst, seed=42)
        assert sol.open_facilities == [0]
        assert len(sol.assignments) == 5

    def test_time_limit(self):
        inst = FacilityLocationInstance.random(m=8, n=15, seed=42)
        sol = local_search(inst, max_iterations=1000000, time_limit=0.5, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Validation errors: {errors}"
