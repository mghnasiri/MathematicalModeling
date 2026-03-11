"""Tests for VNS on 1D Cutting Stock."""

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


_inst = _load_mod("csp_instance_test_vns", os.path.join(_parent_dir, "instance.py"))
CuttingStockInstance = _inst.CuttingStockInstance
CuttingStockSolution = _inst.CuttingStockSolution
validate_solution = _inst.validate_solution

_vns = _load_mod(
    "csp_vns_test",
    os.path.join(_parent_dir, "metaheuristics", "vns.py"),
)
vns = _vns.vns

_greedy = _load_mod(
    "csp_greedy_test_vns",
    os.path.join(_parent_dir, "heuristics", "greedy_csp.py"),
)
greedy_largest_first = _greedy.greedy_largest_first


class TestCSPVNSValidity:
    def test_demands_satisfied(self):
        inst = CuttingStockInstance.random(m=4, seed=42)
        sol = vns(inst, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Validation errors: {errors}"

    def test_patterns_fit_stock(self):
        inst = CuttingStockInstance.random(m=4, seed=42)
        sol = vns(inst, seed=42)
        for pattern, freq in sol.patterns:
            total = float(np.dot(pattern, inst.lengths))
            assert total <= inst.stock_length + 1e-10

    def test_num_rolls_matches(self):
        inst = CuttingStockInstance.random(m=4, seed=42)
        sol = vns(inst, seed=42)
        actual = sum(f for _, f in sol.patterns)
        assert sol.num_rolls == actual


class TestCSPVNSQuality:
    def test_competitive_with_greedy(self):
        inst = CuttingStockInstance.random(m=5, seed=42)
        greedy_sol = greedy_largest_first(inst)
        vns_sol = vns(inst, max_iterations=200, seed=42)
        assert vns_sol.num_rolls <= greedy_sol.num_rolls

    def test_respects_lower_bound(self):
        inst = CuttingStockInstance.random(m=4, seed=42)
        sol = vns(inst, seed=42)
        assert sol.num_rolls >= inst.lower_bound()


class TestCSPVNSDeterminism:
    def test_same_seed(self):
        inst = CuttingStockInstance.random(m=4, seed=42)
        sol1 = vns(inst, max_iterations=100, seed=42)
        sol2 = vns(inst, max_iterations=100, seed=42)
        assert sol1.num_rolls == sol2.num_rolls


class TestCSPVNSEdgeCases:
    def test_time_limit(self):
        inst = CuttingStockInstance.random(m=5, seed=42)
        sol = vns(inst, max_iterations=1000000, time_limit=0.5, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Validation errors: {errors}"
