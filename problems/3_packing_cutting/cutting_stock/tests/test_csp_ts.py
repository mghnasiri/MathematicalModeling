"""Tests for Tabu Search on 1D Cutting Stock Problem."""

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


_inst = _load_mod("csp_instance_test_ts", os.path.join(_parent_dir, "instance.py"))
CuttingStockInstance = _inst.CuttingStockInstance
CuttingStockSolution = _inst.CuttingStockSolution
validate_solution = _inst.validate_solution
simple_csp_3 = _inst.simple_csp_3
classic_csp_4 = _inst.classic_csp_4

_ts = _load_mod(
    "csp_ts_test",
    os.path.join(_parent_dir, "metaheuristics", "tabu_search.py"),
)
tabu_search = _ts.tabu_search


class TestCSPTSValidity:
    """Test that TS produces valid solutions."""

    def test_simple3_valid(self):
        inst = simple_csp_3()
        sol = tabu_search(inst, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Invalid solution: {errors}"

    def test_classic4_valid(self):
        inst = classic_csp_4()
        sol = tabu_search(inst, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Invalid solution: {errors}"

    def test_random_instance_valid(self):
        inst = CuttingStockInstance.random(m=5, stock_length=100, seed=123)
        sol = tabu_search(inst, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Invalid solution: {errors}"


class TestCSPTSQuality:
    """Test solution quality."""

    def test_simple3_meets_lower_bound(self):
        inst = simple_csp_3()
        sol = tabu_search(inst, max_iterations=3000, seed=42)
        lb = inst.lower_bound()
        assert sol.num_rolls >= lb

    def test_ts_not_worse_than_2x_lb(self):
        inst = CuttingStockInstance.random(m=6, stock_length=100, seed=99)
        sol = tabu_search(inst, max_iterations=3000, seed=42)
        lb = inst.lower_bound()
        assert sol.num_rolls <= 2 * lb + 1


class TestCSPTSDeterminism:
    """Test deterministic behavior with seed."""

    def test_same_seed_same_result(self):
        inst = simple_csp_3()
        sol1 = tabu_search(inst, seed=42)
        sol2 = tabu_search(inst, seed=42)
        assert sol1.num_rolls == sol2.num_rolls

    def test_different_seed_both_valid(self):
        inst = classic_csp_4()
        sol1 = tabu_search(inst, seed=1)
        sol2 = tabu_search(inst, seed=999)
        valid1, _ = validate_solution(inst, sol1)
        valid2, _ = validate_solution(inst, sol2)
        assert valid1 and valid2


class TestCSPTSEdgeCases:
    """Test edge cases."""

    def test_zero_demands(self):
        inst = CuttingStockInstance(
            m=2, stock_length=100.0,
            lengths=np.array([30.0, 40.0]),
            demands=np.array([0, 0]),
            name="zero_demand",
        )
        sol = tabu_search(inst, seed=42)
        assert sol.num_rolls == 0

    def test_time_limit(self):
        inst = CuttingStockInstance.random(m=5, stock_length=100, seed=42)
        sol = tabu_search(
            inst, max_iterations=1000000, time_limit=0.5, seed=42
        )
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Invalid: {errors}"
