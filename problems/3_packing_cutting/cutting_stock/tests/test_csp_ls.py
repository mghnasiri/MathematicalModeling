"""Tests for Local Search on Cutting Stock."""

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


_inst = _load_mod("csp_instance_test_ls", os.path.join(_parent_dir, "instance.py"))
CuttingStockInstance = _inst.CuttingStockInstance
CuttingStockSolution = _inst.CuttingStockSolution
validate_solution = _inst.validate_solution
simple_csp_3 = _inst.simple_csp_3
classic_csp_4 = _inst.classic_csp_4

_ls = _load_mod(
    "csp_ls_test",
    os.path.join(_parent_dir, "metaheuristics", "local_search.py"),
)
local_search = _ls.local_search

_greedy = _load_mod(
    "csp_greedy_test_ls",
    os.path.join(_parent_dir, "heuristics", "greedy_csp.py"),
)
ffd_based = _greedy.ffd_based


class TestCSPLSValidity:
    """Test that LS produces valid solutions."""

    def test_simple3_valid(self):
        inst = simple_csp_3()
        sol = local_search(inst, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Invalid solution: {errors}"

    def test_classic4_valid(self):
        inst = classic_csp_4()
        sol = local_search(inst, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Invalid solution: {errors}"

    def test_random_valid(self):
        inst = CuttingStockInstance.random(m=5, seed=42)
        sol = local_search(inst, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Invalid solution: {errors}"

    def test_above_lower_bound(self):
        inst = simple_csp_3()
        sol = local_search(inst, seed=42)
        assert sol.num_rolls >= inst.lower_bound()


class TestCSPLSQuality:
    """Test solution quality."""

    def test_ls_competitive_with_ffd(self):
        inst = CuttingStockInstance.random(m=5, seed=42)
        ffd_sol = ffd_based(inst)
        ls_sol = local_search(inst, max_iterations=200, seed=42)
        assert ls_sol.num_rolls <= ffd_sol.num_rolls + 1


class TestCSPLSDeterminism:
    """Test deterministic behavior with seed."""

    def test_same_seed_same_result(self):
        inst = simple_csp_3()
        sol1 = local_search(inst, max_iterations=100, seed=42)
        sol2 = local_search(inst, max_iterations=100, seed=42)
        assert sol1.num_rolls == sol2.num_rolls

    def test_different_seed_both_valid(self):
        inst = classic_csp_4()
        sol1 = local_search(inst, seed=1)
        sol2 = local_search(inst, seed=999)
        valid1, _ = validate_solution(inst, sol1)
        valid2, _ = validate_solution(inst, sol2)
        assert valid1 and valid2


class TestCSPLSEdgeCases:
    """Test edge cases."""

    def test_single_type(self):
        inst = CuttingStockInstance(
            m=1,
            stock_length=100.0,
            lengths=np.array([40.0]),
            demands=np.array([5]),
        )
        sol = local_search(inst, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Invalid: {errors}"

    def test_time_limit(self):
        inst = CuttingStockInstance.random(m=6, seed=42)
        sol = local_search(inst, max_iterations=1000000, time_limit=0.5, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Invalid: {errors}"
