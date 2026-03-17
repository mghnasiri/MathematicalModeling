"""Tests for Local Search on 1D Bin Packing."""

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


_inst = _load_mod("bpp_instance_test_ls", os.path.join(_parent_dir, "instance.py"))
BinPackingInstance = _inst.BinPackingInstance
BinPackingSolution = _inst.BinPackingSolution
validate_solution = _inst.validate_solution
easy_bpp_6 = _inst.easy_bpp_6
tight_bpp_8 = _inst.tight_bpp_8
uniform_bpp_10 = _inst.uniform_bpp_10

_ls = _load_mod(
    "bpp_ls_test",
    os.path.join(_parent_dir, "metaheuristics", "local_search.py"),
)
local_search = _ls.local_search

_ff = _load_mod(
    "bpp_ff_test_ls",
    os.path.join(_parent_dir, "heuristics", "first_fit.py"),
)
first_fit_decreasing = _ff.first_fit_decreasing


class TestBPPLSValidity:
    """Test that LS produces valid solutions."""

    def test_easy6_valid(self):
        inst = easy_bpp_6()
        sol = local_search(inst, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Invalid solution: {errors}"

    def test_tight8_valid(self):
        inst = tight_bpp_8()
        sol = local_search(inst, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Invalid solution: {errors}"

    def test_random_valid(self):
        inst = BinPackingInstance.random(n=15, seed=42)
        sol = local_search(inst, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Invalid solution: {errors}"


class TestBPPLSQuality:
    """Test solution quality."""

    def test_ls_competitive_with_ffd(self):
        inst = BinPackingInstance.random(n=20, seed=42)
        ffd_sol = first_fit_decreasing(inst)
        ls_sol = local_search(inst, seed=42)
        assert ls_sol.num_bins <= ffd_sol.num_bins

    def test_meets_lower_bound(self):
        inst = easy_bpp_6()
        sol = local_search(inst, seed=42)
        assert sol.num_bins >= inst.lower_bound_l1()

    def test_uniform_optimal(self):
        inst = uniform_bpp_10()
        sol = local_search(inst, seed=42)
        assert sol.num_bins == 5  # Known optimal


class TestBPPLSDeterminism:
    """Test deterministic behavior with seed."""

    def test_same_seed_same_result(self):
        inst = tight_bpp_8()
        sol1 = local_search(inst, seed=42)
        sol2 = local_search(inst, seed=42)
        assert sol1.num_bins == sol2.num_bins

    def test_different_seed_both_valid(self):
        inst = BinPackingInstance.random(n=10, seed=42)
        sol1 = local_search(inst, seed=1)
        sol2 = local_search(inst, seed=999)
        valid1, _ = validate_solution(inst, sol1)
        valid2, _ = validate_solution(inst, sol2)
        assert valid1 and valid2


class TestBPPLSEdgeCases:
    """Test edge cases."""

    def test_single_item(self):
        inst = BinPackingInstance(n=1, sizes=np.array([5.0]), capacity=10.0)
        sol = local_search(inst, seed=42)
        assert sol.num_bins == 1

    def test_time_limit(self):
        inst = BinPackingInstance.random(n=20, seed=42)
        sol = local_search(inst, max_iterations=1000000, time_limit=0.5, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Invalid: {errors}"
