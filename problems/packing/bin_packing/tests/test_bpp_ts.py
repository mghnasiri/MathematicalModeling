"""Tests for Tabu Search on 1D Bin Packing Problem."""

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


_inst = _load_mod("bpp_instance_test_ts", os.path.join(_parent_dir, "instance.py"))
BinPackingInstance = _inst.BinPackingInstance
BinPackingSolution = _inst.BinPackingSolution
validate_solution = _inst.validate_solution
easy_bpp_6 = _inst.easy_bpp_6
tight_bpp_8 = _inst.tight_bpp_8

_ts = _load_mod(
    "bpp_ts_test",
    os.path.join(_parent_dir, "metaheuristics", "tabu_search.py"),
)
tabu_search = _ts.tabu_search


class TestBPPTSValidity:
    """Test that TS produces valid solutions."""

    def test_easy6_valid(self):
        inst = easy_bpp_6()
        sol = tabu_search(inst, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Invalid solution: {errors}"

    def test_tight8_valid(self):
        inst = tight_bpp_8()
        sol = tabu_search(inst, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Invalid solution: {errors}"

    def test_random_instance_valid(self):
        inst = BinPackingInstance.random(n=15, seed=42)
        sol = tabu_search(inst, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Invalid solution: {errors}"


class TestBPPTSQuality:
    """Test solution quality."""

    def test_easy6_reasonable(self):
        inst = easy_bpp_6()
        sol = tabu_search(inst, max_iterations=2000, seed=42)
        lb = inst.lower_bound_l1()
        assert sol.num_bins >= lb
        assert sol.num_bins <= lb + 2

    def test_tight8_reasonable(self):
        inst = tight_bpp_8()
        sol = tabu_search(inst, max_iterations=2000, seed=42)
        lb = inst.lower_bound_l1()
        assert sol.num_bins >= lb
        assert sol.num_bins <= lb + 2


class TestBPPTSDeterminism:
    """Test deterministic behavior with seed."""

    def test_same_seed_same_result(self):
        inst = easy_bpp_6()
        sol1 = tabu_search(inst, seed=42)
        sol2 = tabu_search(inst, seed=42)
        assert sol1.num_bins == sol2.num_bins

    def test_different_seed_both_valid(self):
        inst = tight_bpp_8()
        sol1 = tabu_search(inst, seed=1)
        sol2 = tabu_search(inst, seed=999)
        valid1, _ = validate_solution(inst, sol1)
        valid2, _ = validate_solution(inst, sol2)
        assert valid1 and valid2


class TestBPPTSEdgeCases:
    """Test edge cases."""

    def test_single_item(self):
        inst = BinPackingInstance(
            n=1,
            sizes=np.array([5.0]),
            capacity=10.0,
            name="single",
        )
        sol = tabu_search(inst, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Invalid: {errors}"
        assert sol.num_bins == 1

    def test_time_limit(self):
        inst = BinPackingInstance.random(n=15, seed=42)
        sol = tabu_search(
            inst, max_iterations=1000000, time_limit=0.5, seed=42
        )
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Invalid: {errors}"
