"""
Tests for Bin Packing Simulated Annealing.

Run: python -m pytest problems/packing/bin_packing/tests/test_bpp_sa.py -v
"""

from __future__ import annotations

import os
import sys
import pytest
import numpy as np
import importlib.util

_base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_module(name: str, filepath: str):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_inst_mod = _load_module("bpp_inst_sa_test", os.path.join(_base_dir, "instance.py"))
_sa_mod = _load_module("bpp_sa_test", os.path.join(_base_dir, "metaheuristics", "simulated_annealing.py"))
_ff_mod = _load_module("bpp_ff_sa_test", os.path.join(_base_dir, "heuristics", "first_fit.py"))

BinPackingInstance = _inst_mod.BinPackingInstance
BinPackingSolution = _inst_mod.BinPackingSolution
validate_solution = _inst_mod.validate_solution
easy_bpp_6 = _inst_mod.easy_bpp_6
tight_bpp_8 = _inst_mod.tight_bpp_8
uniform_bpp_10 = _inst_mod.uniform_bpp_10
simulated_annealing = _sa_mod.simulated_annealing
ffd = _ff_mod.first_fit_decreasing


class TestBinPackingSA:
    """Test Simulated Annealing for Bin Packing."""

    def test_returns_valid_solution(self):
        inst = BinPackingInstance.random(n=10, seed=42)
        sol = simulated_annealing(inst, max_iterations=1000, seed=42)
        is_valid, errors = validate_solution(inst, sol)
        assert is_valid, f"Invalid: {errors}"

    def test_all_items_packed(self):
        inst = BinPackingInstance.random(n=15, seed=42)
        sol = simulated_annealing(inst, max_iterations=1000, seed=42)
        all_items = sorted(item for b in sol.bins for item in b)
        assert all_items == list(range(15))

    def test_capacity_respected(self):
        inst = BinPackingInstance.random(n=15, seed=42)
        sol = simulated_annealing(inst, max_iterations=1000, seed=42)
        for b in sol.bins:
            total = sum(inst.sizes[i] for i in b)
            assert total <= inst.capacity + 1e-10

    def test_no_worse_than_ffd(self):
        inst = BinPackingInstance.random(n=20, seed=42)
        ffd_sol = ffd(inst)
        sa_sol = simulated_annealing(inst, max_iterations=5000, seed=42)
        assert sa_sol.num_bins <= ffd_sol.num_bins

    def test_easy6(self):
        inst = easy_bpp_6()
        sol = simulated_annealing(inst, max_iterations=3000, seed=42)
        is_valid, _ = validate_solution(inst, sol)
        assert is_valid
        assert sol.num_bins <= 3  # L1 = 3

    def test_tight8(self):
        inst = tight_bpp_8()
        sol = simulated_annealing(inst, max_iterations=3000, seed=42)
        is_valid, _ = validate_solution(inst, sol)
        assert is_valid
        assert sol.num_bins <= 5  # L1 = 5

    def test_uniform10(self):
        inst = uniform_bpp_10()
        sol = simulated_annealing(inst, max_iterations=3000, seed=42)
        is_valid, _ = validate_solution(inst, sol)
        assert is_valid
        assert sol.num_bins == 5  # Optimal = 5

    def test_deterministic_with_seed(self):
        inst = BinPackingInstance.random(n=10, seed=42)
        sol_a = simulated_annealing(inst, max_iterations=500, seed=123)
        sol_b = simulated_annealing(inst, max_iterations=500, seed=123)
        assert sol_a.num_bins == sol_b.num_bins

    def test_single_item(self):
        inst = BinPackingInstance(n=1, sizes=np.array([5.0]), capacity=10.0)
        sol = simulated_annealing(inst, max_iterations=100, seed=42)
        assert sol.num_bins == 1

    def test_time_limit(self):
        inst = BinPackingInstance.random(n=15, seed=42)
        sol = simulated_annealing(inst, time_limit=1.0, seed=42)
        is_valid, _ = validate_solution(inst, sol)
        assert is_valid
