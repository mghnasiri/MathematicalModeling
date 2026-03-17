"""Tests for Simulated Annealing on 1D Cutting Stock Problem."""

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


_inst = _load_mod("csp_instance_test_sa", os.path.join(_parent_dir, "instance.py"))
CuttingStockInstance = _inst.CuttingStockInstance
CuttingStockSolution = _inst.CuttingStockSolution
validate_solution = _inst.validate_solution
simple_csp_3 = _inst.simple_csp_3
classic_csp_4 = _inst.classic_csp_4

_sa = _load_mod(
    "csp_sa_test",
    os.path.join(_parent_dir, "metaheuristics", "simulated_annealing.py"),
)
simulated_annealing = _sa.simulated_annealing

_greedy = _load_mod(
    "csp_greedy_test_sa",
    os.path.join(_parent_dir, "heuristics", "greedy_csp.py"),
)
ffd_based = _greedy.ffd_based


class TestCSPSAValidity:
    """Test that SA produces valid solutions."""

    def test_simple3_valid(self):
        inst = simple_csp_3()
        sol = simulated_annealing(inst, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Invalid solution: {errors}"

    def test_classic4_valid(self):
        inst = classic_csp_4()
        sol = simulated_annealing(inst, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Invalid solution: {errors}"

    def test_random_instance_valid(self):
        inst = CuttingStockInstance.random(m=5, stock_length=100, seed=123)
        sol = simulated_annealing(inst, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Invalid solution: {errors}"


class TestCSPSAQuality:
    """Test solution quality."""

    def test_simple3_meets_lower_bound(self):
        inst = simple_csp_3()
        sol = simulated_annealing(inst, max_iterations=5000, seed=42)
        lb = inst.lower_bound()
        assert sol.num_rolls >= lb

    def test_classic4_reasonable(self):
        inst = classic_csp_4()
        sol = simulated_annealing(inst, max_iterations=5000, seed=42)
        # FFD provides an upper bound
        ffd_sol = ffd_based(inst)
        assert sol.num_rolls <= ffd_sol.num_rolls + 1

    def test_sa_not_worse_than_2x_lb(self):
        inst = CuttingStockInstance.random(m=6, stock_length=100, seed=99)
        sol = simulated_annealing(inst, max_iterations=5000, seed=42)
        lb = inst.lower_bound()
        assert sol.num_rolls <= 2 * lb + 1


class TestCSPSADeterminism:
    """Test deterministic behavior with seed."""

    def test_same_seed_same_result(self):
        inst = simple_csp_3()
        sol1 = simulated_annealing(inst, seed=42)
        sol2 = simulated_annealing(inst, seed=42)
        assert sol1.num_rolls == sol2.num_rolls

    def test_different_seed_may_differ(self):
        inst = CuttingStockInstance.random(m=5, stock_length=100, seed=10)
        sol1 = simulated_annealing(inst, max_iterations=2000, seed=1)
        sol2 = simulated_annealing(inst, max_iterations=2000, seed=999)
        # Both valid
        valid1, _ = validate_solution(inst, sol1)
        valid2, _ = validate_solution(inst, sol2)
        assert valid1 and valid2


class TestCSPSAEdgeCases:
    """Test edge cases."""

    def test_single_item_type(self):
        inst = CuttingStockInstance(
            m=1, stock_length=10.0,
            lengths=np.array([3.0]),
            demands=np.array([5]),
            name="single_type",
        )
        sol = simulated_annealing(inst, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Invalid: {errors}"
        assert sol.num_rolls >= 2  # 5 items of size 3 in rolls of 10

    def test_zero_demands(self):
        inst = CuttingStockInstance(
            m=2, stock_length=100.0,
            lengths=np.array([30.0, 40.0]),
            demands=np.array([0, 0]),
            name="zero_demand",
        )
        sol = simulated_annealing(inst, seed=42)
        assert sol.num_rolls == 0

    def test_time_limit(self):
        inst = CuttingStockInstance.random(m=5, stock_length=100, seed=42)
        sol = simulated_annealing(inst, max_iterations=1000000, time_limit=0.5, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Invalid: {errors}"
