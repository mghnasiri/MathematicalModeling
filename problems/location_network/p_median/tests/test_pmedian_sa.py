"""
Tests for p-Median Simulated Annealing.

Run: python -m pytest problems/location_network/p_median/tests/test_pmedian_sa.py -v
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


_inst_mod = _load_module("pmed_inst_sa_test", os.path.join(_base_dir, "instance.py"))
_sa_mod = _load_module("pmed_sa_test", os.path.join(_base_dir, "metaheuristics", "simulated_annealing.py"))
_greedy_mod = _load_module("pmed_greedy_sa_test", os.path.join(_base_dir, "heuristics", "greedy_pmedian.py"))

PMedianInstance = _inst_mod.PMedianInstance
PMedianSolution = _inst_mod.PMedianSolution
validate_solution = _inst_mod.validate_solution
small_pmedian_6_2 = _inst_mod.small_pmedian_6_2
simulated_annealing = _sa_mod.simulated_annealing
greedy_pmedian = _greedy_mod.greedy_pmedian


class TestPMedianSA:
    """Test Simulated Annealing for p-Median."""

    def test_returns_valid_solution(self):
        inst = PMedianInstance.random(n=10, p=3, seed=42)
        sol = simulated_annealing(inst, max_iterations=500, seed=42)
        is_valid, errors = validate_solution(inst, sol)
        assert is_valid, f"Invalid: {errors}"

    def test_correct_p_facilities(self):
        inst = PMedianInstance.random(n=10, p=3, seed=42)
        sol = simulated_annealing(inst, max_iterations=500, seed=42)
        assert len(sol.open_facilities) == 3

    def test_cost_correct(self):
        inst = PMedianInstance.random(n=10, p=3, seed=42)
        sol = simulated_annealing(inst, max_iterations=500, seed=42)
        actual = inst.total_cost(sol.open_facilities, sol.assignments)
        assert abs(sol.cost - actual) < 1e-4

    def test_no_worse_than_greedy(self):
        inst = PMedianInstance.random(n=15, p=3, seed=42)
        gr_sol = greedy_pmedian(inst)
        sa_sol = simulated_annealing(inst, max_iterations=2000, seed=42)
        assert sa_sol.cost <= gr_sol.cost + 1e-4

    def test_small_benchmark(self):
        inst = small_pmedian_6_2()
        sol = simulated_annealing(inst, max_iterations=1000, seed=42)
        is_valid, errors = validate_solution(inst, sol)
        assert is_valid, f"Invalid: {errors}"
        assert len(sol.open_facilities) == 2

    def test_deterministic_with_seed(self):
        inst = PMedianInstance.random(n=10, p=3, seed=42)
        sol_a = simulated_annealing(inst, max_iterations=300, seed=123)
        sol_b = simulated_annealing(inst, max_iterations=300, seed=123)
        assert abs(sol_a.cost - sol_b.cost) < 1e-6

    def test_p_equals_m(self):
        """All facilities open — only one possible solution."""
        inst = PMedianInstance.random(n=5, m=5, p=5, seed=42)
        sol = simulated_annealing(inst, max_iterations=100, seed=42)
        assert len(sol.open_facilities) == 5
        assert sorted(sol.open_facilities) == list(range(5))

    def test_p_equals_1(self):
        inst = PMedianInstance.random(n=10, p=1, seed=42)
        sol = simulated_annealing(inst, max_iterations=500, seed=42)
        assert len(sol.open_facilities) == 1
        is_valid, _ = validate_solution(inst, sol)
        assert is_valid

    def test_time_limit(self):
        inst = PMedianInstance.random(n=15, p=3, seed=42)
        sol = simulated_annealing(inst, time_limit=1.0, seed=42)
        is_valid, _ = validate_solution(inst, sol)
        assert is_valid
