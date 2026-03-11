"""
Tests for Knapsack Simulated Annealing.

Run: python -m pytest problems/packing/knapsack/tests/test_knapsack_sa.py -v
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


_inst_mod = _load_module("kp_inst_sa_test", os.path.join(_base_dir, "instance.py"))
_sa_mod = _load_module("kp_sa_test", os.path.join(_base_dir, "metaheuristics", "simulated_annealing.py"))
_gr_mod = _load_module("kp_gr_sa_test", os.path.join(_base_dir, "heuristics", "greedy.py"))

KnapsackInstance = _inst_mod.KnapsackInstance
KnapsackSolution = _inst_mod.KnapsackSolution
validate_solution = _inst_mod.validate_solution
small_knapsack_4 = _inst_mod.small_knapsack_4
medium_knapsack_8 = _inst_mod.medium_knapsack_8
strongly_correlated_10 = _inst_mod.strongly_correlated_10
simulated_annealing = _sa_mod.simulated_annealing
greedy_value_density = _gr_mod.greedy_value_density


class TestKnapsackSA:
    """Test Simulated Annealing for 0-1 Knapsack."""

    def test_returns_valid_solution(self):
        inst = KnapsackInstance.random(n=10, seed=42)
        sol = simulated_annealing(inst, max_iterations=1000, seed=42)
        is_valid, errors = validate_solution(inst, sol)
        assert is_valid, f"Invalid: {errors}"

    def test_feasibility(self):
        inst = KnapsackInstance.random(n=15, seed=42)
        sol = simulated_annealing(inst, max_iterations=1000, seed=42)
        assert sol.weight <= inst.capacity + 1e-10

    def test_small4_optimal(self):
        """SA should find optimal on 4-item instance (optimal=35)."""
        inst = small_knapsack_4()
        sol = simulated_annealing(inst, max_iterations=5000, seed=42)
        assert sol.value >= 35.0 - 1e-6

    def test_medium8_quality(self):
        """SA should find good solution for medium8 (optimal=300)."""
        inst = medium_knapsack_8()
        sol = simulated_annealing(inst, max_iterations=10000, seed=42)
        assert sol.value >= 280  # Within 7% of optimal

    def test_no_worse_than_greedy(self):
        inst = KnapsackInstance.random(n=20, seed=42)
        gr_sol = greedy_value_density(inst)
        sa_sol = simulated_annealing(inst, max_iterations=5000, seed=42)
        assert sa_sol.value >= gr_sol.value - 1e-6

    def test_deterministic_with_seed(self):
        inst = KnapsackInstance.random(n=10, seed=42)
        sol_a = simulated_annealing(inst, max_iterations=500, seed=123)
        sol_b = simulated_annealing(inst, max_iterations=500, seed=123)
        assert abs(sol_a.value - sol_b.value) < 1e-6
        assert sol_a.items == sol_b.items

    def test_single_item_fits(self):
        inst = KnapsackInstance(
            n=1, weights=np.array([5.0]), values=np.array([10.0]),
            capacity=10.0,
        )
        sol = simulated_annealing(inst, max_iterations=100, seed=42)
        assert 0 in sol.items
        assert sol.value == 10.0

    def test_single_item_too_heavy(self):
        inst = KnapsackInstance(
            n=1, weights=np.array([15.0]), values=np.array([10.0]),
            capacity=10.0,
        )
        sol = simulated_annealing(inst, max_iterations=100, seed=42)
        assert sol.items == []
        assert sol.value == 0.0

    def test_time_limit(self):
        inst = KnapsackInstance.random(n=20, seed=42)
        sol = simulated_annealing(inst, time_limit=1.0, seed=42)
        is_valid, _ = validate_solution(inst, sol)
        assert is_valid

    def test_strongly_correlated(self):
        inst = strongly_correlated_10()
        sol = simulated_annealing(inst, max_iterations=10000, seed=42)
        is_valid, _ = validate_solution(inst, sol)
        assert is_valid
        assert sol.value > 0

    def test_no_duplicate_items(self):
        inst = KnapsackInstance.random(n=15, seed=42)
        sol = simulated_annealing(inst, max_iterations=1000, seed=42)
        assert len(sol.items) == len(set(sol.items))
