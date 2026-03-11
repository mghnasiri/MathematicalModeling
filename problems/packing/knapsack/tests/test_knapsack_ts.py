"""Tests for Tabu Search on 0-1 Knapsack Problem."""

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


_inst = _load_mod("kp_instance_test_ts", os.path.join(_parent_dir, "instance.py"))
KnapsackInstance = _inst.KnapsackInstance
KnapsackSolution = _inst.KnapsackSolution
validate_solution = _inst.validate_solution
small_knapsack_4 = _inst.small_knapsack_4
medium_knapsack_8 = _inst.medium_knapsack_8

_ts = _load_mod(
    "kp_ts_test",
    os.path.join(_parent_dir, "metaheuristics", "tabu_search.py"),
)
tabu_search = _ts.tabu_search


class TestKnapsackTSValidity:
    """Test that TS produces valid solutions."""

    def test_small4_valid(self):
        inst = small_knapsack_4()
        sol = tabu_search(inst, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Invalid solution: {errors}"

    def test_medium8_valid(self):
        inst = medium_knapsack_8()
        sol = tabu_search(inst, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Invalid solution: {errors}"

    def test_random_instance_valid(self):
        inst = KnapsackInstance.random(n=15, seed=42)
        sol = tabu_search(inst, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Invalid solution: {errors}"

    def test_feasibility(self):
        inst = small_knapsack_4()
        sol = tabu_search(inst, seed=42)
        assert sol.weight <= inst.capacity + 1e-10


class TestKnapsackTSQuality:
    """Test solution quality."""

    def test_small4_finds_optimal(self):
        inst = small_knapsack_4()
        sol = tabu_search(inst, max_iterations=1000, seed=42)
        assert sol.value >= 35.0 - 1e-6  # Known optimal = 35

    def test_medium8_good_quality(self):
        inst = medium_knapsack_8()
        sol = tabu_search(inst, max_iterations=1000, seed=42)
        assert sol.value >= 250.0  # Known optimal = 300

    def test_value_positive(self):
        inst = KnapsackInstance.random(n=10, seed=42)
        sol = tabu_search(inst, seed=42)
        assert sol.value > 0


class TestKnapsackTSDeterminism:
    """Test deterministic behavior with seed."""

    def test_same_seed_same_result(self):
        inst = small_knapsack_4()
        sol1 = tabu_search(inst, seed=42)
        sol2 = tabu_search(inst, seed=42)
        assert abs(sol1.value - sol2.value) < 1e-6

    def test_different_seed_both_valid(self):
        inst = medium_knapsack_8()
        sol1 = tabu_search(inst, seed=1)
        sol2 = tabu_search(inst, seed=999)
        valid1, _ = validate_solution(inst, sol1)
        valid2, _ = validate_solution(inst, sol2)
        assert valid1 and valid2


class TestKnapsackTSEdgeCases:
    """Test edge cases."""

    def test_single_item_fits(self):
        inst = KnapsackInstance(
            n=1,
            weights=np.array([5.0]),
            values=np.array([10.0]),
            capacity=10.0,
            name="single_fits",
        )
        sol = tabu_search(inst, seed=42)
        assert sol.value >= 10.0 - 1e-6

    def test_time_limit(self):
        inst = KnapsackInstance.random(n=20, seed=42)
        sol = tabu_search(
            inst, max_iterations=1000000, time_limit=0.5, seed=42
        )
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Invalid: {errors}"
