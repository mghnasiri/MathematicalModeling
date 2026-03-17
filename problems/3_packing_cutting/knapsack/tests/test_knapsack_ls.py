"""Tests for Local Search on 0-1 Knapsack."""

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


_inst = _load_mod("kp_instance_test_ls", os.path.join(_parent_dir, "instance.py"))
KnapsackInstance = _inst.KnapsackInstance
KnapsackSolution = _inst.KnapsackSolution
validate_solution = _inst.validate_solution
small_knapsack_4 = _inst.small_knapsack_4
medium_knapsack_8 = _inst.medium_knapsack_8

_ls = _load_mod(
    "kp_ls_test",
    os.path.join(_parent_dir, "metaheuristics", "local_search.py"),
)
local_search = _ls.local_search

_greedy = _load_mod(
    "kp_greedy_test_ls",
    os.path.join(_parent_dir, "heuristics", "greedy.py"),
)
greedy_value_density = _greedy.greedy_value_density


class TestKnapsackLSValidity:
    """Test that LS produces valid solutions."""

    def test_small4_valid(self):
        inst = small_knapsack_4()
        sol = local_search(inst, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Invalid solution: {errors}"

    def test_medium8_valid(self):
        inst = medium_knapsack_8()
        sol = local_search(inst, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Invalid solution: {errors}"

    def test_random_valid(self):
        inst = KnapsackInstance.random(n=15, seed=42)
        sol = local_search(inst, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Invalid solution: {errors}"


class TestKnapsackLSQuality:
    """Test solution quality."""

    def test_small4_optimal(self):
        inst = small_knapsack_4()
        sol = local_search(inst, seed=42)
        assert sol.value >= 35.0 - 1e-6

    def test_ls_competitive_with_greedy(self):
        inst = KnapsackInstance.random(n=20, seed=42)
        greedy_sol = greedy_value_density(inst)
        ls_sol = local_search(inst, seed=42)
        assert ls_sol.value >= greedy_sol.value - 1e-6

    def test_swap_neighborhood(self):
        inst = medium_knapsack_8()
        sol = local_search(inst, neighborhood="swap", seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Invalid: {errors}"


class TestKnapsackLSDeterminism:
    """Test deterministic behavior with seed."""

    def test_same_seed_same_result(self):
        inst = medium_knapsack_8()
        sol1 = local_search(inst, max_iterations=200, seed=42)
        sol2 = local_search(inst, max_iterations=200, seed=42)
        assert abs(sol1.value - sol2.value) < 1e-6

    def test_different_seed_both_valid(self):
        inst = KnapsackInstance.random(n=10, seed=42)
        sol1 = local_search(inst, seed=1)
        sol2 = local_search(inst, seed=999)
        valid1, _ = validate_solution(inst, sol1)
        valid2, _ = validate_solution(inst, sol2)
        assert valid1 and valid2


class TestKnapsackLSEdgeCases:
    """Test edge cases."""

    def test_single_item(self):
        inst = KnapsackInstance(
            n=1, weights=np.array([5.0]),
            values=np.array([10.0]), capacity=10.0,
        )
        sol = local_search(inst, seed=42)
        assert sol.items == [0]

    def test_time_limit(self):
        inst = KnapsackInstance.random(n=20, seed=42)
        sol = local_search(inst, max_iterations=1000000, time_limit=0.5, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Invalid: {errors}"
