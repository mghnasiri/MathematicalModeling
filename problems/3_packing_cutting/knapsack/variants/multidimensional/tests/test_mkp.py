"""Tests for Multi-dimensional Knapsack Problem."""

from __future__ import annotations

import sys
import os
import importlib.util

import numpy as np
import pytest

_variant_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_mod(name, filepath):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_inst = _load_mod("mkp_instance_test", os.path.join(_variant_dir, "instance.py"))
MKPInstance = _inst.MKPInstance
MKPSolution = _inst.MKPSolution
validate_solution = _inst.validate_solution
small_mkp_5_2 = _inst.small_mkp_5_2

_heur = _load_mod("mkp_heuristics_test", os.path.join(_variant_dir, "heuristics.py"))
greedy_pseudo_utility = _heur.greedy_pseudo_utility
greedy_max_value = _heur.greedy_max_value

_meta = _load_mod("mkp_meta_test", os.path.join(_variant_dir, "metaheuristics.py"))
genetic_algorithm = _meta.genetic_algorithm


class TestMKPInstance:
    def test_random_creation(self):
        inst = MKPInstance.random(n=10, d=3, seed=42)
        assert inst.n == 10
        assert inst.d == 3
        assert inst.weights.shape == (3, 10)

    def test_small_benchmark(self):
        inst = small_mkp_5_2()
        assert inst.n == 5
        assert inst.d == 2
        assert inst.is_feasible([0, 2, 4])

    def test_feasibility_check(self):
        inst = small_mkp_5_2()
        # All items should be infeasible
        assert not inst.is_feasible(list(range(5)))


class TestMKPGreedy:
    def test_pseudo_utility_feasible(self):
        inst = MKPInstance.random(n=15, d=3, seed=42)
        sol = greedy_pseudo_utility(inst)
        assert inst.is_feasible(sol.items)

    def test_pseudo_utility_valid(self):
        inst = MKPInstance.random(n=15, d=3, seed=42)
        sol = greedy_pseudo_utility(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_max_value_feasible(self):
        inst = MKPInstance.random(n=15, d=3, seed=42)
        sol = greedy_max_value(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_positive_value(self):
        inst = MKPInstance.random(n=15, d=3, seed=42)
        sol = greedy_pseudo_utility(inst)
        assert sol.value > 0


class TestMKPGA:
    def test_ga_feasible(self):
        inst = MKPInstance.random(n=15, d=3, seed=42)
        sol = genetic_algorithm(inst, max_generations=50, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_ga_competitive(self):
        inst = MKPInstance.random(n=15, d=3, seed=42)
        gr_sol = greedy_pseudo_utility(inst)
        ga_sol = genetic_algorithm(inst, max_generations=100, seed=42)
        assert ga_sol.value >= gr_sol.value - 1e-6

    def test_ga_determinism(self):
        inst = MKPInstance.random(n=10, d=2, seed=42)
        sol1 = genetic_algorithm(inst, max_generations=50, seed=42)
        sol2 = genetic_algorithm(inst, max_generations=50, seed=42)
        assert abs(sol1.value - sol2.value) < 1e-6

    def test_ga_time_limit(self):
        inst = MKPInstance.random(n=20, d=3, seed=42)
        sol = genetic_algorithm(inst, max_generations=100000, time_limit=0.5, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_small_benchmark(self):
        inst = small_mkp_5_2()
        sol = genetic_algorithm(inst, max_generations=100, seed=42)
        assert sol.value >= 100  # Reasonable for small instance
