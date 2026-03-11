"""Tests for Multiple Knapsack Problem."""

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


_inst = _load_mod("mkp_multi_inst_test", os.path.join(_variant_dir, "instance.py"))
MultipleKnapsackInstance = _inst.MultipleKnapsackInstance
MultipleKnapsackSolution = _inst.MultipleKnapsackSolution
validate_solution = _inst.validate_solution
small_mkp_6_2 = _inst.small_mkp_6_2

_heur = _load_mod("mkp_multi_heur_test", os.path.join(_variant_dir, "heuristics.py"))
greedy_value_density = _heur.greedy_value_density
greedy_best_fit = _heur.greedy_best_fit

_meta = _load_mod("mkp_multi_meta_test", os.path.join(_variant_dir, "metaheuristics.py"))
genetic_algorithm = _meta.genetic_algorithm


class TestMultiKPInstance:
    def test_random_creation(self):
        inst = MultipleKnapsackInstance.random(n=10, k=3, seed=42)
        assert inst.n == 10
        assert inst.k == 3

    def test_small_benchmark(self):
        inst = small_mkp_6_2()
        assert inst.n == 6
        assert inst.k == 2

    def test_feasibility(self):
        inst = small_mkp_6_2()
        assert inst.is_feasible([0, -1, -1, 1, 0, -1])


class TestMultiKPGreedy:
    def test_value_density_valid(self):
        inst = MultipleKnapsackInstance.random(n=15, k=3, seed=42)
        sol = greedy_value_density(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_best_fit_valid(self):
        inst = MultipleKnapsackInstance.random(n=15, k=3, seed=42)
        sol = greedy_best_fit(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_positive_value(self):
        inst = MultipleKnapsackInstance.random(n=15, k=3, seed=42)
        sol = greedy_value_density(inst)
        assert sol.value > 0

    def test_small_benchmark(self):
        inst = small_mkp_6_2()
        sol = greedy_value_density(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"


class TestMultiKPGA:
    def test_ga_valid(self):
        inst = MultipleKnapsackInstance.random(n=15, k=3, seed=42)
        sol = genetic_algorithm(inst, max_generations=50, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_ga_competitive(self):
        inst = MultipleKnapsackInstance.random(n=15, k=3, seed=42)
        gr_sol = greedy_value_density(inst)
        ga_sol = genetic_algorithm(inst, max_generations=100, seed=42)
        assert ga_sol.value >= gr_sol.value - 1e-6

    def test_ga_determinism(self):
        inst = MultipleKnapsackInstance.random(n=10, k=2, seed=42)
        sol1 = genetic_algorithm(inst, max_generations=50, seed=42)
        sol2 = genetic_algorithm(inst, max_generations=50, seed=42)
        assert abs(sol1.value - sol2.value) < 1e-6

    def test_ga_time_limit(self):
        inst = MultipleKnapsackInstance.random(n=20, k=3, seed=42)
        sol = genetic_algorithm(inst, max_generations=100000, time_limit=0.5, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"
