"""Tests for Bounded Knapsack Problem."""

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


_inst = _load_mod("bkp_inst_test", os.path.join(_variant_dir, "instance.py"))
BKPInstance = _inst.BKPInstance
BKPSolution = _inst.BKPSolution
validate_solution = _inst.validate_solution
small_bkp_5 = _inst.small_bkp_5

_heur = _load_mod("bkp_heur_test", os.path.join(_variant_dir, "heuristics.py"))
greedy_density = _heur.greedy_density
dynamic_programming = _heur.dynamic_programming

_meta = _load_mod("bkp_meta_test", os.path.join(_variant_dir, "metaheuristics.py"))
simulated_annealing = _meta.simulated_annealing


class TestBKPInstance:
    def test_random(self):
        inst = BKPInstance.random(n=8, seed=42)
        assert inst.n == 8

    def test_small(self):
        inst = small_bkp_5()
        assert inst.n == 5
        assert inst.capacity == 50


class TestBKPHeuristics:
    def test_greedy_valid(self):
        inst = BKPInstance.random(n=8, seed=42)
        sol = greedy_density(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_dp_valid(self):
        inst = small_bkp_5()
        sol = dynamic_programming(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_dp_optimal(self):
        inst = small_bkp_5()
        sol = dynamic_programming(inst)
        # DP should find at least as good as greedy
        gr = greedy_density(inst)
        assert sol.value >= gr.value - 1e-6


class TestBKPSA:
    def test_valid(self):
        inst = BKPInstance.random(n=8, seed=42)
        sol = simulated_annealing(inst, max_iterations=5000, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_competitive(self):
        inst = BKPInstance.random(n=8, seed=42)
        gr = greedy_density(inst)
        sa = simulated_annealing(inst, max_iterations=10000, seed=42)
        assert sa.value >= gr.value - 1e-6

    def test_determinism(self):
        inst = BKPInstance.random(n=6, seed=42)
        s1 = simulated_annealing(inst, max_iterations=3000, seed=42)
        s2 = simulated_annealing(inst, max_iterations=3000, seed=42)
        assert abs(s1.value - s2.value) < 1e-6

    def test_time_limit(self):
        inst = BKPInstance.random(n=10, seed=42)
        sol = simulated_annealing(inst, max_iterations=1000000, time_limit=0.5, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"
