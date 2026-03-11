"""Tests for Variable-Size Bin Packing."""

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


_inst = _load_mod("vsbpp_inst_test", os.path.join(_variant_dir, "instance.py"))
VSBPPInstance = _inst.VSBPPInstance
VSBPPSolution = _inst.VSBPPSolution
validate_solution = _inst.validate_solution
small_vsbpp_8 = _inst.small_vsbpp_8

_heur = _load_mod("vsbpp_heur_test", os.path.join(_variant_dir, "heuristics.py"))
ffd_best_type = _heur.ffd_best_type
cost_ratio_greedy = _heur.cost_ratio_greedy

_meta = _load_mod("vsbpp_meta_test", os.path.join(_variant_dir, "metaheuristics.py"))
simulated_annealing = _meta.simulated_annealing


class TestVSBPPInstance:
    def test_random_creation(self):
        inst = VSBPPInstance.random(n=10, seed=42)
        assert inst.n == 10
        assert inst.num_bin_types == 3

    def test_small_benchmark(self):
        inst = small_vsbpp_8()
        assert inst.n == 8
        assert inst.num_bin_types == 3


class TestVSBPPHeuristics:
    def test_ffd_valid(self):
        inst = VSBPPInstance.random(n=12, seed=42)
        sol = ffd_best_type(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_cost_ratio_valid(self):
        inst = VSBPPInstance.random(n=12, seed=42)
        sol = cost_ratio_greedy(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_positive_cost(self):
        inst = VSBPPInstance.random(n=12, seed=42)
        sol = ffd_best_type(inst)
        assert sol.total_cost > 0

    def test_small_benchmark(self):
        inst = small_vsbpp_8()
        sol = ffd_best_type(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"


class TestVSBPPSA:
    def test_valid(self):
        inst = VSBPPInstance.random(n=12, seed=42)
        sol = simulated_annealing(inst, max_iterations=5000, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_competitive(self):
        inst = VSBPPInstance.random(n=12, seed=42)
        ffd_sol = ffd_best_type(inst)
        sa_sol = simulated_annealing(inst, max_iterations=10000, seed=42)
        assert sa_sol.total_cost <= ffd_sol.total_cost + 1e-6

    def test_determinism(self):
        inst = VSBPPInstance.random(n=8, seed=42)
        sol1 = simulated_annealing(inst, max_iterations=3000, seed=42)
        sol2 = simulated_annealing(inst, max_iterations=3000, seed=42)
        assert abs(sol1.total_cost - sol2.total_cost) < 1e-6

    def test_time_limit(self):
        inst = VSBPPInstance.random(n=15, seed=42)
        sol = simulated_annealing(inst, max_iterations=1000000, time_limit=0.5, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"
