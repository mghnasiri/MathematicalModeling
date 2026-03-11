"""Tests for Job Shop with Weighted Tardiness."""

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


_inst = _load_mod("wtjsp_inst_test", os.path.join(_variant_dir, "instance.py"))
WTJSPInstance = _inst.WTJSPInstance
WTJSPSolution = _inst.WTJSPSolution
validate_solution = _inst.validate_solution
small_wtjsp_3_3 = _inst.small_wtjsp_3_3

_heur = _load_mod("wtjsp_heur_test", os.path.join(_variant_dir, "heuristics.py"))
atc_dispatch = _heur.atc_dispatch
wspt_dispatch = _heur.wspt_dispatch

_meta = _load_mod("wtjsp_meta_test", os.path.join(_variant_dir, "metaheuristics.py"))
simulated_annealing = _meta.simulated_annealing


class TestWTJSPInstance:
    def test_random_creation(self):
        inst = WTJSPInstance.random(n=5, m=3, seed=42)
        assert inst.n == 5
        assert inst.m == 3
        assert inst.due_dates.shape == (5,)
        assert inst.weights.shape == (5,)

    def test_small_benchmark(self):
        inst = small_wtjsp_3_3()
        assert inst.n == 3
        assert inst.m == 3


class TestWTJSPHeuristics:
    def test_atc_valid(self):
        inst = WTJSPInstance.random(n=6, m=3, seed=42)
        sol = atc_dispatch(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_wspt_valid(self):
        inst = WTJSPInstance.random(n=6, m=3, seed=42)
        sol = wspt_dispatch(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_nonnegative_wt(self):
        inst = WTJSPInstance.random(n=6, m=3, seed=42)
        sol = atc_dispatch(inst)
        assert sol.weighted_tardiness >= -1e-10

    def test_small_benchmark(self):
        inst = small_wtjsp_3_3()
        sol = atc_dispatch(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"


class TestWTJSPSA:
    def test_valid(self):
        inst = WTJSPInstance.random(n=5, m=3, seed=42)
        sol = simulated_annealing(inst, max_iterations=5000, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_determinism(self):
        inst = WTJSPInstance.random(n=4, m=3, seed=42)
        sol1 = simulated_annealing(inst, max_iterations=3000, seed=42)
        sol2 = simulated_annealing(inst, max_iterations=3000, seed=42)
        assert abs(sol1.weighted_tardiness - sol2.weighted_tardiness) < 1e-6

    def test_nonnegative(self):
        inst = WTJSPInstance.random(n=5, m=3, seed=42)
        sol = simulated_annealing(inst, max_iterations=5000, seed=42)
        assert sol.weighted_tardiness >= -1e-10

    def test_time_limit(self):
        inst = WTJSPInstance.random(n=6, m=3, seed=42)
        sol = simulated_annealing(inst, max_iterations=1000000, time_limit=0.5, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"
