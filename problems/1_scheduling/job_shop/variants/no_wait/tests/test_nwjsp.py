"""Tests for No-Wait Job Shop Scheduling."""

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


_inst = _load_mod("nwjsp_inst_test", os.path.join(_variant_dir, "instance.py"))
NWJSPInstance = _inst.NWJSPInstance
NWJSPSolution = _inst.NWJSPSolution
validate_solution = _inst.validate_solution
small_nwjsp_3_3 = _inst.small_nwjsp_3_3

_heur = _load_mod("nwjsp_heur_test", os.path.join(_variant_dir, "heuristics.py"))
greedy_insertion = _heur.greedy_insertion
spt_schedule = _heur.spt_schedule

_meta = _load_mod("nwjsp_meta_test", os.path.join(_variant_dir, "metaheuristics.py"))
simulated_annealing = _meta.simulated_annealing


class TestNWJSPInstance:
    def test_random_creation(self):
        inst = NWJSPInstance.random(n=5, m=3, seed=42)
        assert inst.n == 5
        assert inst.m == 3

    def test_small_benchmark(self):
        inst = small_nwjsp_3_3()
        assert inst.n == 3
        assert inst.m == 3

    def test_job_duration(self):
        inst = small_nwjsp_3_3()
        assert inst.job_duration(0) == 9  # 3+2+4


class TestNWJSPHeuristics:
    def test_greedy_valid(self):
        inst = NWJSPInstance.random(n=5, m=3, seed=42)
        sol = greedy_insertion(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_spt_valid(self):
        inst = NWJSPInstance.random(n=5, m=3, seed=42)
        sol = spt_schedule(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_positive_makespan(self):
        inst = NWJSPInstance.random(n=5, m=3, seed=42)
        sol = greedy_insertion(inst)
        assert sol.makespan > 0

    def test_small_benchmark(self):
        inst = small_nwjsp_3_3()
        sol = greedy_insertion(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"


class TestNWJSPSA:
    def test_valid(self):
        inst = NWJSPInstance.random(n=5, m=3, seed=42)
        sol = simulated_annealing(inst, max_iterations=5000, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_competitive(self):
        inst = NWJSPInstance.random(n=5, m=3, seed=42)
        gr_sol = greedy_insertion(inst)
        sa_sol = simulated_annealing(inst, max_iterations=10000, seed=42)
        assert sa_sol.makespan <= gr_sol.makespan + 1e-6

    def test_determinism(self):
        inst = NWJSPInstance.random(n=4, m=3, seed=42)
        sol1 = simulated_annealing(inst, max_iterations=3000, seed=42)
        sol2 = simulated_annealing(inst, max_iterations=3000, seed=42)
        assert abs(sol1.makespan - sol2.makespan) < 1e-6

    def test_time_limit(self):
        inst = NWJSPInstance.random(n=6, m=3, seed=42)
        sol = simulated_annealing(inst, max_iterations=1000000, time_limit=0.5, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"
