"""Tests for Parallel Machine with Sequence-Dependent Setup Times."""

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


_inst = _load_mod("pmsdst_inst_test", os.path.join(_variant_dir, "instance.py"))
PMSDSTInstance = _inst.PMSDSTInstance
PMSDSTSolution = _inst.PMSDSTSolution
validate_solution = _inst.validate_solution
small_pmsdst_4_2 = _inst.small_pmsdst_4_2

_heur = _load_mod("pmsdst_heur_test", os.path.join(_variant_dir, "heuristics.py"))
greedy_ect_sdst = _heur.greedy_ect_sdst
lpt_sdst = _heur.lpt_sdst

_meta = _load_mod("pmsdst_meta_test", os.path.join(_variant_dir, "metaheuristics.py"))
simulated_annealing = _meta.simulated_annealing


class TestPMSDSTInstance:
    def test_random_creation(self):
        inst = PMSDSTInstance.random(n=8, m=3, seed=42)
        assert inst.n == 8
        assert inst.m == 3
        assert inst.setup_times.shape == (8, 8, 3)

    def test_small_benchmark(self):
        inst = small_pmsdst_4_2()
        assert inst.n == 4
        assert inst.m == 2

    def test_makespan_computation(self):
        inst = small_pmsdst_4_2()
        ms = inst.makespan([[0, 1], [2, 3]])
        assert ms > 0


class TestPMSDSTHeuristics:
    def test_ect_valid(self):
        inst = PMSDSTInstance.random(n=10, m=3, seed=42)
        sol = greedy_ect_sdst(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_lpt_valid(self):
        inst = PMSDSTInstance.random(n=10, m=3, seed=42)
        sol = lpt_sdst(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_positive_makespan(self):
        inst = PMSDSTInstance.random(n=10, m=3, seed=42)
        sol = greedy_ect_sdst(inst)
        assert sol.makespan > 0

    def test_small_benchmark(self):
        inst = small_pmsdst_4_2()
        sol = greedy_ect_sdst(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"


class TestPMSDSTSA:
    def test_valid(self):
        inst = PMSDSTInstance.random(n=10, m=3, seed=42)
        sol = simulated_annealing(inst, max_iterations=5000, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_competitive(self):
        inst = PMSDSTInstance.random(n=10, m=3, seed=42)
        ect_sol = greedy_ect_sdst(inst)
        sa_sol = simulated_annealing(inst, max_iterations=10000, seed=42)
        assert sa_sol.makespan <= ect_sol.makespan + 1e-6

    def test_determinism(self):
        inst = PMSDSTInstance.random(n=8, m=2, seed=42)
        sol1 = simulated_annealing(inst, max_iterations=3000, seed=42)
        sol2 = simulated_annealing(inst, max_iterations=3000, seed=42)
        assert abs(sol1.makespan - sol2.makespan) < 1e-6

    def test_time_limit(self):
        inst = PMSDSTInstance.random(n=15, m=3, seed=42)
        sol = simulated_annealing(inst, max_iterations=1000000, time_limit=0.5, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"
