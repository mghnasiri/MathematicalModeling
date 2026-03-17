"""Tests for Capacitated p-Median Problem."""

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


_inst = _load_mod("cpmp_inst_test", os.path.join(_variant_dir, "instance.py"))
CPMedianInstance = _inst.CPMedianInstance
CPMedianSolution = _inst.CPMedianSolution
validate_solution = _inst.validate_solution
small_cpmp_6 = _inst.small_cpmp_6

_heur = _load_mod("cpmp_heur_test", os.path.join(_variant_dir, "heuristics.py"))
greedy_add = _heur.greedy_add
teitz_bart = _heur.teitz_bart

_meta = _load_mod("cpmp_meta_test", os.path.join(_variant_dir, "metaheuristics.py"))
simulated_annealing = _meta.simulated_annealing


class TestCPMPInstance:
    def test_random(self):
        inst = CPMedianInstance.random(n=10, m=5, p=2, seed=42)
        assert inst.n == 10
        assert inst.m == 5
        assert inst.p == 2

    def test_small(self):
        inst = small_cpmp_6()
        assert inst.n == 6
        assert inst.p == 2


class TestCPMPHeuristics:
    def test_greedy_valid(self):
        inst = small_cpmp_6()
        sol = greedy_add(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_teitz_bart_valid(self):
        inst = small_cpmp_6()
        sol = teitz_bart(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_greedy_random(self):
        inst = CPMedianInstance.random(n=10, m=5, p=2, seed=42)
        sol = greedy_add(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_teitz_bart_improves(self):
        inst = CPMedianInstance.random(n=10, m=5, p=2, seed=42)
        greedy = greedy_add(inst)
        tb = teitz_bart(inst)
        assert tb.total_cost <= greedy.total_cost + 1e-6


class TestCPMPSA:
    def test_valid(self):
        inst = small_cpmp_6()
        sol = simulated_annealing(inst, max_iterations=5000, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_determinism(self):
        inst = small_cpmp_6()
        s1 = simulated_annealing(inst, max_iterations=3000, seed=42)
        s2 = simulated_annealing(inst, max_iterations=3000, seed=42)
        assert abs(s1.total_cost - s2.total_cost) < 1e-6

    def test_time_limit(self):
        inst = CPMedianInstance.random(n=12, m=6, p=3, seed=42)
        sol = simulated_annealing(inst, max_iterations=1000000,
                                   time_limit=0.5, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"
