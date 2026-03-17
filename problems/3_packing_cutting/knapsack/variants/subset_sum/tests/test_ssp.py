"""Tests for Subset Sum Problem."""

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


_inst = _load_mod("ssp_inst_test", os.path.join(_variant_dir, "instance.py"))
SubsetSumInstance = _inst.SubsetSumInstance
SubsetSumSolution = _inst.SubsetSumSolution
validate_solution = _inst.validate_solution
small_ssp_6 = _inst.small_ssp_6

_heur = _load_mod("ssp_heur_test", os.path.join(_variant_dir, "heuristics.py"))
greedy_largest = _heur.greedy_largest
dynamic_programming = _heur.dynamic_programming

_meta = _load_mod("ssp_meta_test", os.path.join(_variant_dir, "metaheuristics.py"))
simulated_annealing = _meta.simulated_annealing


class TestSSPInstance:
    def test_random(self):
        inst = SubsetSumInstance.random(n=8, seed=42)
        assert inst.n == 8

    def test_small(self):
        inst = small_ssp_6()
        assert inst.n == 6
        assert inst.target == 14


class TestSSPHeuristics:
    def test_greedy_valid(self):
        inst = SubsetSumInstance.random(n=8, seed=42)
        sol = greedy_largest(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_dp_valid(self):
        inst = small_ssp_6()
        sol = dynamic_programming(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_dp_exact(self):
        inst = small_ssp_6()
        sol = dynamic_programming(inst)
        # Target 14 is achievable: {7, 5, 2} or {8, 5, 1} or {3, 8, 2, 1} etc.
        assert sol.total == 14

    def test_dp_beats_greedy(self):
        inst = SubsetSumInstance.random(n=10, seed=42)
        gr = greedy_largest(inst)
        dp = dynamic_programming(inst)
        assert dp.total >= gr.total


class TestSSPSA:
    def test_valid(self):
        inst = SubsetSumInstance.random(n=8, seed=42)
        sol = simulated_annealing(inst, max_iterations=5000, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_competitive(self):
        inst = SubsetSumInstance.random(n=10, seed=42)
        gr = greedy_largest(inst)
        sa = simulated_annealing(inst, max_iterations=10000, seed=42)
        assert sa.total >= gr.total

    def test_determinism(self):
        inst = SubsetSumInstance.random(n=6, seed=42)
        s1 = simulated_annealing(inst, max_iterations=3000, seed=42)
        s2 = simulated_annealing(inst, max_iterations=3000, seed=42)
        assert s1.total == s2.total

    def test_time_limit(self):
        inst = SubsetSumInstance.random(n=15, seed=42)
        sol = simulated_annealing(inst, max_iterations=1000000, time_limit=0.5, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"
