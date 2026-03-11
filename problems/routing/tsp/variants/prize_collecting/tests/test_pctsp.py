"""Tests for Prize-Collecting TSP."""

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


_inst = _load_mod("pctsp_inst_test", os.path.join(_variant_dir, "instance.py"))
PCTSPInstance = _inst.PCTSPInstance
PCTSPSolution = _inst.PCTSPSolution
validate_solution = _inst.validate_solution
small_pctsp_6 = _inst.small_pctsp_6

_heur = _load_mod("pctsp_heur_test", os.path.join(_variant_dir, "heuristics.py"))
greedy_prize = _heur.greedy_prize
nearest_neighbor_pctsp = _heur.nearest_neighbor_pctsp

_meta = _load_mod("pctsp_meta_test", os.path.join(_variant_dir, "metaheuristics.py"))
simulated_annealing = _meta.simulated_annealing


class TestPCTSPInstance:
    def test_random(self):
        inst = PCTSPInstance.random(n=8, seed=42)
        assert inst.n == 8

    def test_small(self):
        inst = small_pctsp_6()
        assert inst.n == 6
        assert inst.min_prize == 30.0


class TestPCTSPHeuristics:
    def test_greedy_valid(self):
        inst = small_pctsp_6()
        sol = greedy_prize(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_nn_valid(self):
        inst = small_pctsp_6()
        sol = nearest_neighbor_pctsp(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_greedy_meets_min(self):
        inst = small_pctsp_6()
        sol = greedy_prize(inst)
        assert sol.total_prize >= inst.min_prize - 1e-4

    def test_nn_meets_min(self):
        inst = small_pctsp_6()
        sol = nearest_neighbor_pctsp(inst)
        assert sol.total_prize >= inst.min_prize - 1e-4


class TestPCTSPSA:
    def test_valid(self):
        inst = PCTSPInstance.random(n=8, seed=42)
        sol = simulated_annealing(inst, max_iterations=5000, seed=42)
        # Check permutation validity (no duplicates, in range)
        assert len(set(sol.tour)) == len(sol.tour)
        for c in sol.tour:
            assert 0 <= c < inst.n

    def test_small(self):
        inst = small_pctsp_6()
        sol = simulated_annealing(inst, max_iterations=10000, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_determinism(self):
        inst = PCTSPInstance.random(n=6, seed=42)
        s1 = simulated_annealing(inst, max_iterations=3000, seed=42)
        s2 = simulated_annealing(inst, max_iterations=3000, seed=42)
        assert abs(s1.objective - s2.objective) < 1e-6

    def test_time_limit(self):
        inst = PCTSPInstance.random(n=10, seed=42)
        sol = simulated_annealing(inst, max_iterations=1000000, time_limit=0.5, seed=42)
        assert len(sol.tour) > 0
