"""Tests for Generalized Assignment Problem."""

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


_inst = _load_mod("gap_inst_test", os.path.join(_variant_dir, "instance.py"))
GAPInstance = _inst.GAPInstance
GAPSolution = _inst.GAPSolution
validate_solution = _inst.validate_solution
small_gap_6x3 = _inst.small_gap_6x3

_heur = _load_mod("gap_heur_test", os.path.join(_variant_dir, "heuristics.py"))
greedy_ratio = _heur.greedy_ratio
first_fit_decreasing = _heur.first_fit_decreasing

_meta = _load_mod("gap_meta_test", os.path.join(_variant_dir, "metaheuristics.py"))
simulated_annealing = _meta.simulated_annealing


class TestGAPInstance:
    def test_random(self):
        inst = GAPInstance.random(n=8, m=3, seed=42)
        assert inst.n == 8
        assert inst.m == 3
        assert inst.cost.shape == (3, 8)
        assert inst.resource.shape == (3, 8)

    def test_small(self):
        inst = small_gap_6x3()
        assert inst.n == 6
        assert inst.m == 3

    def test_total_cost(self):
        inst = small_gap_6x3()
        assignment = [0, 1, 0, 1, 2, 2]
        cost = inst.total_cost(assignment)
        expected = inst.cost[0][0] + inst.cost[1][1] + inst.cost[0][2] + \
                   inst.cost[1][3] + inst.cost[2][4] + inst.cost[2][5]
        assert abs(cost - expected) < 1e-6


class TestGAPHeuristics:
    def test_greedy_valid(self):
        inst = small_gap_6x3()
        sol = greedy_ratio(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_ffd_valid(self):
        inst = small_gap_6x3()
        sol = first_fit_decreasing(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_greedy_random(self):
        inst = GAPInstance.random(n=10, m=4, seed=42)
        sol = greedy_ratio(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_ffd_random(self):
        inst = GAPInstance.random(n=10, m=4, seed=42)
        sol = first_fit_decreasing(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"


class TestGAPSA:
    def test_valid(self):
        inst = small_gap_6x3()
        sol = simulated_annealing(inst, max_iterations=5000, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_competitive(self):
        inst = GAPInstance.random(n=8, m=3, seed=42)
        greedy = greedy_ratio(inst)
        sa = simulated_annealing(inst, max_iterations=10000, seed=42)
        assert sa.total_cost <= greedy.total_cost + 1e-6

    def test_determinism(self):
        inst = GAPInstance.random(n=6, m=3, seed=42)
        s1 = simulated_annealing(inst, max_iterations=3000, seed=42)
        s2 = simulated_annealing(inst, max_iterations=3000, seed=42)
        assert abs(s1.total_cost - s2.total_cost) < 1e-6

    def test_time_limit(self):
        inst = GAPInstance.random(n=10, m=4, seed=42)
        sol = simulated_annealing(inst, max_iterations=1000000, time_limit=0.5, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"
