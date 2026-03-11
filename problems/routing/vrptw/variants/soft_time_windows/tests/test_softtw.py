"""Tests for VRPTW with Soft Time Windows."""

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


_inst = _load_mod("softtw_inst_test", os.path.join(_variant_dir, "instance.py"))
SoftTWInstance = _inst.SoftTWInstance
SoftTWSolution = _inst.SoftTWSolution
validate_solution = _inst.validate_solution
small_softtw_6 = _inst.small_softtw_6

_heur = _load_mod("softtw_heur_test", os.path.join(_variant_dir, "heuristics.py"))
nearest_neighbor_stw = _heur.nearest_neighbor_stw

_meta = _load_mod("softtw_meta_test", os.path.join(_variant_dir, "metaheuristics.py"))
simulated_annealing = _meta.simulated_annealing


class TestSoftTWInstance:
    def test_random(self):
        inst = SoftTWInstance.random(n=8, seed=42)
        assert inst.n == 8

    def test_small(self):
        inst = small_softtw_6()
        assert inst.n == 6
        assert inst.penalty_rate == 2.0


class TestSoftTWHeuristics:
    def test_nn_valid(self):
        inst = SoftTWInstance.random(n=8, seed=42)
        sol = nearest_neighbor_stw(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_nn_small(self):
        inst = small_softtw_6()
        sol = nearest_neighbor_stw(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_positive_dist(self):
        inst = SoftTWInstance.random(n=8, seed=42)
        sol = nearest_neighbor_stw(inst)
        assert sol.total_distance > 0


class TestSoftTWSA:
    def test_valid(self):
        inst = SoftTWInstance.random(n=8, seed=42)
        sol = simulated_annealing(inst, max_iterations=5000, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_competitive(self):
        inst = SoftTWInstance.random(n=8, seed=42)
        nn = nearest_neighbor_stw(inst)
        sa = simulated_annealing(inst, max_iterations=10000, seed=42)
        assert sa.total_cost <= nn.total_cost + 1e-6

    def test_determinism(self):
        inst = SoftTWInstance.random(n=6, seed=42)
        s1 = simulated_annealing(inst, max_iterations=3000, seed=42)
        s2 = simulated_annealing(inst, max_iterations=3000, seed=42)
        assert abs(s1.total_cost - s2.total_cost) < 1e-6

    def test_time_limit(self):
        inst = SoftTWInstance.random(n=10, seed=42)
        sol = simulated_annealing(inst, max_iterations=1000000, time_limit=0.5, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"
