"""Tests for Multi-Trip VRP."""

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


_inst = _load_mod("mtvrp_inst_test", os.path.join(_variant_dir, "instance.py"))
MTVRPInstance = _inst.MTVRPInstance
MTVRPSolution = _inst.MTVRPSolution
validate_solution = _inst.validate_solution
small_mtvrp_8 = _inst.small_mtvrp_8

_heur = _load_mod("mtvrp_heur_test", os.path.join(_variant_dir, "heuristics.py"))
greedy_multi_trip = _heur.greedy_multi_trip

_meta = _load_mod("mtvrp_meta_test", os.path.join(_variant_dir, "metaheuristics.py"))
simulated_annealing = _meta.simulated_annealing


class TestMTVRPInstance:
    def test_random(self):
        inst = MTVRPInstance.random(n=12, num_vehicles=2, seed=42)
        assert inst.n == 12
        assert inst.num_vehicles == 2

    def test_small(self):
        inst = small_mtvrp_8()
        assert inst.n == 8
        assert inst.num_vehicles == 2


class TestMTVRPHeuristics:
    def test_greedy_valid(self):
        inst = small_mtvrp_8()
        sol = greedy_multi_trip(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_greedy_random(self):
        inst = MTVRPInstance.random(n=12, num_vehicles=2, seed=42)
        sol = greedy_multi_trip(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"


class TestMTVRPSA:
    def test_valid(self):
        inst = small_mtvrp_8()
        sol = simulated_annealing(inst, max_iterations=5000, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_determinism(self):
        inst = small_mtvrp_8()
        s1 = simulated_annealing(inst, max_iterations=3000, seed=42)
        s2 = simulated_annealing(inst, max_iterations=3000, seed=42)
        assert abs(s1.total_distance - s2.total_distance) < 1e-6

    def test_time_limit(self):
        inst = MTVRPInstance.random(n=12, num_vehicles=2, seed=42)
        sol = simulated_annealing(inst, max_iterations=1000000,
                                   time_limit=0.5, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"
