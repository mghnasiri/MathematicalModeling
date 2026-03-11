"""Tests for Periodic VRP."""

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


_inst = _load_mod("pvrp_inst_test", os.path.join(_variant_dir, "instance.py"))
PVRPInstance = _inst.PVRPInstance
PVRPSolution = _inst.PVRPSolution
validate_solution = _inst.validate_solution
small_pvrp_6 = _inst.small_pvrp_6

_heur = _load_mod("pvrp_heur_test", os.path.join(_variant_dir, "heuristics.py"))
spread_then_route = _heur.spread_then_route

_meta = _load_mod("pvrp_meta_test", os.path.join(_variant_dir, "metaheuristics.py"))
simulated_annealing = _meta.simulated_annealing


class TestPVRPInstance:
    def test_random(self):
        inst = PVRPInstance.random(n=6, seed=42)
        assert inst.n == 6

    def test_small(self):
        inst = small_pvrp_6()
        assert inst.n == 6
        assert inst.num_periods == 3


class TestPVRPHeuristics:
    def test_spread_valid(self):
        inst = small_pvrp_6()
        sol = spread_then_route(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_random_valid(self):
        inst = PVRPInstance.random(n=6, seed=42)
        sol = spread_then_route(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_positive_dist(self):
        inst = small_pvrp_6()
        sol = spread_then_route(inst)
        assert sol.total_distance > 0


class TestPVRPSA:
    def test_valid(self):
        inst = small_pvrp_6()
        sol = simulated_annealing(inst, max_iterations=5000, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_competitive(self):
        inst = small_pvrp_6()
        heur = spread_then_route(inst)
        sa = simulated_annealing(inst, max_iterations=10000, seed=42)
        assert sa.total_distance <= heur.total_distance + 1e-6

    def test_determinism(self):
        inst = small_pvrp_6()
        s1 = simulated_annealing(inst, max_iterations=3000, seed=42)
        s2 = simulated_annealing(inst, max_iterations=3000, seed=42)
        assert abs(s1.total_distance - s2.total_distance) < 1e-6

    def test_time_limit(self):
        inst = PVRPInstance.random(n=8, seed=42)
        sol = simulated_annealing(inst, max_iterations=1000000, time_limit=0.5, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"
