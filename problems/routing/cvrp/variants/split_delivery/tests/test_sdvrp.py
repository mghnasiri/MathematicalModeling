"""Tests for Split Delivery VRP."""

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


_inst = _load_mod("sdvrp_inst_test", os.path.join(_variant_dir, "instance.py"))
SDVRPInstance = _inst.SDVRPInstance
SDVRPSolution = _inst.SDVRPSolution
validate_solution = _inst.validate_solution
small_sdvrp_6 = _inst.small_sdvrp_6

_heur = _load_mod("sdvrp_heur_test", os.path.join(_variant_dir, "heuristics.py"))
nearest_neighbor_split = _heur.nearest_neighbor_split
savings_split = _heur.savings_split

_meta = _load_mod("sdvrp_meta_test", os.path.join(_variant_dir, "metaheuristics.py"))
simulated_annealing = _meta.simulated_annealing


class TestSDVRPInstance:
    def test_random(self):
        inst = SDVRPInstance.random(n=8, seed=42)
        assert inst.n == 8

    def test_small(self):
        inst = small_sdvrp_6()
        assert inst.n == 6
        assert inst.capacity == 40.0


class TestSDVRPHeuristics:
    def test_nn_valid(self):
        inst = SDVRPInstance.random(n=8, seed=42)
        sol = nearest_neighbor_split(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_savings_valid(self):
        inst = SDVRPInstance.random(n=8, seed=42)
        sol = savings_split(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_small_nn(self):
        inst = small_sdvrp_6()
        sol = nearest_neighbor_split(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_positive_dist(self):
        inst = SDVRPInstance.random(n=8, seed=42)
        sol = nearest_neighbor_split(inst)
        assert sol.total_distance > 0


class TestSDVRPSA:
    def test_valid(self):
        inst = SDVRPInstance.random(n=8, seed=42)
        sol = simulated_annealing(inst, max_iterations=5000, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_competitive(self):
        inst = SDVRPInstance.random(n=8, seed=42)
        nn = nearest_neighbor_split(inst)
        sa = simulated_annealing(inst, max_iterations=10000, seed=42)
        assert sa.total_distance <= nn.total_distance + 1e-6

    def test_determinism(self):
        inst = SDVRPInstance.random(n=6, seed=42)
        s1 = simulated_annealing(inst, max_iterations=3000, seed=42)
        s2 = simulated_annealing(inst, max_iterations=3000, seed=42)
        assert abs(s1.total_distance - s2.total_distance) < 1e-6

    def test_time_limit(self):
        inst = SDVRPInstance.random(n=10, seed=42)
        sol = simulated_annealing(inst, max_iterations=1000000, time_limit=0.5, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"
