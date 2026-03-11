"""Tests for Open Vehicle Routing Problem."""

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


_inst = _load_mod("ovrp_inst_test", os.path.join(_variant_dir, "instance.py"))
OVRPInstance = _inst.OVRPInstance
OVRPSolution = _inst.OVRPSolution
validate_solution = _inst.validate_solution
small_ovrp_6 = _inst.small_ovrp_6

_heur = _load_mod("ovrp_heur_test", os.path.join(_variant_dir, "heuristics.py"))
nearest_neighbor_ovrp = _heur.nearest_neighbor_ovrp
savings_ovrp = _heur.savings_ovrp

_meta = _load_mod("ovrp_meta_test", os.path.join(_variant_dir, "metaheuristics.py"))
simulated_annealing = _meta.simulated_annealing


class TestOVRPInstance:
    def test_random_creation(self):
        inst = OVRPInstance.random(n=8, seed=42)
        assert inst.n == 8
        assert inst.distance_matrix.shape == (9, 9)

    def test_small_benchmark(self):
        inst = small_ovrp_6()
        assert inst.n == 6
        assert inst.capacity == 50

    def test_route_distance(self):
        inst = small_ovrp_6()
        dist = inst.route_distance([1, 2, 3])
        assert dist > 0


class TestOVRPHeuristics:
    def test_nn_valid(self):
        inst = OVRPInstance.random(n=10, num_vehicles=3, seed=42)
        sol = nearest_neighbor_ovrp(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_savings_valid(self):
        inst = OVRPInstance.random(n=10, num_vehicles=3, seed=42)
        sol = savings_ovrp(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_positive_distance(self):
        inst = OVRPInstance.random(n=10, seed=42)
        sol = nearest_neighbor_ovrp(inst)
        assert sol.total_distance > 0

    def test_small_benchmark(self):
        inst = small_ovrp_6()
        sol = nearest_neighbor_ovrp(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"


class TestOVRPSA:
    def test_valid(self):
        inst = OVRPInstance.random(n=10, num_vehicles=3, seed=42)
        sol = simulated_annealing(inst, max_iterations=5000, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_competitive(self):
        inst = OVRPInstance.random(n=10, num_vehicles=3, seed=42)
        nn_sol = nearest_neighbor_ovrp(inst)
        sa_sol = simulated_annealing(inst, max_iterations=10000, seed=42)
        assert sa_sol.total_distance <= nn_sol.total_distance + 1e-6

    def test_determinism(self):
        inst = OVRPInstance.random(n=8, num_vehicles=2, seed=42)
        sol1 = simulated_annealing(inst, max_iterations=3000, seed=42)
        sol2 = simulated_annealing(inst, max_iterations=3000, seed=42)
        assert abs(sol1.total_distance - sol2.total_distance) < 1e-6

    def test_time_limit(self):
        inst = OVRPInstance.random(n=12, num_vehicles=3, seed=42)
        sol = simulated_annealing(inst, max_iterations=1000000, time_limit=0.5, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"
