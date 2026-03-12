"""Tests for Multi-Compartment VRP."""

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


_inst = _load_mod("mcvrp_inst_test", os.path.join(_variant_dir, "instance.py"))
MCVRPInstance = _inst.MCVRPInstance
MCVRPSolution = _inst.MCVRPSolution
validate_solution = _inst.validate_solution
small_mcvrp_6 = _inst.small_mcvrp_6

_heur = _load_mod("mcvrp_heur_test", os.path.join(_variant_dir, "heuristics.py"))
nearest_neighbor_mcvrp = _heur.nearest_neighbor_mcvrp
savings_mcvrp = _heur.savings_mcvrp

_meta = _load_mod("mcvrp_meta_test", os.path.join(_variant_dir, "metaheuristics.py"))
simulated_annealing = _meta.simulated_annealing


class TestMCVRPInstance:
    def test_random(self):
        inst = MCVRPInstance.random(n=8, num_compartments=3, seed=42)
        assert inst.n == 8
        assert inst.num_compartments == 3

    def test_small(self):
        inst = small_mcvrp_6()
        assert inst.n == 6

    def test_route_loads(self):
        inst = small_mcvrp_6()
        loads = inst.route_loads([1, 2])
        assert loads.shape == (3,)


class TestMCVRPHeuristics:
    def test_nn_valid(self):
        inst = small_mcvrp_6()
        sol = nearest_neighbor_mcvrp(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_savings_valid(self):
        inst = small_mcvrp_6()
        sol = savings_mcvrp(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_nn_random(self):
        inst = MCVRPInstance.random(n=10, num_compartments=3, seed=42)
        sol = nearest_neighbor_mcvrp(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"


class TestMCVRPSA:
    def test_valid(self):
        inst = small_mcvrp_6()
        sol = simulated_annealing(inst, max_iterations=5000, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_competitive(self):
        inst = MCVRPInstance.random(n=8, num_compartments=2, seed=42)
        heur = savings_mcvrp(inst)
        sa = simulated_annealing(inst, max_iterations=10000, seed=42)
        assert sa.total_distance <= heur.total_distance + 1e-6

    def test_determinism(self):
        inst = small_mcvrp_6()
        s1 = simulated_annealing(inst, max_iterations=3000, seed=42)
        s2 = simulated_annealing(inst, max_iterations=3000, seed=42)
        assert abs(s1.total_distance - s2.total_distance) < 1e-6

    def test_time_limit(self):
        inst = MCVRPInstance.random(n=10, num_compartments=3, seed=42)
        sol = simulated_annealing(inst, max_iterations=1000000,
                                   time_limit=0.5, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"
