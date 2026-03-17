"""Tests for Multi-Depot VRP."""

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


_inst = _load_mod("mdvrp_inst_test", os.path.join(_variant_dir, "instance.py"))
MDVRPInstance = _inst.MDVRPInstance
MDVRPSolution = _inst.MDVRPSolution
validate_solution = _inst.validate_solution
small_mdvrp_2_6 = _inst.small_mdvrp_2_6

_heur = _load_mod("mdvrp_heur_test", os.path.join(_variant_dir, "heuristics.py"))
nearest_depot_nn = _heur.nearest_depot_nn

_meta = _load_mod("mdvrp_meta_test", os.path.join(_variant_dir, "metaheuristics.py"))
simulated_annealing = _meta.simulated_annealing


class TestMDVRPInstance:
    def test_random(self):
        inst = MDVRPInstance.random(num_depots=2, n=10, seed=42)
        assert inst.num_depots == 2
        assert inst.n == 10

    def test_small(self):
        inst = small_mdvrp_2_6()
        assert inst.n == 6


class TestMDVRPHeuristics:
    def test_nn_valid(self):
        inst = MDVRPInstance.random(num_depots=2, n=10, seed=42)
        sol = nearest_depot_nn(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_positive_dist(self):
        inst = MDVRPInstance.random(seed=42)
        sol = nearest_depot_nn(inst)
        assert sol.total_distance > 0

    def test_small(self):
        inst = small_mdvrp_2_6()
        sol = nearest_depot_nn(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"


class TestMDVRPSA:
    def test_valid(self):
        inst = MDVRPInstance.random(num_depots=2, n=10, seed=42)
        sol = simulated_annealing(inst, max_iterations=5000, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_competitive(self):
        inst = MDVRPInstance.random(num_depots=2, n=10, seed=42)
        nn = nearest_depot_nn(inst)
        sa = simulated_annealing(inst, max_iterations=10000, seed=42)
        assert sa.total_distance <= nn.total_distance + 1e-6

    def test_determinism(self):
        inst = MDVRPInstance.random(num_depots=2, n=8, seed=42)
        s1 = simulated_annealing(inst, max_iterations=3000, seed=42)
        s2 = simulated_annealing(inst, max_iterations=3000, seed=42)
        assert abs(s1.total_distance - s2.total_distance) < 1e-6

    def test_time_limit(self):
        inst = MDVRPInstance.random(num_depots=3, n=12, seed=42)
        sol = simulated_annealing(inst, max_iterations=1000000, time_limit=0.5, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"
