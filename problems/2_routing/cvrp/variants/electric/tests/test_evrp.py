"""Tests for Electric Vehicle Routing Problem."""

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


_inst = _load_mod("evrp_inst_test", os.path.join(_variant_dir, "instance.py"))
EVRPInstance = _inst.EVRPInstance
EVRPSolution = _inst.EVRPSolution
validate_solution = _inst.validate_solution
small_evrp_6 = _inst.small_evrp_6

_heur = _load_mod("evrp_heur_test", os.path.join(_variant_dir, "heuristics.py"))
nearest_neighbor_evrp = _heur.nearest_neighbor_evrp

_meta = _load_mod("evrp_meta_test", os.path.join(_variant_dir, "metaheuristics.py"))
simulated_annealing = _meta.simulated_annealing


class TestEVRPInstance:
    def test_random(self):
        inst = EVRPInstance.random(n=8, num_stations=2, seed=42)
        assert inst.n == 8
        assert inst.num_stations == 2
        assert inst.total_nodes == 11

    def test_small(self):
        inst = small_evrp_6()
        assert inst.n == 6
        assert inst.num_stations == 2
        assert len(inst.station_nodes) == 2

    def test_dist(self):
        inst = small_evrp_6()
        assert inst.dist(0, 0) == 0.0
        assert inst.dist(0, 1) > 0

    def test_energy_cost(self):
        inst = small_evrp_6()
        assert inst.energy_cost(0, 1) == inst.energy_rate * inst.dist(0, 1)


class TestEVRPHeuristics:
    def test_nn_valid(self):
        inst = small_evrp_6()
        sol = nearest_neighbor_evrp(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_nn_random(self):
        inst = EVRPInstance.random(n=8, num_stations=3, battery_capacity=200.0,
                                   seed=42)
        sol = nearest_neighbor_evrp(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_nn_all_customers_visited(self):
        inst = small_evrp_6()
        sol = nearest_neighbor_evrp(inst)
        visited = set()
        for route in sol.routes:
            for node in route:
                if 1 <= node <= inst.n:
                    visited.add(node)
        assert visited == set(range(1, inst.n + 1))


class TestEVRPSA:
    def test_valid(self):
        inst = small_evrp_6()
        sol = simulated_annealing(inst, max_iterations=5000, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_determinism(self):
        inst = small_evrp_6()
        s1 = simulated_annealing(inst, max_iterations=3000, seed=42)
        s2 = simulated_annealing(inst, max_iterations=3000, seed=42)
        assert abs(s1.total_distance - s2.total_distance) < 1e-6

    def test_time_limit(self):
        inst = EVRPInstance.random(n=8, num_stations=3, battery_capacity=200.0,
                                   seed=42)
        sol = simulated_annealing(inst, max_iterations=1000000, time_limit=0.5,
                                   seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"
