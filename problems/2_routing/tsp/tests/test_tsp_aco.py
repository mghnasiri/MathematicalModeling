"""Tests for Ant Colony Optimization on TSP."""

from __future__ import annotations

import sys
import os
import importlib.util

import numpy as np
import pytest

_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_mod(name, filepath):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_inst = _load_mod("tsp_instance_test_aco", os.path.join(_parent_dir, "instance.py"))
TSPInstance = _inst.TSPInstance
TSPSolution = _inst.TSPSolution
validate_tour = _inst.validate_tour
small4 = _inst.small4
small5 = _inst.small5
gr17 = _inst.gr17

_aco = _load_mod(
    "tsp_aco_test",
    os.path.join(_parent_dir, "metaheuristics", "ant_colony.py"),
)
ant_colony = _aco.ant_colony

_nn_mod = _load_mod(
    "tsp_nn_test_aco",
    os.path.join(_parent_dir, "heuristics", "nearest_neighbor.py"),
)
nearest_neighbor = _nn_mod.nearest_neighbor


class TestTSPACOValidity:
    """Test that ACO produces valid tours."""

    def test_small4_valid(self):
        inst = small4()
        sol = ant_colony(inst, seed=42)
        valid, errors = validate_tour(inst, sol.tour)
        assert valid, f"Invalid tour: {errors}"

    def test_small5_valid(self):
        inst = small5()
        sol = ant_colony(inst, seed=42)
        valid, errors = validate_tour(inst, sol.tour)
        assert valid, f"Invalid tour: {errors}"

    def test_random_instance_valid(self):
        inst = TSPInstance.random(n=15, seed=42)
        sol = ant_colony(inst, seed=42)
        valid, errors = validate_tour(inst, sol.tour)
        assert valid, f"Invalid tour: {errors}"

    def test_distance_matches(self):
        inst = small5()
        sol = ant_colony(inst, seed=42)
        actual = inst.tour_distance(sol.tour)
        assert abs(sol.distance - actual) < 1e-6


class TestTSPACOQuality:
    """Test solution quality."""

    def test_small4_optimal(self):
        inst = small4()
        sol = ant_colony(inst, max_iterations=100, seed=42)
        assert sol.distance <= 9.0  # Optimal = 8.0, allow small margin

    def test_gr17_reasonable(self):
        inst = gr17()
        sol = ant_colony(inst, max_iterations=200, seed=42)
        assert sol.distance >= 2016  # Optimal = 2016
        assert sol.distance <= 2500  # Should be reasonably good

    def test_aco_competitive_with_nn(self):
        inst = TSPInstance.random(n=15, seed=42)
        nn_sol = nearest_neighbor(inst)
        aco_sol = ant_colony(inst, max_iterations=100, seed=42)
        assert aco_sol.distance <= nn_sol.distance * 1.05


class TestTSPACODeterminism:
    """Test deterministic behavior with seed."""

    def test_same_seed_same_result(self):
        inst = small5()
        sol1 = ant_colony(inst, max_iterations=50, seed=42)
        sol2 = ant_colony(inst, max_iterations=50, seed=42)
        assert abs(sol1.distance - sol2.distance) < 1e-6

    def test_different_seed_both_valid(self):
        inst = small5()
        sol1 = ant_colony(inst, max_iterations=50, seed=1)
        sol2 = ant_colony(inst, max_iterations=50, seed=999)
        valid1, _ = validate_tour(inst, sol1.tour)
        valid2, _ = validate_tour(inst, sol2.tour)
        assert valid1 and valid2


class TestTSPACOEdgeCases:
    """Test edge cases."""

    def test_three_cities(self):
        dist = [[0, 1, 2], [1, 0, 3], [2, 3, 0]]
        inst = TSPInstance.from_distance_matrix(dist)
        sol = ant_colony(inst, seed=42)
        assert set(sol.tour) == {0, 1, 2}

    def test_time_limit(self):
        inst = TSPInstance.random(n=20, seed=42)
        sol = ant_colony(inst, max_iterations=1000000, time_limit=0.5, seed=42)
        valid, errors = validate_tour(inst, sol.tour)
        assert valid, f"Invalid: {errors}"
