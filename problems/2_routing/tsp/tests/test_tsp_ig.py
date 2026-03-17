"""Tests for Iterated Greedy on TSP."""

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


_inst = _load_mod("tsp_instance_test_ig", os.path.join(_parent_dir, "instance.py"))
TSPInstance = _inst.TSPInstance
TSPSolution = _inst.TSPSolution
validate_tour = _inst.validate_tour
small5 = _inst.small5
gr17 = _inst.gr17

_ig = _load_mod(
    "tsp_ig_test",
    os.path.join(_parent_dir, "metaheuristics", "iterated_greedy.py"),
)
iterated_greedy = _ig.iterated_greedy

_nn = _load_mod(
    "tsp_nn_test_ig",
    os.path.join(_parent_dir, "heuristics", "nearest_neighbor.py"),
)
nearest_neighbor_multistart = _nn.nearest_neighbor_multistart


class TestTSPIGValidity:
    def test_small5_valid(self):
        inst = small5()
        sol = iterated_greedy(inst, seed=42)
        valid, errors = validate_tour(inst, sol.tour)
        assert valid, f"Invalid: {errors}"

    def test_random_valid(self):
        inst = TSPInstance.random(n=15, seed=42)
        sol = iterated_greedy(inst, seed=42)
        valid, errors = validate_tour(inst, sol.tour)
        assert valid, f"Invalid: {errors}"

    def test_distance_matches(self):
        inst = TSPInstance.random(n=10, seed=42)
        sol = iterated_greedy(inst, seed=42)
        actual = inst.tour_distance(sol.tour)
        assert abs(sol.distance - actual) < 1e-6


class TestTSPIGQuality:
    def test_ig_competitive_with_nn(self):
        inst = TSPInstance.random(n=15, seed=42)
        nn_sol = nearest_neighbor_multistart(inst)
        ig_sol = iterated_greedy(inst, max_iterations=1000, seed=42)
        assert ig_sol.distance <= nn_sol.distance * 1.05


class TestTSPIGDeterminism:
    def test_same_seed(self):
        inst = TSPInstance.random(n=10, seed=42)
        sol1 = iterated_greedy(inst, max_iterations=200, seed=42)
        sol2 = iterated_greedy(inst, max_iterations=200, seed=42)
        assert abs(sol1.distance - sol2.distance) < 1e-6

    def test_different_seed_both_valid(self):
        inst = TSPInstance.random(n=10, seed=42)
        sol1 = iterated_greedy(inst, seed=1)
        sol2 = iterated_greedy(inst, seed=999)
        v1, _ = validate_tour(inst, sol1.tour)
        v2, _ = validate_tour(inst, sol2.tour)
        assert v1 and v2


class TestTSPIGEdgeCases:
    def test_time_limit(self):
        inst = TSPInstance.random(n=20, seed=42)
        sol = iterated_greedy(inst, max_iterations=1000000, time_limit=0.5, seed=42)
        valid, errors = validate_tour(inst, sol.tour)
        assert valid, f"Invalid: {errors}"
