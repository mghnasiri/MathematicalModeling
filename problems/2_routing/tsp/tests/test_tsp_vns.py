"""Tests for VNS on TSP."""

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


_inst = _load_mod("tsp_instance_test_vns", os.path.join(_parent_dir, "instance.py"))
TSPInstance = _inst.TSPInstance
TSPSolution = _inst.TSPSolution
validate_tour = _inst.validate_tour

_vns = _load_mod(
    "tsp_vns_test",
    os.path.join(_parent_dir, "metaheuristics", "vns.py"),
)
vns = _vns.vns

_nn = _load_mod(
    "tsp_nn_test_vns",
    os.path.join(_parent_dir, "heuristics", "nearest_neighbor.py"),
)
nearest_neighbor = _nn.nearest_neighbor


class TestTSPVNSValidity:
    def test_valid_tour(self):
        inst = TSPInstance.random(n=10, seed=42)
        sol = vns(inst, seed=42)
        valid, errors = validate_tour(inst, sol.tour)
        assert valid, f"Validation errors: {errors}"

    def test_distance_matches(self):
        inst = TSPInstance.random(n=10, seed=42)
        sol = vns(inst, seed=42)
        actual = inst.tour_distance(sol.tour)
        assert abs(sol.distance - actual) < 1e-6


class TestTSPVNSQuality:
    def test_competitive_with_nn(self):
        inst = TSPInstance.random(n=12, seed=42)
        nn_sol = nearest_neighbor(inst)
        vns_sol = vns(inst, max_iterations=200, seed=42)
        assert vns_sol.distance <= nn_sol.distance + 1e-6

    def test_positive_distance(self):
        inst = TSPInstance.random(n=10, seed=42)
        sol = vns(inst, seed=42)
        assert sol.distance > 0


class TestTSPVNSDeterminism:
    def test_same_seed(self):
        inst = TSPInstance.random(n=8, seed=42)
        sol1 = vns(inst, max_iterations=100, seed=42)
        sol2 = vns(inst, max_iterations=100, seed=42)
        assert abs(sol1.distance - sol2.distance) < 1e-6

    def test_different_seed_both_valid(self):
        inst = TSPInstance.random(n=8, seed=42)
        sol1 = vns(inst, seed=1)
        sol2 = vns(inst, seed=999)
        valid1, _ = validate_tour(inst, sol1.tour)
        valid2, _ = validate_tour(inst, sol2.tour)
        assert valid1
        assert valid2


class TestTSPVNSEdgeCases:
    def test_small_instance(self):
        inst = TSPInstance.random(n=3, seed=42)
        sol = vns(inst, seed=42)
        assert sorted(sol.tour) == [0, 1, 2]

    def test_time_limit(self):
        inst = TSPInstance.random(n=15, seed=42)
        sol = vns(inst, max_iterations=1000000, time_limit=0.5, seed=42)
        valid, errors = validate_tour(inst, sol.tour)
        assert valid, f"Validation errors: {errors}"
