"""Tests for VRP with Backhauls."""

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


_inst = _load_mod("vrpb_backhaul_inst_test", os.path.join(_variant_dir, "instance.py"))
VRPBInstance = _inst.VRPBInstance
VRPBSolution = _inst.VRPBSolution
validate_solution = _inst.validate_solution
small_vrpb_5 = _inst.small_vrpb_5

_heur = _load_mod("vrpb_backhaul_heur_test", os.path.join(_variant_dir, "heuristics.py"))
nearest_neighbor_vrpb = _heur.nearest_neighbor_vrpb

_meta = _load_mod("vrpb_backhaul_meta_test", os.path.join(_variant_dir, "metaheuristics.py"))
simulated_annealing = _meta.simulated_annealing


class TestVRPBInstance:
    def test_random(self):
        inst = VRPBInstance.random(n_linehaul=6, n_backhaul=3, seed=42)
        assert inst.n_linehaul == 6
        assert inst.n_backhaul == 3
        assert inst.n_total == 9

    def test_small(self):
        inst = small_vrpb_5()
        assert inst.n_linehaul == 3
        assert inst.n_backhaul == 2

    def test_node_classification(self):
        inst = small_vrpb_5()
        assert inst.is_linehaul(1)
        assert inst.is_linehaul(3)
        assert inst.is_backhaul(4)
        assert not inst.is_linehaul(4)


class TestVRPBHeuristics:
    def test_nn_valid(self):
        inst = small_vrpb_5()
        sol = nearest_neighbor_vrpb(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_nn_random(self):
        inst = VRPBInstance.random(n_linehaul=8, n_backhaul=4, seed=42)
        sol = nearest_neighbor_vrpb(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"


class TestVRPBSA:
    def test_valid(self):
        inst = small_vrpb_5()
        sol = simulated_annealing(inst, max_iterations=5000, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_determinism(self):
        inst = small_vrpb_5()
        s1 = simulated_annealing(inst, max_iterations=3000, seed=42)
        s2 = simulated_annealing(inst, max_iterations=3000, seed=42)
        assert abs(s1.total_distance - s2.total_distance) < 1e-6

    def test_time_limit(self):
        inst = VRPBInstance.random(n_linehaul=8, n_backhaul=4, seed=42)
        sol = simulated_annealing(inst, max_iterations=1000000,
                                   time_limit=0.5, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"
