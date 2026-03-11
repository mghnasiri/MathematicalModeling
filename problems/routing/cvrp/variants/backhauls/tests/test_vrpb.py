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


_inst = _load_mod("vrpb_inst_test", os.path.join(_variant_dir, "instance.py"))
VRPBInstance = _inst.VRPBInstance
VRPBSolution = _inst.VRPBSolution
validate_solution = _inst.validate_solution
small_vrpb_4_3 = _inst.small_vrpb_4_3

_heur = _load_mod("vrpb_heur_test", os.path.join(_variant_dir, "heuristics.py"))
nearest_neighbor_vrpb = _heur.nearest_neighbor_vrpb

_meta = _load_mod("vrpb_meta_test", os.path.join(_variant_dir, "metaheuristics.py"))
simulated_annealing = _meta.simulated_annealing


class TestVRPBInstance:
    def test_random_creation(self):
        inst = VRPBInstance.random(n_linehaul=6, n_backhaul=4, seed=42)
        assert inst.n == 10
        assert inst.n_linehaul == 6
        assert inst.n_backhaul == 4

    def test_small_benchmark(self):
        inst = small_vrpb_4_3()
        assert inst.n == 7

    def test_customer_types(self):
        inst = small_vrpb_4_3()
        assert inst.is_linehaul(1)
        assert inst.is_linehaul(4)
        assert inst.is_backhaul(5)
        assert inst.is_backhaul(7)

    def test_precedence(self):
        inst = small_vrpb_4_3()
        assert inst.route_precedence_feasible([1, 2, 5, 6])
        assert not inst.route_precedence_feasible([5, 1, 2, 6])


class TestVRPBHeuristics:
    def test_nn_valid(self):
        inst = VRPBInstance.random(n_linehaul=6, n_backhaul=4, seed=42)
        sol = nearest_neighbor_vrpb(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_positive_distance(self):
        inst = VRPBInstance.random(seed=42)
        sol = nearest_neighbor_vrpb(inst)
        assert sol.total_distance > 0

    def test_small_benchmark(self):
        inst = small_vrpb_4_3()
        sol = nearest_neighbor_vrpb(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"


class TestVRPBSA:
    def test_valid(self):
        inst = VRPBInstance.random(n_linehaul=6, n_backhaul=4, seed=42)
        sol = simulated_annealing(inst, max_iterations=5000, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_competitive(self):
        inst = VRPBInstance.random(n_linehaul=6, n_backhaul=4, seed=42)
        nn_sol = nearest_neighbor_vrpb(inst)
        sa_sol = simulated_annealing(inst, max_iterations=10000, seed=42)
        assert sa_sol.total_distance <= nn_sol.total_distance + 1e-6

    def test_determinism(self):
        inst = VRPBInstance.random(n_linehaul=4, n_backhaul=3, seed=42)
        sol1 = simulated_annealing(inst, max_iterations=3000, seed=42)
        sol2 = simulated_annealing(inst, max_iterations=3000, seed=42)
        assert abs(sol1.total_distance - sol2.total_distance) < 1e-6

    def test_time_limit(self):
        inst = VRPBInstance.random(n_linehaul=8, n_backhaul=5, seed=42)
        sol = simulated_annealing(inst, max_iterations=1000000, time_limit=0.5, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"
