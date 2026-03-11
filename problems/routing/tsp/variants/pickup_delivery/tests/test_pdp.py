"""Tests for Pickup and Delivery Problem."""

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


_inst = _load_mod("pdp_inst_test", os.path.join(_variant_dir, "instance.py"))
PDPInstance = _inst.PDPInstance
PDPSolution = _inst.PDPSolution
validate_solution = _inst.validate_solution
small_pdp_3 = _inst.small_pdp_3

_heur = _load_mod("pdp_heur_test", os.path.join(_variant_dir, "heuristics.py"))
nearest_feasible = _heur.nearest_feasible
cheapest_pair_insertion = _heur.cheapest_pair_insertion

_meta = _load_mod("pdp_meta_test", os.path.join(_variant_dir, "metaheuristics.py"))
simulated_annealing = _meta.simulated_annealing


class TestPDPInstance:
    def test_random_creation(self):
        inst = PDPInstance.random(num_pairs=5, seed=42)
        assert inst.num_pairs == 5
        assert inst.num_locations == 11

    def test_small_benchmark(self):
        inst = small_pdp_3()
        assert inst.num_pairs == 3
        assert inst.num_locations == 7

    def test_precedence_valid(self):
        inst = small_pdp_3()
        assert inst.precedence_feasible([0, 1, 2, 3, 4, 5, 6])

    def test_precedence_invalid(self):
        inst = small_pdp_3()
        # delivery 1 (=4) before pickup 1 (=1)
        assert not inst.precedence_feasible([0, 4, 1, 2, 3, 5, 6])


class TestPDPHeuristics:
    def test_nearest_feasible_valid(self):
        inst = PDPInstance.random(num_pairs=5, seed=42)
        sol = nearest_feasible(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_cheapest_insertion_valid(self):
        inst = PDPInstance.random(num_pairs=5, seed=42)
        sol = cheapest_pair_insertion(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_small_benchmark(self):
        inst = small_pdp_3()
        sol = nearest_feasible(inst)
        assert sol.feasible
        assert sol.distance > 0


class TestPDPSA:
    def test_valid(self):
        inst = PDPInstance.random(num_pairs=5, seed=42)
        sol = simulated_annealing(inst, max_iterations=10000, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_competitive(self):
        inst = PDPInstance.random(num_pairs=5, seed=42)
        nf_sol = nearest_feasible(inst)
        sa_sol = simulated_annealing(inst, max_iterations=20000, seed=42)
        assert sa_sol.distance <= nf_sol.distance + 1e-6

    def test_determinism(self):
        inst = PDPInstance.random(num_pairs=4, seed=42)
        sol1 = simulated_annealing(inst, max_iterations=5000, seed=42)
        sol2 = simulated_annealing(inst, max_iterations=5000, seed=42)
        assert abs(sol1.distance - sol2.distance) < 1e-6

    def test_time_limit(self):
        inst = PDPInstance.random(num_pairs=6, seed=42)
        sol = simulated_annealing(inst, max_iterations=1000000, time_limit=0.5, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"
