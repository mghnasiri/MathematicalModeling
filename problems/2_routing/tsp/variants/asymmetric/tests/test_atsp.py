"""Tests for Asymmetric TSP."""

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


_inst = _load_mod("atsp_inst_test", os.path.join(_variant_dir, "instance.py"))
ATSPInstance = _inst.ATSPInstance
ATSPSolution = _inst.ATSPSolution
validate_solution = _inst.validate_solution
small_atsp_5 = _inst.small_atsp_5

_heur = _load_mod("atsp_heur_test", os.path.join(_variant_dir, "heuristics.py"))
nearest_neighbor_atsp = _heur.nearest_neighbor_atsp
multi_start_nn = _heur.multi_start_nn

_meta = _load_mod("atsp_meta_test", os.path.join(_variant_dir, "metaheuristics.py"))
simulated_annealing = _meta.simulated_annealing


class TestATSPInstance:
    def test_random(self):
        inst = ATSPInstance.random(n=6, seed=42)
        assert inst.n == 6

    def test_small(self):
        inst = small_atsp_5()
        assert inst.n == 5

    def test_asymmetry(self):
        inst = small_atsp_5()
        # Verify matrix is indeed asymmetric
        assert not np.allclose(inst.dist_matrix, inst.dist_matrix.T)


class TestATSPHeuristics:
    def test_nn_valid(self):
        inst = ATSPInstance.random(n=6, seed=42)
        sol = nearest_neighbor_atsp(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_msnn_valid(self):
        inst = ATSPInstance.random(n=6, seed=42)
        sol = multi_start_nn(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_msnn_beats_nn(self):
        inst = ATSPInstance.random(n=8, seed=42)
        nn = nearest_neighbor_atsp(inst)
        msnn = multi_start_nn(inst)
        assert msnn.cost <= nn.cost + 1e-6

    def test_small(self):
        inst = small_atsp_5()
        sol = multi_start_nn(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"


class TestATSPSA:
    def test_valid(self):
        inst = ATSPInstance.random(n=6, seed=42)
        sol = simulated_annealing(inst, max_iterations=5000, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_competitive(self):
        inst = ATSPInstance.random(n=8, seed=42)
        msnn = multi_start_nn(inst)
        sa = simulated_annealing(inst, max_iterations=10000, seed=42)
        assert sa.cost <= msnn.cost + 1e-6

    def test_determinism(self):
        inst = ATSPInstance.random(n=6, seed=42)
        s1 = simulated_annealing(inst, max_iterations=3000, seed=42)
        s2 = simulated_annealing(inst, max_iterations=3000, seed=42)
        assert abs(s1.cost - s2.cost) < 1e-6

    def test_time_limit(self):
        inst = ATSPInstance.random(n=10, seed=42)
        sol = simulated_annealing(inst, max_iterations=1000000, time_limit=0.5, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"
