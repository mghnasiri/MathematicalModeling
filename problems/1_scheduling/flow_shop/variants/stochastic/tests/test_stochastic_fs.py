"""Tests for Stochastic Flow Shop."""

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


_inst = _load_mod("sfs_inst_test", os.path.join(_variant_dir, "instance.py"))
StochasticFlowShopInstance = _inst.StochasticFlowShopInstance
StochasticFlowShopSolution = _inst.StochasticFlowShopSolution
validate_solution = _inst.validate_solution
small_stoch_fs_4x3 = _inst.small_stoch_fs_4x3

_heur = _load_mod("sfs_heur_test", os.path.join(_variant_dir, "heuristics.py"))
neh_deterministic = _heur.neh_deterministic
neh_stochastic = _heur.neh_stochastic

_meta = _load_mod("sfs_meta_test", os.path.join(_variant_dir, "metaheuristics.py"))
simulated_annealing = _meta.simulated_annealing


class TestStochFSInstance:
    def test_random(self):
        inst = StochasticFlowShopInstance.random(n=6, m=3, seed=42)
        assert inst.n == 6
        assert inst.m == 3

    def test_small(self):
        inst = small_stoch_fs_4x3()
        assert inst.n == 4
        assert inst.m == 3

    def test_sample_times(self):
        inst = small_stoch_fs_4x3()
        rng = np.random.default_rng(42)
        times = inst.sample_times(rng)
        assert times.shape == (4, 3)
        assert (times > 0).all()

    def test_deterministic_makespan(self):
        inst = small_stoch_fs_4x3()
        ms = inst.deterministic_makespan([0, 1, 2, 3])
        assert ms > 0


class TestStochFSHeuristics:
    def test_neh_det_valid(self):
        inst = StochasticFlowShopInstance.random(n=6, m=3, seed=42)
        sol = neh_deterministic(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_neh_stoch_valid(self):
        inst = StochasticFlowShopInstance.random(n=6, m=3, seed=42)
        sol = neh_stochastic(inst, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_neh_small(self):
        inst = small_stoch_fs_4x3()
        sol = neh_deterministic(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"


class TestStochFSSA:
    def test_valid(self):
        inst = StochasticFlowShopInstance.random(n=6, m=3, seed=42)
        sol = simulated_annealing(inst, max_iterations=3000, num_samples=10,
                                   seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_determinism(self):
        inst = small_stoch_fs_4x3()
        s1 = simulated_annealing(inst, max_iterations=2000, num_samples=10,
                                  seed=42)
        s2 = simulated_annealing(inst, max_iterations=2000, num_samples=10,
                                  seed=42)
        assert s1.permutation == s2.permutation

    def test_time_limit(self):
        inst = StochasticFlowShopInstance.random(n=8, m=4, seed=42)
        sol = simulated_annealing(inst, max_iterations=1000000,
                                   num_samples=10, time_limit=0.5, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"
