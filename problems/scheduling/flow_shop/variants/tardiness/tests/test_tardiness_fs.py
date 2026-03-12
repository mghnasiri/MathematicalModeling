"""Tests for Tardiness Flow Shop."""

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


_inst = _load_mod("tfs_inst_test", os.path.join(_variant_dir, "instance.py"))
TardinessFlowShopInstance = _inst.TardinessFlowShopInstance
TardinessFlowShopSolution = _inst.TardinessFlowShopSolution
validate_solution = _inst.validate_solution
small_tfs_4x3 = _inst.small_tfs_4x3

_heur = _load_mod("tfs_heur_test", os.path.join(_variant_dir, "heuristics.py"))
edd_rule = _heur.edd_rule
wspt_rule = _heur.wspt_rule
neh_tardiness = _heur.neh_tardiness

_meta = _load_mod("tfs_meta_test", os.path.join(_variant_dir, "metaheuristics.py"))
simulated_annealing = _meta.simulated_annealing


class TestTFSInstance:
    def test_random(self):
        inst = TardinessFlowShopInstance.random(n=6, m=3, seed=42)
        assert inst.n == 6

    def test_small(self):
        inst = small_tfs_4x3()
        assert inst.n == 4


class TestTFSHeuristics:
    def test_edd_valid(self):
        inst = TardinessFlowShopInstance.random(n=6, m=3, seed=42)
        sol = edd_rule(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_wspt_valid(self):
        inst = TardinessFlowShopInstance.random(n=6, m=3, seed=42)
        sol = wspt_rule(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_neh_valid(self):
        inst = TardinessFlowShopInstance.random(n=6, m=3, seed=42)
        sol = neh_tardiness(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_small(self):
        inst = small_tfs_4x3()
        sol = neh_tardiness(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"


class TestTFSSA:
    def test_valid(self):
        inst = TardinessFlowShopInstance.random(n=6, m=3, seed=42)
        sol = simulated_annealing(inst, max_iterations=5000, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_competitive(self):
        inst = TardinessFlowShopInstance.random(n=8, m=3, seed=42)
        neh = neh_tardiness(inst)
        sa = simulated_annealing(inst, max_iterations=10000, seed=42)
        assert sa.total_weighted_tardiness <= neh.total_weighted_tardiness + 1e-6

    def test_determinism(self):
        inst = TardinessFlowShopInstance.random(n=6, m=3, seed=42)
        s1 = simulated_annealing(inst, max_iterations=3000, seed=42)
        s2 = simulated_annealing(inst, max_iterations=3000, seed=42)
        assert abs(s1.total_weighted_tardiness - s2.total_weighted_tardiness) < 1e-6

    def test_time_limit(self):
        inst = TardinessFlowShopInstance.random(n=8, m=4, seed=42)
        sol = simulated_annealing(inst, max_iterations=1000000, time_limit=0.5, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"
