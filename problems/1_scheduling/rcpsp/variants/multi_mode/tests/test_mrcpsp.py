"""Tests for Multi-Mode RCPSP."""

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


_inst = _load_mod("mrcpsp_inst_test", os.path.join(_variant_dir, "instance.py"))
MRCPSPInstance = _inst.MRCPSPInstance
MRCPSPSolution = _inst.MRCPSPSolution
validate_solution = _inst.validate_solution
small_mrcpsp_4 = _inst.small_mrcpsp_4

_heur = _load_mod("mrcpsp_heur_test", os.path.join(_variant_dir, "heuristics.py"))
serial_sgs_shortest = _heur.serial_sgs_shortest

_meta = _load_mod("mrcpsp_meta_test", os.path.join(_variant_dir, "metaheuristics.py"))
simulated_annealing = _meta.simulated_annealing


class TestMRCPSPInstance:
    def test_random(self):
        inst = MRCPSPInstance.random(n=6, seed=42)
        assert inst.n == 6

    def test_small(self):
        inst = small_mrcpsp_4()
        assert inst.n == 4
        assert inst.num_resources == 2


class TestMRCPSPHeuristics:
    def test_sgs_valid(self):
        inst = MRCPSPInstance.random(n=6, seed=42)
        sol = serial_sgs_shortest(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_small(self):
        inst = small_mrcpsp_4()
        sol = serial_sgs_shortest(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_positive_makespan(self):
        inst = MRCPSPInstance.random(n=6, seed=42)
        sol = serial_sgs_shortest(inst)
        assert sol.makespan > 0


class TestMRCPSPSA:
    def test_valid(self):
        inst = MRCPSPInstance.random(n=6, seed=42)
        sol = simulated_annealing(inst, max_iterations=5000, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_competitive(self):
        inst = MRCPSPInstance.random(n=6, seed=42)
        sgs = serial_sgs_shortest(inst)
        sa = simulated_annealing(inst, max_iterations=10000, seed=42)
        assert sa.makespan <= sgs.makespan

    def test_determinism(self):
        inst = MRCPSPInstance.random(n=6, seed=42)
        s1 = simulated_annealing(inst, max_iterations=3000, seed=42)
        s2 = simulated_annealing(inst, max_iterations=3000, seed=42)
        assert s1.makespan == s2.makespan

    def test_time_limit(self):
        inst = MRCPSPInstance.random(n=8, seed=42)
        sol = simulated_annealing(inst, max_iterations=1000000, time_limit=0.5, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"
