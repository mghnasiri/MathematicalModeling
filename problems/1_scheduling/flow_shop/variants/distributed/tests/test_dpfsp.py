"""Tests for Distributed Permutation Flow Shop."""

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


_inst = _load_mod("dpfsp_inst_test", os.path.join(_variant_dir, "instance.py"))
DPFSPInstance = _inst.DPFSPInstance
DPFSPSolution = _inst.DPFSPSolution
validate_solution = _inst.validate_solution
small_dpfsp_6x3x2 = _inst.small_dpfsp_6x3x2

_heur = _load_mod("dpfsp_heur_test", os.path.join(_variant_dir, "heuristics.py"))
neh_dpfsp = _heur.neh_dpfsp
round_robin = _heur.round_robin

_meta = _load_mod("dpfsp_meta_test", os.path.join(_variant_dir, "metaheuristics.py"))
simulated_annealing = _meta.simulated_annealing


class TestDPFSPInstance:
    def test_random(self):
        inst = DPFSPInstance.random(n=8, m=3, f=2, seed=42)
        assert inst.n == 8
        assert inst.f == 2

    def test_small(self):
        inst = small_dpfsp_6x3x2()
        assert inst.n == 6
        assert inst.f == 2


class TestDPFSPHeuristics:
    def test_neh_valid(self):
        inst = DPFSPInstance.random(n=8, m=3, f=2, seed=42)
        sol = neh_dpfsp(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_rr_valid(self):
        inst = DPFSPInstance.random(n=8, m=3, f=2, seed=42)
        sol = round_robin(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_neh_beats_rr(self):
        inst = DPFSPInstance.random(n=10, m=3, f=2, seed=42)
        neh = neh_dpfsp(inst)
        rr = round_robin(inst)
        assert neh.makespan <= rr.makespan + 1e-6

    def test_small(self):
        inst = small_dpfsp_6x3x2()
        sol = neh_dpfsp(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"


class TestDPFSPSA:
    def test_valid(self):
        inst = DPFSPInstance.random(n=8, m=3, f=2, seed=42)
        sol = simulated_annealing(inst, max_iterations=5000, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_competitive(self):
        inst = DPFSPInstance.random(n=10, m=3, f=2, seed=42)
        neh = neh_dpfsp(inst)
        sa = simulated_annealing(inst, max_iterations=10000, seed=42)
        assert sa.makespan <= neh.makespan + 1e-6

    def test_determinism(self):
        inst = DPFSPInstance.random(n=6, m=3, f=2, seed=42)
        s1 = simulated_annealing(inst, max_iterations=3000, seed=42)
        s2 = simulated_annealing(inst, max_iterations=3000, seed=42)
        assert abs(s1.makespan - s2.makespan) < 1e-6

    def test_time_limit(self):
        inst = DPFSPInstance.random(n=12, m=4, f=3, seed=42)
        sol = simulated_annealing(inst, max_iterations=1000000, time_limit=0.5, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"
