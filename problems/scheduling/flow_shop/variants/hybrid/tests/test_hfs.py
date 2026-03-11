"""Tests for Hybrid Flow Shop."""

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


_inst = _load_mod("hfs_inst_test", os.path.join(_variant_dir, "instance.py"))
HFSInstance = _inst.HFSInstance
HFSSolution = _inst.HFSSolution
validate_solution = _inst.validate_solution
small_hfs_4x3 = _inst.small_hfs_4x3

_heur = _load_mod("hfs_heur_test", os.path.join(_variant_dir, "heuristics.py"))
neh_hfs = _heur.neh_hfs
lpt_hfs = _heur.lpt_hfs
spt_hfs = _heur.spt_hfs

_meta = _load_mod("hfs_meta_test", os.path.join(_variant_dir, "metaheuristics.py"))
simulated_annealing = _meta.simulated_annealing


class TestHFSInstance:
    def test_random(self):
        inst = HFSInstance.random(n=6, stages=3, seed=42)
        assert inst.n == 6
        assert inst.stages == 3

    def test_small(self):
        inst = small_hfs_4x3()
        assert inst.n == 4
        assert inst.stages == 3

    def test_makespan(self):
        inst = small_hfs_4x3()
        ms = inst.makespan([0, 1, 2, 3])
        assert ms > 0


class TestHFSHeuristics:
    def test_neh_valid(self):
        inst = HFSInstance.random(n=6, stages=3, seed=42)
        sol = neh_hfs(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_lpt_valid(self):
        inst = HFSInstance.random(n=6, stages=3, seed=42)
        sol = lpt_hfs(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_neh_beats_lpt(self):
        inst = HFSInstance.random(n=10, stages=3, seed=42)
        neh = neh_hfs(inst)
        lpt = lpt_hfs(inst)
        assert neh.makespan <= lpt.makespan + 1e-6

    def test_small(self):
        inst = small_hfs_4x3()
        sol = neh_hfs(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"


class TestHFSSA:
    def test_valid(self):
        inst = HFSInstance.random(n=6, stages=3, seed=42)
        sol = simulated_annealing(inst, max_iterations=5000, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_competitive(self):
        inst = HFSInstance.random(n=8, stages=3, seed=42)
        neh = neh_hfs(inst)
        sa = simulated_annealing(inst, max_iterations=10000, seed=42)
        assert sa.makespan <= neh.makespan + 1e-6

    def test_determinism(self):
        inst = HFSInstance.random(n=6, stages=3, seed=42)
        s1 = simulated_annealing(inst, max_iterations=3000, seed=42)
        s2 = simulated_annealing(inst, max_iterations=3000, seed=42)
        assert abs(s1.makespan - s2.makespan) < 1e-6

    def test_time_limit(self):
        inst = HFSInstance.random(n=10, stages=4, seed=42)
        sol = simulated_annealing(inst, max_iterations=1000000, time_limit=0.5, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"
