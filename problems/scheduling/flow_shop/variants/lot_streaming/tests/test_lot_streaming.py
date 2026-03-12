"""Tests for Lot Streaming Flow Shop."""

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


_inst = _load_mod("lotstream_inst_test", os.path.join(_variant_dir, "instance.py"))
LotStreamInstance = _inst.LotStreamInstance
LotStreamSolution = _inst.LotStreamSolution
validate_solution = _inst.validate_solution
small_ls_4x3 = _inst.small_ls_4x3

_heur = _load_mod("lotstream_heur_test", os.path.join(_variant_dir, "heuristics.py"))
neh_ls = _heur.neh_ls
lpt_ls = _heur.lpt_ls

_meta = _load_mod("lotstream_meta_test", os.path.join(_variant_dir, "metaheuristics.py"))
simulated_annealing = _meta.simulated_annealing


class TestLotStreamInstance:
    def test_random(self):
        inst = LotStreamInstance.random(n=6, m=3, seed=42)
        assert inst.n == 6
        assert inst.m == 3

    def test_small(self):
        inst = small_ls_4x3()
        assert inst.n == 4
        assert inst.num_sublots == 3

    def test_streaming_reduces(self):
        """Lot streaming should reduce or equal no-streaming makespan."""
        inst = small_ls_4x3()
        perm = list(range(inst.n))
        ms_ns = inst.makespan_no_streaming(perm)
        ms_s = inst.makespan_streaming(perm)
        assert ms_s <= ms_ns + 1e-6


class TestLotStreamHeuristics:
    def test_neh_valid(self):
        inst = LotStreamInstance.random(n=6, m=3, seed=42)
        sol = neh_ls(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_lpt_valid(self):
        inst = LotStreamInstance.random(n=6, m=3, seed=42)
        sol = lpt_ls(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_neh_beats_lpt(self):
        inst = LotStreamInstance.random(n=8, m=3, seed=42)
        neh = neh_ls(inst)
        lpt = lpt_ls(inst)
        assert neh.makespan <= lpt.makespan + 1e-6

    def test_small(self):
        inst = small_ls_4x3()
        sol = neh_ls(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"


class TestLotStreamSA:
    def test_valid(self):
        inst = LotStreamInstance.random(n=6, m=3, seed=42)
        sol = simulated_annealing(inst, max_iterations=5000, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_competitive(self):
        inst = LotStreamInstance.random(n=8, m=3, seed=42)
        neh = neh_ls(inst)
        sa = simulated_annealing(inst, max_iterations=10000, seed=42)
        assert sa.makespan <= neh.makespan + 1e-6

    def test_determinism(self):
        inst = LotStreamInstance.random(n=6, m=3, seed=42)
        s1 = simulated_annealing(inst, max_iterations=3000, seed=42)
        s2 = simulated_annealing(inst, max_iterations=3000, seed=42)
        assert abs(s1.makespan - s2.makespan) < 1e-6

    def test_time_limit(self):
        inst = LotStreamInstance.random(n=8, m=4, seed=42)
        sol = simulated_annealing(inst, max_iterations=1000000, time_limit=0.5, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"
