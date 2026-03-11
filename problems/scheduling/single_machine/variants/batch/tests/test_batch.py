"""Tests for Batch Single Machine Scheduling."""

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


_inst = _load_mod("batch_inst_test", os.path.join(_variant_dir, "instance.py"))
BatchSMInstance = _inst.BatchSMInstance
BatchSMSolution = _inst.BatchSMSolution
validate_solution = _inst.validate_solution
small_batch_6 = _inst.small_batch_6

_heur = _load_mod("batch_heur_test", os.path.join(_variant_dir, "heuristics.py"))
wspt_single = _heur.wspt_single
greedy_batching = _heur.greedy_batching

_meta = _load_mod("batch_meta_test", os.path.join(_variant_dir, "metaheuristics.py"))
simulated_annealing = _meta.simulated_annealing


class TestBatchInstance:
    def test_random(self):
        inst = BatchSMInstance.random(n=6, seed=42)
        assert inst.n == 6

    def test_small(self):
        inst = small_batch_6()
        assert inst.n == 6
        assert inst.setup_time == 3.0

    def test_evaluate(self):
        inst = small_batch_6()
        obj = inst.evaluate([[0, 1, 2], [3, 4, 5]])
        assert obj > 0


class TestBatchHeuristics:
    def test_wspt_valid(self):
        inst = BatchSMInstance.random(n=6, seed=42)
        sol = wspt_single(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_greedy_valid(self):
        inst = BatchSMInstance.random(n=6, seed=42)
        sol = greedy_batching(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_small(self):
        inst = small_batch_6()
        sol = wspt_single(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"


class TestBatchSA:
    def test_valid(self):
        inst = BatchSMInstance.random(n=6, seed=42)
        sol = simulated_annealing(inst, max_iterations=5000, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_competitive(self):
        inst = BatchSMInstance.random(n=8, seed=42)
        wspt = wspt_single(inst)
        sa = simulated_annealing(inst, max_iterations=10000, seed=42)
        assert sa.objective <= wspt.objective + 1e-6

    def test_determinism(self):
        inst = BatchSMInstance.random(n=6, seed=42)
        s1 = simulated_annealing(inst, max_iterations=3000, seed=42)
        s2 = simulated_annealing(inst, max_iterations=3000, seed=42)
        assert abs(s1.objective - s2.objective) < 1e-6

    def test_time_limit(self):
        inst = BatchSMInstance.random(n=10, seed=42)
        sol = simulated_annealing(inst, max_iterations=1000000, time_limit=0.5, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"
