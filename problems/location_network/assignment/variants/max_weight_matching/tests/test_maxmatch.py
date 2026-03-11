"""Tests for Maximum Weight Matching."""

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


_inst = _load_mod("maxmatch_inst_test", os.path.join(_variant_dir, "instance.py"))
MaxMatchingInstance = _inst.MaxMatchingInstance
MaxMatchingSolution = _inst.MaxMatchingSolution
validate_solution = _inst.validate_solution
small_matching_4x5 = _inst.small_matching_4x5

_heur = _load_mod("maxmatch_heur_test", os.path.join(_variant_dir, "heuristics.py"))
greedy_matching = _heur.greedy_matching
hungarian_max = _heur.hungarian_max


class TestMaxMatchInstance:
    def test_random(self):
        inst = MaxMatchingInstance.random(n_workers=5, n_tasks=7, seed=42)
        assert inst.n_workers == 5
        assert inst.n_tasks == 7

    def test_small(self):
        inst = small_matching_4x5()
        assert inst.n_workers == 4
        assert inst.n_tasks == 5


class TestMaxMatchHeuristics:
    def test_greedy_valid(self):
        inst = MaxMatchingInstance.random(seed=42)
        sol = greedy_matching(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_hungarian_valid(self):
        inst = MaxMatchingInstance.random(seed=42)
        sol = hungarian_max(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_hungarian_optimal(self):
        inst = MaxMatchingInstance.random(seed=42)
        gr = greedy_matching(inst)
        hu = hungarian_max(inst)
        assert hu.total_weight >= gr.total_weight - 1e-6

    def test_small_greedy(self):
        inst = small_matching_4x5()
        sol = greedy_matching(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_small_hungarian(self):
        inst = small_matching_4x5()
        sol = hungarian_max(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_positive_weight(self):
        inst = MaxMatchingInstance.random(seed=42)
        sol = hungarian_max(inst)
        assert sol.total_weight > 0
