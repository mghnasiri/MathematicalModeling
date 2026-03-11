"""Tests for Iterated Greedy on p-Median."""

from __future__ import annotations

import sys
import os
import importlib.util

import numpy as np
import pytest

_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_mod(name, filepath):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_inst = _load_mod("pmed_instance_test_ig", os.path.join(_parent_dir, "instance.py"))
PMedianInstance = _inst.PMedianInstance
validate_solution = _inst.validate_solution

_ig = _load_mod(
    "pmed_ig_test",
    os.path.join(_parent_dir, "metaheuristics", "iterated_greedy.py"),
)
iterated_greedy = _ig.iterated_greedy

_gr = _load_mod(
    "pmed_gr_test_ig",
    os.path.join(_parent_dir, "heuristics", "greedy_pmedian.py"),
)
greedy_pmedian = _gr.greedy_pmedian


class TestPMedIGValidity:
    def test_valid_solution(self):
        inst = PMedianInstance.random(n=10, m=10, p=3, seed=42)
        sol = iterated_greedy(inst, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Validation errors: {errors}"

    def test_correct_p(self):
        inst = PMedianInstance.random(n=10, m=10, p=3, seed=42)
        sol = iterated_greedy(inst, seed=42)
        assert len(sol.open_facilities) == inst.p


class TestPMedIGQuality:
    def test_competitive_with_greedy(self):
        inst = PMedianInstance.random(n=15, m=15, p=3, seed=42)
        gr_sol = greedy_pmedian(inst)
        ig_sol = iterated_greedy(inst, max_iterations=500, seed=42)
        assert ig_sol.cost <= gr_sol.cost + 1e-6


class TestPMedIGDeterminism:
    def test_same_seed(self):
        inst = PMedianInstance.random(n=10, m=10, p=3, seed=42)
        sol1 = iterated_greedy(inst, max_iterations=200, seed=42)
        sol2 = iterated_greedy(inst, max_iterations=200, seed=42)
        assert abs(sol1.cost - sol2.cost) < 1e-6


class TestPMedIGEdgeCases:
    def test_p_equals_m(self):
        inst = PMedianInstance.random(n=8, m=5, p=5, seed=42)
        sol = iterated_greedy(inst, seed=42)
        assert len(sol.open_facilities) == 5

    def test_time_limit(self):
        inst = PMedianInstance.random(n=12, m=12, p=3, seed=42)
        sol = iterated_greedy(inst, max_iterations=1000000, time_limit=0.5, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Validation errors: {errors}"
