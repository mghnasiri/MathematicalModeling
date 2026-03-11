"""Tests for Online Bin Packing."""

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


_inst = _load_mod("onlinebp_inst_test", os.path.join(_variant_dir, "instance.py"))
OnlineBPInstance = _inst.OnlineBPInstance
OnlineBPSolution = _inst.OnlineBPSolution
validate_solution = _inst.validate_solution
small_online_8 = _inst.small_online_8

_heur = _load_mod("onlinebp_heur_test", os.path.join(_variant_dir, "heuristics.py"))
next_fit = _heur.next_fit
first_fit = _heur.first_fit
best_fit = _heur.best_fit


class TestOnlineBPInstance:
    def test_random(self):
        inst = OnlineBPInstance.random(n=15, seed=42)
        assert inst.n == 15

    def test_small(self):
        inst = small_online_8()
        assert inst.n == 8


class TestNextFit:
    def test_valid(self):
        inst = OnlineBPInstance.random(n=15, seed=42)
        sol = next_fit(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_small(self):
        inst = small_online_8()
        sol = next_fit(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"


class TestFirstFit:
    def test_valid(self):
        inst = OnlineBPInstance.random(n=15, seed=42)
        sol = first_fit(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_better_than_nf(self):
        inst = OnlineBPInstance.random(n=20, seed=42)
        nf = next_fit(inst)
        ff = first_fit(inst)
        assert ff.num_bins <= nf.num_bins

    def test_small(self):
        inst = small_online_8()
        sol = first_fit(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"


class TestBestFit:
    def test_valid(self):
        inst = OnlineBPInstance.random(n=15, seed=42)
        sol = best_fit(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_better_than_nf(self):
        inst = OnlineBPInstance.random(n=20, seed=42)
        nf = next_fit(inst)
        bf = best_fit(inst)
        assert bf.num_bins <= nf.num_bins

    def test_small(self):
        inst = small_online_8()
        sol = best_fit(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"
