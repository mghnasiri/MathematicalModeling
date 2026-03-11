"""Tests for 2D Bin Packing (Strip Packing)."""

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


_inst = _load_mod("bpp2d_inst_test", os.path.join(_variant_dir, "instance.py"))
BPP2DInstance = _inst.BPP2DInstance
BPP2DSolution = _inst.BPP2DSolution
validate_solution = _inst.validate_solution
small_2dbpp_5 = _inst.small_2dbpp_5

_heur = _load_mod("bpp2d_heur_test", os.path.join(_variant_dir, "heuristics.py"))
bottom_left_dh = _heur.bottom_left_dh
nfdh = _heur.nfdh

_meta = _load_mod("bpp2d_meta_test", os.path.join(_variant_dir, "metaheuristics.py"))
simulated_annealing = _meta.simulated_annealing


class TestBPP2DInstance:
    def test_random_creation(self):
        inst = BPP2DInstance.random(n=8, seed=42)
        assert inst.n == 8
        assert inst.strip_width == 100.0

    def test_small_benchmark(self):
        inst = small_2dbpp_5()
        assert inst.n == 5

    def test_area_lower_bound(self):
        inst = small_2dbpp_5()
        lb = inst.area_lower_bound()
        assert lb > 0


class TestBPP2DHeuristics:
    def test_bldh_valid(self):
        inst = BPP2DInstance.random(n=8, seed=42)
        sol = bottom_left_dh(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_nfdh_valid(self):
        inst = BPP2DInstance.random(n=8, seed=42)
        sol = nfdh(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_respects_lower_bound(self):
        inst = BPP2DInstance.random(n=8, seed=42)
        sol = bottom_left_dh(inst)
        assert sol.height >= inst.area_lower_bound() - 1e-10

    def test_small_benchmark(self):
        inst = small_2dbpp_5()
        sol = bottom_left_dh(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"


class TestBPP2DSA:
    def test_valid(self):
        inst = BPP2DInstance.random(n=8, seed=42)
        sol = simulated_annealing(inst, max_iterations=5000, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_competitive(self):
        inst = BPP2DInstance.random(n=8, seed=42)
        bldh_sol = bottom_left_dh(inst)
        sa_sol = simulated_annealing(inst, max_iterations=10000, seed=42)
        assert sa_sol.height <= bldh_sol.height + 1e-6

    def test_determinism(self):
        inst = BPP2DInstance.random(n=6, seed=42)
        sol1 = simulated_annealing(inst, max_iterations=3000, seed=42)
        sol2 = simulated_annealing(inst, max_iterations=3000, seed=42)
        assert abs(sol1.height - sol2.height) < 1e-6

    def test_time_limit(self):
        inst = BPP2DInstance.random(n=10, seed=42)
        sol = simulated_annealing(inst, max_iterations=1000000, time_limit=0.5, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"
