"""Tests for Two-Dimensional Cutting Stock Problem."""

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


_inst = _load_mod("twod_csp_inst_test", os.path.join(_variant_dir, "instance.py"))
TwoDCSPInstance = _inst.TwoDCSPInstance
TwoDCSPSolution = _inst.TwoDCSPSolution
validate_solution = _inst.validate_solution
small_2dcsp_4 = _inst.small_2dcsp_4

_heur = _load_mod("twod_csp_heur_test", os.path.join(_variant_dir, "heuristics.py"))
bottom_left_ffd = _heur.bottom_left_ffd
shelf_nfdh = _heur.shelf_nfdh

_meta = _load_mod("twod_csp_meta_test", os.path.join(_variant_dir, "metaheuristics.py"))
simulated_annealing = _meta.simulated_annealing


class TestTwoDCSPInstance:
    def test_random(self):
        inst = TwoDCSPInstance.random(num_types=5, seed=42)
        assert inst.num_types == 5

    def test_small(self):
        inst = small_2dcsp_4()
        assert inst.num_types == 4

    def test_item_fits(self):
        inst = small_2dcsp_4()
        assert inst.item_fits(30, 20)
        assert not inst.item_fits(150, 150)


class TestTwoDCSPHeuristics:
    def test_bl_ffd_valid(self):
        inst = small_2dcsp_4()
        sol = bottom_left_ffd(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_shelf_valid(self):
        inst = small_2dcsp_4()
        sol = shelf_nfdh(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_bl_random(self):
        inst = TwoDCSPInstance.random(num_types=5, seed=42)
        sol = bottom_left_ffd(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_shelf_random(self):
        inst = TwoDCSPInstance.random(num_types=5, seed=42)
        sol = shelf_nfdh(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"


class TestTwoDCSPSA:
    def test_valid(self):
        inst = small_2dcsp_4()
        sol = simulated_annealing(inst, max_iterations=5000, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_determinism(self):
        inst = small_2dcsp_4()
        s1 = simulated_annealing(inst, max_iterations=3000, seed=42)
        s2 = simulated_annealing(inst, max_iterations=3000, seed=42)
        assert s1.num_sheets == s2.num_sheets

    def test_time_limit(self):
        inst = TwoDCSPInstance.random(num_types=6, seed=42)
        sol = simulated_annealing(inst, max_iterations=1000000,
                                   time_limit=0.5, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"
