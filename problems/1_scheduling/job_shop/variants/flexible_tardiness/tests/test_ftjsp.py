"""Tests for Flexible Job Shop with Tardiness."""

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


_inst = _load_mod("ftjsp_inst_test", os.path.join(_variant_dir, "instance.py"))
FlexTardJSPInstance = _inst.FlexTardJSPInstance
FlexTardJSPSolution = _inst.FlexTardJSPSolution
validate_solution = _inst.validate_solution
small_ftjsp_3x3 = _inst.small_ftjsp_3x3

_heur = _load_mod("ftjsp_heur_test", os.path.join(_variant_dir, "heuristics.py"))
edd_ect = _heur.edd_ect
watc_dispatch = _heur.watc_dispatch

_meta = _load_mod("ftjsp_meta_test", os.path.join(_variant_dir, "metaheuristics.py"))
simulated_annealing = _meta.simulated_annealing


class TestFlexTardJSPInstance:
    def test_random(self):
        inst = FlexTardJSPInstance.random(n=4, m=3, seed=42)
        assert inst.n == 4
        assert inst.m == 3

    def test_small(self):
        inst = small_ftjsp_3x3()
        assert inst.n == 3
        assert inst.m == 3

    def test_total_ops(self):
        inst = small_ftjsp_3x3()
        assert inst.total_operations() == 6


class TestFlexTardJSPHeuristics:
    def test_edd_valid(self):
        inst = small_ftjsp_3x3()
        sol = edd_ect(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_watc_valid(self):
        inst = small_ftjsp_3x3()
        sol = watc_dispatch(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_edd_random(self):
        inst = FlexTardJSPInstance.random(n=5, m=3, seed=42)
        sol = edd_ect(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_watc_random(self):
        inst = FlexTardJSPInstance.random(n=5, m=3, seed=42)
        sol = watc_dispatch(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"


class TestFlexTardJSPSA:
    def test_valid(self):
        inst = small_ftjsp_3x3()
        sol = simulated_annealing(inst, max_iterations=5000, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_competitive(self):
        inst = FlexTardJSPInstance.random(n=4, m=3, seed=42)
        edd = edd_ect(inst)
        sa = simulated_annealing(inst, max_iterations=10000, seed=42)
        assert sa.total_weighted_tardiness <= edd.total_weighted_tardiness + 1e-6

    def test_determinism(self):
        inst = small_ftjsp_3x3()
        s1 = simulated_annealing(inst, max_iterations=3000, seed=42)
        s2 = simulated_annealing(inst, max_iterations=3000, seed=42)
        assert abs(s1.total_weighted_tardiness - s2.total_weighted_tardiness) < 1e-6

    def test_time_limit(self):
        inst = FlexTardJSPInstance.random(n=5, m=4, seed=42)
        sol = simulated_annealing(inst, max_iterations=1000000,
                                   time_limit=0.5, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"
