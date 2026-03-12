"""Tests for Unrelated Parallel Machine with Tardiness."""

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


_inst = _load_mod("rm_tard_inst_test", os.path.join(_variant_dir, "instance.py"))
RmTardinessInstance = _inst.RmTardinessInstance
RmTardinessSolution = _inst.RmTardinessSolution
validate_solution = _inst.validate_solution
small_rm_tard_6x2 = _inst.small_rm_tard_6x2

_heur = _load_mod("rm_tard_heur_test", os.path.join(_variant_dir, "heuristics.py"))
edd_ect = _heur.edd_ect
atc_dispatch = _heur.atc_dispatch

_meta = _load_mod("rm_tard_meta_test", os.path.join(_variant_dir, "metaheuristics.py"))
simulated_annealing = _meta.simulated_annealing


class TestRmTardInstance:
    def test_random(self):
        inst = RmTardinessInstance.random(n=8, m=3, seed=42)
        assert inst.n == 8
        assert inst.m == 3

    def test_small(self):
        inst = small_rm_tard_6x2()
        assert inst.n == 6
        assert inst.m == 2


class TestRmTardHeuristics:
    def test_edd_valid(self):
        inst = small_rm_tard_6x2()
        sol = edd_ect(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_atc_valid(self):
        inst = small_rm_tard_6x2()
        sol = atc_dispatch(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_edd_random(self):
        inst = RmTardinessInstance.random(n=10, m=3, seed=42)
        sol = edd_ect(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"


class TestRmTardSA:
    def test_valid(self):
        inst = small_rm_tard_6x2()
        sol = simulated_annealing(inst, max_iterations=5000, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_competitive(self):
        inst = RmTardinessInstance.random(n=8, m=3, seed=42)
        edd = edd_ect(inst)
        sa = simulated_annealing(inst, max_iterations=10000, seed=42)
        assert sa.total_tardiness <= edd.total_tardiness + 1e-6

    def test_determinism(self):
        inst = small_rm_tard_6x2()
        s1 = simulated_annealing(inst, max_iterations=3000, seed=42)
        s2 = simulated_annealing(inst, max_iterations=3000, seed=42)
        assert abs(s1.total_tardiness - s2.total_tardiness) < 1e-6

    def test_time_limit(self):
        inst = RmTardinessInstance.random(n=10, m=4, seed=42)
        sol = simulated_annealing(inst, max_iterations=1000000,
                                   time_limit=0.5, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"
