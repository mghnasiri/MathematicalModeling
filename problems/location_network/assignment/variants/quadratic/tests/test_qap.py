"""Tests for Quadratic Assignment Problem."""

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


_inst = _load_mod("qap_inst_test", os.path.join(_variant_dir, "instance.py"))
QAPInstance = _inst.QAPInstance
QAPSolution = _inst.QAPSolution
validate_solution = _inst.validate_solution
small_qap_4 = _inst.small_qap_4

_heur = _load_mod("qap_heur_test", os.path.join(_variant_dir, "heuristics.py"))
greedy_construction = _heur.greedy_construction
local_search_2opt = _heur.local_search_2opt

_meta = _load_mod("qap_meta_test", os.path.join(_variant_dir, "metaheuristics.py"))
simulated_annealing = _meta.simulated_annealing


class TestQAPInstance:
    def test_random(self):
        inst = QAPInstance.random(n=6, seed=42)
        assert inst.n == 6

    def test_small(self):
        inst = small_qap_4()
        assert inst.n == 4

    def test_objective(self):
        inst = small_qap_4()
        cost = inst.objective([0, 1, 2, 3])
        assert cost > 0


class TestQAPHeuristics:
    def test_greedy_valid(self):
        inst = QAPInstance.random(n=6, seed=42)
        sol = greedy_construction(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_ls_valid(self):
        inst = QAPInstance.random(n=6, seed=42)
        sol = local_search_2opt(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_ls_improves(self):
        inst = QAPInstance.random(n=8, seed=42)
        gr = greedy_construction(inst)
        ls = local_search_2opt(inst, initial=gr)
        assert ls.cost <= gr.cost + 1e-6

    def test_small(self):
        inst = small_qap_4()
        sol = local_search_2opt(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"


class TestQAPSA:
    def test_valid(self):
        inst = QAPInstance.random(n=6, seed=42)
        sol = simulated_annealing(inst, max_iterations=5000, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_competitive(self):
        inst = QAPInstance.random(n=8, seed=42)
        gr = greedy_construction(inst)
        sa = simulated_annealing(inst, max_iterations=10000, seed=42)
        assert sa.cost <= gr.cost + 1e-6

    def test_determinism(self):
        inst = QAPInstance.random(n=6, seed=42)
        s1 = simulated_annealing(inst, max_iterations=3000, seed=42)
        s2 = simulated_annealing(inst, max_iterations=3000, seed=42)
        assert abs(s1.cost - s2.cost) < 1e-6

    def test_time_limit(self):
        inst = QAPInstance.random(n=8, seed=42)
        sol = simulated_annealing(inst, max_iterations=1000000, time_limit=0.5, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"
