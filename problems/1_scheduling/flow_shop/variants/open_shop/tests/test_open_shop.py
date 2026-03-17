"""Tests for Open Shop Scheduling."""

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


_inst = _load_mod("os_inst_test", os.path.join(_variant_dir, "instance.py"))
OpenShopInstance = _inst.OpenShopInstance
OpenShopSolution = _inst.OpenShopSolution
validate_solution = _inst.validate_solution
small_os_3x3 = _inst.small_os_3x3

_heur = _load_mod("os_heur_test", os.path.join(_variant_dir, "heuristics.py"))
lpt_open_shop = _heur.lpt_open_shop
greedy_open_shop = _heur.greedy_open_shop

_meta = _load_mod("os_meta_test", os.path.join(_variant_dir, "metaheuristics.py"))
simulated_annealing = _meta.simulated_annealing


class TestOpenShopInstance:
    def test_random(self):
        inst = OpenShopInstance.random(n=5, m=3, seed=42)
        assert inst.n == 5
        assert inst.m == 3

    def test_small(self):
        inst = small_os_3x3()
        assert inst.n == 3
        assert inst.m == 3


class TestOpenShopHeuristics:
    def test_lpt_valid(self):
        inst = small_os_3x3()
        sol = lpt_open_shop(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_greedy_valid(self):
        inst = small_os_3x3()
        sol = greedy_open_shop(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_lpt_random(self):
        inst = OpenShopInstance.random(n=6, m=4, seed=42)
        sol = lpt_open_shop(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_greedy_random(self):
        inst = OpenShopInstance.random(n=6, m=4, seed=42)
        sol = greedy_open_shop(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"


class TestOpenShopSA:
    def test_valid(self):
        inst = small_os_3x3()
        sol = simulated_annealing(inst, max_iterations=5000, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_competitive(self):
        inst = OpenShopInstance.random(n=5, m=3, seed=42)
        greedy = greedy_open_shop(inst)
        sa = simulated_annealing(inst, max_iterations=10000, seed=42)
        assert sa.makespan <= greedy.makespan + 1e-6

    def test_determinism(self):
        inst = small_os_3x3()
        s1 = simulated_annealing(inst, max_iterations=3000, seed=42)
        s2 = simulated_annealing(inst, max_iterations=3000, seed=42)
        assert abs(s1.makespan - s2.makespan) < 1e-6

    def test_time_limit(self):
        inst = OpenShopInstance.random(n=6, m=4, seed=42)
        sol = simulated_annealing(inst, max_iterations=1000000,
                                   time_limit=0.5, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"
