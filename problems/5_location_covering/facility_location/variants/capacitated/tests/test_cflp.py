"""Tests for Capacitated Facility Location Problem."""

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


_inst = _load_mod("cflp_instance_test", os.path.join(_variant_dir, "instance.py"))
CFLPInstance = _inst.CFLPInstance
CFLPSolution = _inst.CFLPSolution
validate_solution = _inst.validate_solution
small_cflp_3_5 = _inst.small_cflp_3_5

_heur = _load_mod("cflp_heuristics_test", os.path.join(_variant_dir, "heuristics.py"))
greedy_add = _heur.greedy_add

_meta = _load_mod("cflp_meta_test", os.path.join(_variant_dir, "metaheuristics.py"))
simulated_annealing = _meta.simulated_annealing


class TestCFLPInstance:
    def test_random_creation(self):
        inst = CFLPInstance.random(m=5, n=10, seed=42)
        assert inst.m == 5
        assert inst.n == 10
        assert inst.capacities.shape == (5,)
        assert inst.demands.shape == (10,)

    def test_small_benchmark(self):
        inst = small_cflp_3_5()
        assert inst.m == 3
        assert inst.n == 5
        assert inst.demands.sum() == 36.0

    def test_capacity_sufficient(self):
        inst = CFLPInstance.random(m=5, n=10, seed=42)
        assert inst.capacities.sum() >= inst.demands.sum()


class TestCFLPGreedy:
    def test_valid_solution(self):
        inst = CFLPInstance.random(m=6, n=12, seed=42)
        sol = greedy_add(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_all_assigned(self):
        inst = CFLPInstance.random(m=6, n=12, seed=42)
        sol = greedy_add(inst)
        assert len(sol.assignments) == inst.n

    def test_positive_cost(self):
        inst = CFLPInstance.random(m=6, n=12, seed=42)
        sol = greedy_add(inst)
        assert sol.cost > 0

    def test_small_benchmark(self):
        inst = small_cflp_3_5()
        sol = greedy_add(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"


class TestCFLPSA:
    def test_valid(self):
        inst = CFLPInstance.random(m=6, n=12, seed=42)
        sol = simulated_annealing(inst, max_iterations=5000, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_competitive(self):
        inst = CFLPInstance.random(m=6, n=12, seed=42)
        gr_sol = greedy_add(inst)
        sa_sol = simulated_annealing(inst, max_iterations=10000, seed=42)
        assert sa_sol.cost <= gr_sol.cost + 1e-6

    def test_determinism(self):
        inst = CFLPInstance.random(m=5, n=10, seed=42)
        sol1 = simulated_annealing(inst, max_iterations=3000, seed=42)
        sol2 = simulated_annealing(inst, max_iterations=3000, seed=42)
        assert abs(sol1.cost - sol2.cost) < 1e-6

    def test_time_limit(self):
        inst = CFLPInstance.random(m=8, n=15, seed=42)
        sol = simulated_annealing(inst, max_iterations=1000000, time_limit=0.5, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"
