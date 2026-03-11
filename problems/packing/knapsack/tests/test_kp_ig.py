"""Tests for Iterated Greedy on 0-1 Knapsack."""

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


_inst = _load_mod("kp_instance_test_ig", os.path.join(_parent_dir, "instance.py"))
KnapsackInstance = _inst.KnapsackInstance
KnapsackSolution = _inst.KnapsackSolution
validate_solution = _inst.validate_solution

_ig = _load_mod(
    "kp_ig_test",
    os.path.join(_parent_dir, "metaheuristics", "iterated_greedy.py"),
)
iterated_greedy = _ig.iterated_greedy

_gr = _load_mod(
    "kp_greedy_test_ig",
    os.path.join(_parent_dir, "heuristics", "greedy.py"),
)
greedy_value_density = _gr.greedy_value_density


class TestKPIGValidity:
    def test_within_capacity(self):
        inst = KnapsackInstance.random(n=15, seed=42)
        sol = iterated_greedy(inst, seed=42)
        assert sol.weight <= inst.capacity + 1e-10

    def test_valid_solution(self):
        inst = KnapsackInstance.random(n=15, seed=42)
        sol = iterated_greedy(inst, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Validation errors: {errors}"

    def test_no_duplicate_items(self):
        inst = KnapsackInstance.random(n=15, seed=42)
        sol = iterated_greedy(inst, seed=42)
        assert len(sol.items) == len(set(sol.items))


class TestKPIGQuality:
    def test_competitive_with_greedy(self):
        inst = KnapsackInstance.random(n=20, seed=42)
        gr_sol = greedy_value_density(inst)
        ig_sol = iterated_greedy(inst, max_iterations=1000, seed=42)
        assert ig_sol.value >= gr_sol.value - 1e-6

    def test_nonnegative_value(self):
        inst = KnapsackInstance.random(n=15, seed=42)
        sol = iterated_greedy(inst, seed=42)
        assert sol.value >= 0


class TestKPIGDeterminism:
    def test_same_seed(self):
        inst = KnapsackInstance.random(n=12, seed=42)
        sol1 = iterated_greedy(inst, max_iterations=200, seed=42)
        sol2 = iterated_greedy(inst, max_iterations=200, seed=42)
        assert abs(sol1.value - sol2.value) < 1e-6


class TestKPIGEdgeCases:
    def test_time_limit(self):
        inst = KnapsackInstance.random(n=20, seed=42)
        sol = iterated_greedy(inst, max_iterations=1000000, time_limit=0.5, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Validation errors: {errors}"
