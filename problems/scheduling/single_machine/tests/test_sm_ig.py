"""Tests for Iterated Greedy on Single Machine Scheduling."""

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


_inst = _load_mod("sm_instance_test_ig", os.path.join(_parent_dir, "instance.py"))
SingleMachineInstance = _inst.SingleMachineInstance
SingleMachineSolution = _inst.SingleMachineSolution
compute_weighted_tardiness = _inst.compute_weighted_tardiness
compute_total_tardiness = _inst.compute_total_tardiness

_ig = _load_mod(
    "sm_ig_test",
    os.path.join(_parent_dir, "metaheuristics", "iterated_greedy.py"),
)
iterated_greedy = _ig.iterated_greedy

_atc_mod = _load_mod(
    "sm_atc_test_ig",
    os.path.join(_parent_dir, "heuristics", "apparent_tardiness_cost.py"),
)
atc = _atc_mod.atc


class TestSMIGValidity:
    """Test that IG produces valid solutions."""

    def test_random_valid(self):
        inst = SingleMachineInstance.random(n=10, seed=42)
        sol = iterated_greedy(inst, seed=42)
        assert set(sol.sequence) == set(range(inst.n))

    def test_weighted_tardiness_matches(self):
        inst = SingleMachineInstance.random(n=10, seed=42)
        sol = iterated_greedy(inst, objective="weighted_tardiness", seed=42)
        actual = compute_weighted_tardiness(inst, sol.sequence)
        assert abs(sol.objective_value - actual) < 1e-6

    def test_total_tardiness_matches(self):
        inst = SingleMachineInstance.random(n=10, seed=42)
        sol = iterated_greedy(inst, objective="total_tardiness", seed=42)
        actual = compute_total_tardiness(inst, sol.sequence)
        assert abs(sol.objective_value - actual) < 1e-6


class TestSMIGQuality:
    """Test solution quality."""

    def test_ig_improves_or_matches_atc(self):
        inst = SingleMachineInstance.random(n=10, seed=42)
        atc_sol = atc(inst)
        ig_sol = iterated_greedy(inst, objective="weighted_tardiness", seed=42)
        assert ig_sol.objective_value <= atc_sol.objective_value + 1e-6


class TestSMIGDeterminism:
    """Test deterministic behavior with seed."""

    def test_same_seed_same_result(self):
        inst = SingleMachineInstance.random(n=10, seed=42)
        sol1 = iterated_greedy(inst, max_iterations=200, seed=42)
        sol2 = iterated_greedy(inst, max_iterations=200, seed=42)
        assert abs(sol1.objective_value - sol2.objective_value) < 1e-6

    def test_different_seed_both_valid(self):
        inst = SingleMachineInstance.random(n=8, seed=42)
        sol1 = iterated_greedy(inst, seed=1)
        sol2 = iterated_greedy(inst, seed=999)
        assert set(sol1.sequence) == set(range(inst.n))
        assert set(sol2.sequence) == set(range(inst.n))


class TestSMIGEdgeCases:
    """Test edge cases."""

    def test_single_job(self):
        inst = SingleMachineInstance.from_arrays(
            processing_times=[5], weights=[1], due_dates=[3],
        )
        sol = iterated_greedy(inst, seed=42)
        assert sol.sequence == [0]

    def test_time_limit(self):
        inst = SingleMachineInstance.random(n=15, seed=42)
        sol = iterated_greedy(
            inst, max_iterations=1000000, time_limit=0.5, seed=42
        )
        assert set(sol.sequence) == set(range(inst.n))
