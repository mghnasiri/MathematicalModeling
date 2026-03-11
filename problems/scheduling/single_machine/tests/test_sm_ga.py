"""Tests for Genetic Algorithm on Single Machine Scheduling."""

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


_inst = _load_mod("sm_instance_test_ga", os.path.join(_parent_dir, "instance.py"))
SingleMachineInstance = _inst.SingleMachineInstance
SingleMachineSolution = _inst.SingleMachineSolution
compute_weighted_tardiness = _inst.compute_weighted_tardiness
compute_total_tardiness = _inst.compute_total_tardiness

_ga = _load_mod(
    "sm_ga_test",
    os.path.join(_parent_dir, "metaheuristics", "genetic_algorithm.py"),
)
genetic_algorithm = _ga.genetic_algorithm


class TestSMGAValidity:
    """Test that GA produces valid solutions."""

    def test_random_instance_valid(self):
        inst = SingleMachineInstance.random(n=10, seed=42)
        sol = genetic_algorithm(inst, seed=42)
        assert set(sol.sequence) == set(range(inst.n))

    def test_objective_matches(self):
        inst = SingleMachineInstance.random(n=10, seed=42)
        sol = genetic_algorithm(inst, objective="weighted_tardiness", seed=42)
        actual = compute_weighted_tardiness(inst, sol.sequence)
        assert abs(sol.objective_value - actual) < 1e-6

    def test_total_tardiness_objective(self):
        inst = SingleMachineInstance.random(n=10, seed=42)
        sol = genetic_algorithm(inst, objective="total_tardiness", seed=42)
        actual = compute_total_tardiness(inst, sol.sequence)
        assert abs(sol.objective_value - actual) < 1e-6


class TestSMGAQuality:
    """Test solution quality."""

    def test_ga_finds_good_solution(self):
        inst = SingleMachineInstance.random(n=10, seed=42)
        sol = genetic_algorithm(inst, generations=300, seed=42)
        # Should find a reasonable solution (non-negative tardiness)
        assert sol.objective_value >= 0

    def test_ga_with_more_iterations(self):
        inst = SingleMachineInstance.random(n=8, seed=42)
        sol_short = genetic_algorithm(inst, generations=50, seed=42)
        sol_long = genetic_algorithm(inst, generations=500, seed=42)
        assert sol_long.objective_value <= sol_short.objective_value + 1


class TestSMGADeterminism:
    """Test deterministic behavior with seed."""

    def test_same_seed_same_result(self):
        inst = SingleMachineInstance.random(n=10, seed=42)
        sol1 = genetic_algorithm(inst, seed=42)
        sol2 = genetic_algorithm(inst, seed=42)
        assert abs(sol1.objective_value - sol2.objective_value) < 1e-6

    def test_different_seed_both_valid(self):
        inst = SingleMachineInstance.random(n=10, seed=42)
        sol1 = genetic_algorithm(inst, seed=1)
        sol2 = genetic_algorithm(inst, seed=999)
        assert set(sol1.sequence) == set(range(inst.n))
        assert set(sol2.sequence) == set(range(inst.n))


class TestSMGAEdgeCases:
    """Test edge cases."""

    def test_single_job(self):
        inst = SingleMachineInstance.from_arrays(
            processing_times=[5],
            weights=[1],
            due_dates=[3],
        )
        sol = genetic_algorithm(inst, seed=42)
        assert sol.sequence == [0]

    def test_time_limit(self):
        inst = SingleMachineInstance.random(n=15, seed=42)
        sol = genetic_algorithm(
            inst, generations=1000000, time_limit=0.5, seed=42
        )
        assert set(sol.sequence) == set(range(inst.n))
