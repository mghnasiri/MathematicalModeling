"""Tests for Iterated Greedy on Parallel Machine Scheduling."""

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


_inst = _load_mod("pm_instance_test_ig", os.path.join(_parent_dir, "instance.py"))
ParallelMachineInstance = _inst.ParallelMachineInstance
ParallelMachineSolution = _inst.ParallelMachineSolution
compute_makespan = _inst.compute_makespan

_ig = _load_mod(
    "pm_ig_test",
    os.path.join(_parent_dir, "metaheuristics", "iterated_greedy.py"),
)
iterated_greedy = _ig.iterated_greedy

_lpt_mod = _load_mod(
    "pm_lpt_test_ig",
    os.path.join(_parent_dir, "heuristics", "lpt.py"),
)
lpt = _lpt_mod.lpt


class TestPMIGValidity:
    """Test that IG produces valid solutions."""

    def test_identical_valid(self):
        inst = ParallelMachineInstance.random_identical(n=15, m=3, seed=42)
        sol = iterated_greedy(inst, seed=42)
        all_jobs = sorted(j for jobs in sol.assignment for j in jobs)
        assert all_jobs == list(range(inst.n))

    def test_uniform_valid(self):
        inst = ParallelMachineInstance.random_uniform(n=12, m=3, seed=42)
        sol = iterated_greedy(inst, seed=42)
        all_jobs = sorted(j for jobs in sol.assignment for j in jobs)
        assert all_jobs == list(range(inst.n))

    def test_makespan_matches(self):
        inst = ParallelMachineInstance.random_identical(n=10, m=3, seed=42)
        sol = iterated_greedy(inst, seed=42)
        actual = compute_makespan(inst, sol.assignment)
        assert abs(sol.makespan - actual) < 1e-6


class TestPMIGQuality:
    """Test solution quality."""

    def test_ig_competitive_with_lpt(self):
        inst = ParallelMachineInstance.random_identical(n=20, m=4, seed=42)
        lpt_sol = lpt(inst)
        ig_sol = iterated_greedy(inst, max_iterations=1000, seed=42)
        assert ig_sol.makespan <= lpt_sol.makespan + 1e-6

    def test_unrelated_quality(self):
        inst = ParallelMachineInstance.random_unrelated(n=15, m=3, seed=42)
        lpt_sol = lpt(inst)
        ig_sol = iterated_greedy(inst, max_iterations=1000, seed=42)
        assert ig_sol.makespan <= lpt_sol.makespan * 1.1


class TestPMIGDeterminism:
    """Test deterministic behavior with seed."""

    def test_same_seed_same_result(self):
        inst = ParallelMachineInstance.random_identical(n=12, m=3, seed=42)
        sol1 = iterated_greedy(inst, max_iterations=200, seed=42)
        sol2 = iterated_greedy(inst, max_iterations=200, seed=42)
        assert abs(sol1.makespan - sol2.makespan) < 1e-6

    def test_different_seed_both_valid(self):
        inst = ParallelMachineInstance.random_identical(n=10, m=3, seed=42)
        sol1 = iterated_greedy(inst, seed=1)
        sol2 = iterated_greedy(inst, seed=999)
        jobs1 = sorted(j for jobs in sol1.assignment for j in jobs)
        jobs2 = sorted(j for jobs in sol2.assignment for j in jobs)
        assert jobs1 == list(range(inst.n))
        assert jobs2 == list(range(inst.n))


class TestPMIGEdgeCases:
    """Test edge cases."""

    def test_single_job(self):
        inst = ParallelMachineInstance(
            n=1, m=3,
            processing_times=np.array([10.0]),
        )
        sol = iterated_greedy(inst, seed=42)
        all_jobs = [j for jobs in sol.assignment for j in jobs]
        assert sorted(all_jobs) == [0]

    def test_time_limit(self):
        inst = ParallelMachineInstance.random_identical(n=20, m=4, seed=42)
        sol = iterated_greedy(
            inst, max_iterations=1000000, time_limit=0.5, seed=42
        )
        all_jobs = sorted(j for jobs in sol.assignment for j in jobs)
        assert all_jobs == list(range(inst.n))
