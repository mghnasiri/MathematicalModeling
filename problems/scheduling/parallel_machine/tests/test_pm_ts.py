"""Tests for Tabu Search on Parallel Machine Scheduling."""

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


_inst = _load_mod("pm_instance_test_ts", os.path.join(_parent_dir, "instance.py"))
ParallelMachineInstance = _inst.ParallelMachineInstance
ParallelMachineSolution = _inst.ParallelMachineSolution
compute_makespan = _inst.compute_makespan

_ts = _load_mod(
    "pm_ts_test",
    os.path.join(_parent_dir, "metaheuristics", "tabu_search.py"),
)
tabu_search = _ts.tabu_search

_lpt = _load_mod(
    "pm_lpt_test_ts",
    os.path.join(_parent_dir, "heuristics", "lpt.py"),
)
lpt = _lpt.lpt


class TestPMTSValidity:
    """Test that TS produces valid solutions."""

    def test_identical_valid(self):
        inst = ParallelMachineInstance.random_identical(n=10, m=3, seed=42)
        sol = tabu_search(inst, seed=42)
        # All jobs assigned
        all_jobs = set()
        for machine_jobs in sol.assignment:
            all_jobs.update(machine_jobs)
        assert all_jobs == set(range(inst.n))

    def test_uniform_valid(self):
        inst = ParallelMachineInstance.random_uniform(n=10, m=3, seed=42)
        sol = tabu_search(inst, seed=42)
        all_jobs = set()
        for machine_jobs in sol.assignment:
            all_jobs.update(machine_jobs)
        assert all_jobs == set(range(inst.n))

    def test_makespan_matches(self):
        inst = ParallelMachineInstance.random_identical(n=10, m=3, seed=42)
        sol = tabu_search(inst, seed=42)
        actual_ms = compute_makespan(inst, sol.assignment)
        assert abs(sol.makespan - actual_ms) < 1e-6


class TestPMTSQuality:
    """Test solution quality."""

    def test_ts_not_worse_than_lpt(self):
        inst = ParallelMachineInstance.random_identical(n=15, m=3, seed=42)
        lpt_sol = lpt(inst)
        ts_sol = tabu_search(inst, max_iterations=1000, seed=42)
        assert ts_sol.makespan <= lpt_sol.makespan + 1e-6

    def test_makespan_positive(self):
        inst = ParallelMachineInstance.random_identical(n=10, m=3, seed=42)
        sol = tabu_search(inst, seed=42)
        assert sol.makespan > 0


class TestPMTSDeterminism:
    """Test deterministic behavior with seed."""

    def test_same_seed_same_result(self):
        inst = ParallelMachineInstance.random_identical(n=10, m=3, seed=42)
        sol1 = tabu_search(inst, seed=42)
        sol2 = tabu_search(inst, seed=42)
        assert abs(sol1.makespan - sol2.makespan) < 1e-6

    def test_different_seed_both_valid(self):
        inst = ParallelMachineInstance.random_identical(n=10, m=3, seed=42)
        sol1 = tabu_search(inst, seed=1)
        sol2 = tabu_search(inst, seed=999)
        for sol in [sol1, sol2]:
            all_jobs = set()
            for machine_jobs in sol.assignment:
                all_jobs.update(machine_jobs)
            assert all_jobs == set(range(inst.n))


class TestPMTSEdgeCases:
    """Test edge cases."""

    def test_single_machine(self):
        inst = ParallelMachineInstance.random_identical(n=5, m=1, seed=42)
        sol = tabu_search(inst, seed=42)
        assert len(sol.assignment) == 1
        assert set(sol.assignment[0]) == set(range(inst.n))

    def test_time_limit(self):
        inst = ParallelMachineInstance.random_identical(n=15, m=3, seed=42)
        sol = tabu_search(
            inst, max_iterations=1000000, time_limit=0.5, seed=42
        )
        all_jobs = set()
        for machine_jobs in sol.assignment:
            all_jobs.update(machine_jobs)
        assert all_jobs == set(range(inst.n))
