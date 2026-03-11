"""Tests for Local Search on Parallel Machine Scheduling."""

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


_inst = _load_mod("pm_instance_test_ls", os.path.join(_parent_dir, "instance.py"))
ParallelMachineInstance = _inst.ParallelMachineInstance
ParallelMachineSolution = _inst.ParallelMachineSolution
compute_makespan = _inst.compute_makespan

_ls = _load_mod(
    "pm_ls_test",
    os.path.join(_parent_dir, "metaheuristics", "local_search.py"),
)
local_search = _ls.local_search

_lpt = _load_mod(
    "pm_lpt_test_ls",
    os.path.join(_parent_dir, "heuristics", "lpt.py"),
)
lpt = _lpt.lpt


class TestPMLSValidity:
    """Test that LS produces valid solutions."""

    def test_identical_valid(self):
        inst = ParallelMachineInstance.random_identical(n=10, m=3, seed=42)
        sol = local_search(inst, seed=42)
        all_jobs = sorted(j for m_jobs in sol.assignment for j in m_jobs)
        assert all_jobs == list(range(inst.n))

    def test_uniform_valid(self):
        inst = ParallelMachineInstance.random_uniform(n=10, m=3, seed=42)
        sol = local_search(inst, seed=42)
        all_jobs = sorted(j for m_jobs in sol.assignment for j in m_jobs)
        assert all_jobs == list(range(inst.n))

    def test_makespan_matches(self):
        inst = ParallelMachineInstance.random_identical(n=10, m=3, seed=42)
        sol = local_search(inst, seed=42)
        actual = compute_makespan(inst, sol.assignment)
        assert abs(sol.makespan - actual) < 1e-6


class TestPMLSQuality:
    """Test solution quality."""

    def test_ls_competitive_with_lpt(self):
        inst = ParallelMachineInstance.random_identical(n=12, m=3, seed=42)
        lpt_sol = lpt(inst)
        ls_sol = local_search(inst, max_iterations=200, seed=42)
        assert ls_sol.makespan <= lpt_sol.makespan + 1e-6


class TestPMLSDeterminism:
    """Test deterministic behavior with seed."""

    def test_same_seed_same_result(self):
        inst = ParallelMachineInstance.random_identical(n=10, m=3, seed=42)
        sol1 = local_search(inst, max_iterations=100, seed=42)
        sol2 = local_search(inst, max_iterations=100, seed=42)
        assert abs(sol1.makespan - sol2.makespan) < 1e-6

    def test_different_seed_both_valid(self):
        inst = ParallelMachineInstance.random_identical(n=8, m=3, seed=42)
        sol1 = local_search(inst, seed=1)
        sol2 = local_search(inst, seed=999)
        jobs1 = sorted(j for m_jobs in sol1.assignment for j in m_jobs)
        jobs2 = sorted(j for m_jobs in sol2.assignment for j in m_jobs)
        assert jobs1 == list(range(inst.n))
        assert jobs2 == list(range(inst.n))


class TestPMLSEdgeCases:
    """Test edge cases."""

    def test_single_machine(self):
        inst = ParallelMachineInstance.random_identical(n=5, m=1, seed=42)
        sol = local_search(inst, seed=42)
        assert len(sol.assignment) == 1
        assert sorted(sol.assignment[0]) == list(range(5))

    def test_time_limit(self):
        inst = ParallelMachineInstance.random_identical(n=15, m=4, seed=42)
        sol = local_search(inst, max_iterations=1000000, time_limit=0.5, seed=42)
        all_jobs = sorted(j for m_jobs in sol.assignment for j in m_jobs)
        assert all_jobs == list(range(inst.n))
