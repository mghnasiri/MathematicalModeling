"""Tests for VNS on Parallel Machine Scheduling."""

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


_inst = _load_mod("pm_instance_test_vns", os.path.join(_parent_dir, "instance.py"))
ParallelMachineInstance = _inst.ParallelMachineInstance
compute_makespan = _inst.compute_makespan

_vns = _load_mod(
    "pm_vns_test",
    os.path.join(_parent_dir, "metaheuristics", "vns.py"),
)
vns = _vns.vns

_lpt = _load_mod("pm_lpt_test_vns", os.path.join(_parent_dir, "heuristics", "lpt.py"))
lpt = _lpt.lpt


class TestPMVNSValidity:
    def test_identical_valid(self):
        inst = ParallelMachineInstance.random_identical(n=10, m=3, seed=42)
        sol = vns(inst, seed=42)
        all_jobs = sorted(j for m_jobs in sol.assignment for j in m_jobs)
        assert all_jobs == list(range(inst.n))

    def test_makespan_matches(self):
        inst = ParallelMachineInstance.random_identical(n=10, m=3, seed=42)
        sol = vns(inst, seed=42)
        actual = compute_makespan(inst, sol.assignment)
        assert abs(sol.makespan - actual) < 1e-6


class TestPMVNSQuality:
    def test_vns_competitive_with_lpt(self):
        inst = ParallelMachineInstance.random_identical(n=12, m=3, seed=42)
        lpt_sol = lpt(inst)
        vns_sol = vns(inst, max_iterations=200, seed=42)
        assert vns_sol.makespan <= lpt_sol.makespan + 1e-6


class TestPMVNSDeterminism:
    def test_same_seed(self):
        inst = ParallelMachineInstance.random_identical(n=10, m=3, seed=42)
        sol1 = vns(inst, max_iterations=100, seed=42)
        sol2 = vns(inst, max_iterations=100, seed=42)
        assert abs(sol1.makespan - sol2.makespan) < 1e-6

    def test_different_seed_both_valid(self):
        inst = ParallelMachineInstance.random_identical(n=8, m=3, seed=42)
        sol1 = vns(inst, seed=1)
        sol2 = vns(inst, seed=999)
        j1 = sorted(j for m in sol1.assignment for j in m)
        j2 = sorted(j for m in sol2.assignment for j in m)
        assert j1 == list(range(inst.n))
        assert j2 == list(range(inst.n))


class TestPMVNSEdgeCases:
    def test_time_limit(self):
        inst = ParallelMachineInstance.random_identical(n=15, m=4, seed=42)
        sol = vns(inst, max_iterations=1000000, time_limit=0.5, seed=42)
        all_jobs = sorted(j for m in sol.assignment for j in m)
        assert all_jobs == list(range(inst.n))
