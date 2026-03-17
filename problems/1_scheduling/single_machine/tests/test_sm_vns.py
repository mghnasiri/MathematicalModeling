"""Tests for VNS on Single Machine Scheduling."""

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


_inst = _load_mod("sm_instance_test_vns", os.path.join(_parent_dir, "instance.py"))
SingleMachineInstance = _inst.SingleMachineInstance
compute_weighted_tardiness = _inst.compute_weighted_tardiness
compute_total_tardiness = _inst.compute_total_tardiness

_vns = _load_mod(
    "sm_vns_test",
    os.path.join(_parent_dir, "metaheuristics", "vns.py"),
)
vns = _vns.vns


class TestSMVNSValidity:
    def test_weighted_tardiness_valid(self):
        inst = SingleMachineInstance.random(n=10, seed=42)
        sol = vns(inst, objective="weighted_tardiness", seed=42)
        assert sorted(sol.sequence) == list(range(inst.n))
        actual = compute_weighted_tardiness(inst, sol.sequence)
        assert abs(sol.objective_value - actual) < 1e-6

    def test_total_tardiness_valid(self):
        inst = SingleMachineInstance.random(n=10, seed=42)
        sol = vns(inst, objective="total_tardiness", seed=42)
        assert sorted(sol.sequence) == list(range(inst.n))
        actual = compute_total_tardiness(inst, sol.sequence)
        assert abs(sol.objective_value - actual) < 1e-6


class TestSMVNSQuality:
    def test_nonnegative_objective(self):
        inst = SingleMachineInstance.random(n=10, seed=42)
        sol = vns(inst, objective="weighted_tardiness", seed=42)
        assert sol.objective_value >= 0


class TestSMVNSDeterminism:
    def test_same_seed(self):
        inst = SingleMachineInstance.random(n=8, seed=42)
        sol1 = vns(inst, max_iterations=100, seed=42)
        sol2 = vns(inst, max_iterations=100, seed=42)
        assert abs(sol1.objective_value - sol2.objective_value) < 1e-6

    def test_different_seed_both_valid(self):
        inst = SingleMachineInstance.random(n=8, seed=42)
        sol1 = vns(inst, seed=1)
        sol2 = vns(inst, seed=999)
        assert sorted(sol1.sequence) == list(range(inst.n))
        assert sorted(sol2.sequence) == list(range(inst.n))


class TestSMVNSEdgeCases:
    def test_single_job(self):
        inst = SingleMachineInstance(
            n=1,
            processing_times=np.array([10.0]),
            weights=np.array([1.0]),
            due_dates=np.array([5.0]),
        )
        sol = vns(inst, seed=42)
        assert sol.sequence == [0]

    def test_time_limit(self):
        inst = SingleMachineInstance.random(n=12, seed=42)
        sol = vns(inst, max_iterations=1000000, time_limit=0.5, seed=42)
        assert sorted(sol.sequence) == list(range(inst.n))
