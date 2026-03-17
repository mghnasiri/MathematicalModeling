"""Tests for Variable Neighborhood Search on Job Shop Scheduling."""

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


_inst = _load_mod("jsp_instance_test_vns", os.path.join(_parent_dir, "instance.py"))
JobShopInstance = _inst.JobShopInstance
JobShopSolution = _inst.JobShopSolution
compute_makespan = _inst.compute_makespan
ft06 = _inst.ft06

_vns = _load_mod(
    "jsp_vns_test",
    os.path.join(_parent_dir, "metaheuristics", "vns.py"),
)
vns = _vns.vns

_disp = _load_mod(
    "jsp_disp_test_vns",
    os.path.join(_parent_dir, "heuristics", "dispatching_rules.py"),
)
dispatching_rule = _disp.dispatching_rule


class TestJSPVNSValidity:
    """Test that VNS produces valid solutions."""

    def test_ft06_valid(self):
        inst = ft06()
        sol = vns(inst, max_iterations=50, seed=42)
        ms = compute_makespan(inst, sol.start_times)
        assert ms == sol.makespan

    def test_random_instance_valid(self):
        inst = JobShopInstance.random(n=5, m=4, seed=42)
        sol = vns(inst, max_iterations=50, seed=42)
        ms = compute_makespan(inst, sol.start_times)
        assert ms == sol.makespan

    def test_all_operations_scheduled(self):
        inst = ft06()
        sol = vns(inst, max_iterations=50, seed=42)
        for j in range(inst.n):
            for k in range(len(inst.jobs[j])):
                assert (j, k) in sol.start_times


class TestJSPVNSQuality:
    """Test solution quality."""

    def test_ft06_reasonable(self):
        inst = ft06()
        sol = vns(inst, max_iterations=100, seed=42)
        assert sol.makespan >= 55  # Optimal = 55
        assert sol.makespan <= 80  # Should be reasonably good

    def test_vns_not_worse_than_dispatching(self):
        inst = ft06()
        best_disp = min(
            dispatching_rule(inst, rule=r).makespan
            for r in ["spt", "mwr", "lpt"]
        )
        sol_vns = vns(inst, max_iterations=100, seed=42)
        assert sol_vns.makespan <= best_disp


class TestJSPVNSDeterminism:
    """Test deterministic behavior with seed."""

    def test_same_seed_same_result(self):
        inst = ft06()
        sol1 = vns(inst, max_iterations=50, seed=42)
        sol2 = vns(inst, max_iterations=50, seed=42)
        assert sol1.makespan == sol2.makespan

    def test_different_seed_both_valid(self):
        inst = ft06()
        sol1 = vns(inst, max_iterations=50, seed=1)
        sol2 = vns(inst, max_iterations=50, seed=999)
        assert sol1.makespan >= 55
        assert sol2.makespan >= 55


class TestJSPVNSEdgeCases:
    """Test edge cases."""

    def test_time_limit(self):
        inst = ft06()
        sol = vns(inst, max_iterations=1000000, time_limit=0.5, seed=42)
        ms = compute_makespan(inst, sol.start_times)
        assert ms == sol.makespan
