"""Tests for VNS on RCPSP."""

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


_inst = _load_mod("rcpsp_instance_test_vns", os.path.join(_parent_dir, "instance.py"))
RCPSPInstance = _inst.RCPSPInstance
validate_solution = _inst.validate_solution

_sgs = _load_mod(
    "rcpsp_sgs_test_vns",
    os.path.join(_parent_dir, "heuristics", "serial_sgs.py"),
)
serial_sgs = _sgs.serial_sgs

_vns = _load_mod(
    "rcpsp_vns_test",
    os.path.join(_parent_dir, "metaheuristics", "vns.py"),
)
vns = _vns.vns


class TestRCPSPVNSValidity:
    def test_feasible_schedule(self):
        inst = RCPSPInstance.random(n=8, num_resources=2, seed=42)
        sol = vns(inst, seed=42)
        valid, errors = validate_solution(inst, sol.start_times)
        assert valid, f"Validation errors: {errors}"

    def test_makespan_positive(self):
        inst = RCPSPInstance.random(n=8, num_resources=2, seed=42)
        sol = vns(inst, seed=42)
        assert sol.makespan > 0

    def test_respects_critical_path(self):
        inst = RCPSPInstance.random(n=8, num_resources=2, seed=42)
        sol = vns(inst, seed=42)
        assert sol.makespan >= inst.critical_path_length()


class TestRCPSPVNSQuality:
    def test_vns_competitive_with_sgs(self):
        inst = RCPSPInstance.random(n=10, num_resources=2, seed=42)
        sgs_sol = serial_sgs(inst, priority_rule="lft")
        vns_sol = vns(inst, max_iterations=200, seed=42)
        assert vns_sol.makespan <= sgs_sol.makespan + 1e-6


class TestRCPSPVNSDeterminism:
    def test_same_seed(self):
        inst = RCPSPInstance.random(n=8, num_resources=2, seed=42)
        sol1 = vns(inst, max_iterations=100, seed=42)
        sol2 = vns(inst, max_iterations=100, seed=42)
        assert abs(sol1.makespan - sol2.makespan) < 1e-6

    def test_different_seed_both_valid(self):
        inst = RCPSPInstance.random(n=8, num_resources=2, seed=42)
        sol1 = vns(inst, seed=1)
        sol2 = vns(inst, seed=999)
        valid1, _ = validate_solution(inst, sol1.start_times)
        valid2, _ = validate_solution(inst, sol2.start_times)
        assert valid1
        assert valid2


class TestRCPSPVNSEdgeCases:
    def test_time_limit(self):
        inst = RCPSPInstance.random(n=12, num_resources=2, seed=42)
        sol = vns(inst, max_iterations=1000000, time_limit=0.5, seed=42)
        valid, _ = validate_solution(inst, sol.start_times)
        assert valid
