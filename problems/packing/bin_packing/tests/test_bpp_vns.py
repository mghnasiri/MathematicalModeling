"""Tests for VNS on 1D Bin Packing."""

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


_inst = _load_mod("bpp_instance_test_vns", os.path.join(_parent_dir, "instance.py"))
BinPackingInstance = _inst.BinPackingInstance
BinPackingSolution = _inst.BinPackingSolution
validate_solution = _inst.validate_solution

_vns = _load_mod(
    "bpp_vns_test",
    os.path.join(_parent_dir, "metaheuristics", "vns.py"),
)
vns = _vns.vns

_ff = _load_mod(
    "bpp_ff_test_vns",
    os.path.join(_parent_dir, "heuristics", "first_fit.py"),
)
first_fit_decreasing = _ff.first_fit_decreasing


class TestBPPVNSValidity:
    def test_all_items_packed(self):
        inst = BinPackingInstance.random(n=15, seed=42)
        sol = vns(inst, seed=42)
        all_items = sorted(i for b in sol.bins for i in b)
        assert all_items == list(range(inst.n))

    def test_capacity_respected(self):
        inst = BinPackingInstance.random(n=15, seed=42)
        sol = vns(inst, seed=42)
        for b in sol.bins:
            total = sum(inst.sizes[i] for i in b)
            assert total <= inst.capacity + 1e-10

    def test_validate_solution(self):
        inst = BinPackingInstance.random(n=15, seed=42)
        sol = vns(inst, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Validation errors: {errors}"


class TestBPPVNSQuality:
    def test_competitive_with_ffd(self):
        inst = BinPackingInstance.random(n=20, seed=42)
        ffd_sol = first_fit_decreasing(inst)
        vns_sol = vns(inst, max_iterations=200, seed=42)
        assert vns_sol.num_bins <= ffd_sol.num_bins

    def test_respects_lower_bound(self):
        inst = BinPackingInstance.random(n=15, seed=42)
        sol = vns(inst, seed=42)
        assert sol.num_bins >= inst.lower_bound_l1()


class TestBPPVNSDeterminism:
    def test_same_seed(self):
        inst = BinPackingInstance.random(n=12, seed=42)
        sol1 = vns(inst, max_iterations=100, seed=42)
        sol2 = vns(inst, max_iterations=100, seed=42)
        assert sol1.num_bins == sol2.num_bins


class TestBPPVNSEdgeCases:
    def test_time_limit(self):
        inst = BinPackingInstance.random(n=20, seed=42)
        sol = vns(inst, max_iterations=1000000, time_limit=0.5, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Validation errors: {errors}"
