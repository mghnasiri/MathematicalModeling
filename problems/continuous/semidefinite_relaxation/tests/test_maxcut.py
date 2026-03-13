"""Tests for SDP Relaxation / MAX-CUT problem.

Tests: cut value computation, relaxation bound, approximation quality.
"""
from __future__ import annotations

import sys
import os
import importlib.util
import numpy as np
import pytest

def _load_mod(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_base = os.path.join(os.path.dirname(__file__), "..")

_instance_mod = _load_mod("mc_inst_test", os.path.join(_base, "instance.py"))
_gw_mod = _load_mod("mc_gw_test", os.path.join(_base, "heuristics", "goemans_williamson.py"))

MaxCutInstance = _instance_mod.MaxCutInstance
MaxCutSolution = _instance_mod.MaxCutSolution
goemans_williamson = _gw_mod.goemans_williamson


class TestMaxCutInstance:
    """Test instance creation and cut computation."""

    def test_random_instance(self):
        inst = MaxCutInstance.random(n=8, density=0.5, seed=42)
        assert inst.n == 8
        assert inst.adjacency.shape == (8, 8)
        np.testing.assert_array_equal(inst.adjacency, inst.adjacency.T)

    def test_cut_value_known(self):
        adj = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ], dtype=float)
        inst = MaxCutInstance(n=3, adjacency=adj)
        # {0} vs {1,2} -> edge (0,1) crosses -> cut = 1
        assert inst.cut_value([0, 1, 1]) == 1.0
        # {0,2} vs {1} -> edges (0,1) and (1,2) cross -> cut = 2
        assert inst.cut_value([0, 1, 0]) == 2.0

    def test_cut_value_all_same_partition(self):
        adj = np.array([[0, 5], [5, 0]], dtype=float)
        inst = MaxCutInstance(n=2, adjacency=adj)
        assert inst.cut_value([0, 0]) == 0.0

    def test_total_edge_weight(self):
        adj = np.array([
            [0, 3, 0],
            [3, 0, 7],
            [0, 7, 0],
        ], dtype=float)
        inst = MaxCutInstance(n=3, adjacency=adj)
        assert inst.total_edge_weight() == 10.0

    def test_cut_with_negative_labels(self):
        """Support -1/1 labels."""
        adj = np.array([[0, 5], [5, 0]], dtype=float)
        inst = MaxCutInstance(n=2, adjacency=adj)
        assert inst.cut_value([-1, 1]) == 5.0


class TestGoemansWilliamson:
    """Test Goemans-Williamson heuristic."""

    def test_simple_cut(self):
        adj = np.array([[0, 1], [1, 0]], dtype=float)
        inst = MaxCutInstance(n=2, adjacency=adj)
        sol = goemans_williamson(inst, n_rounds=50, seed=42)
        assert sol.cut_value == 1.0

    def test_triangle(self):
        adj = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ], dtype=float)
        inst = MaxCutInstance(n=3, adjacency=adj)
        sol = goemans_williamson(inst, n_rounds=100, seed=42)
        # Best cut for K3 is 2 (any 1 vs 2 split)
        assert sol.cut_value == 2.0

    def test_sdp_bound_valid(self):
        inst = MaxCutInstance.random(n=8, density=0.5, seed=42)
        sol = goemans_williamson(inst, n_rounds=100)
        assert sol.sdp_bound is not None
        assert sol.sdp_bound >= sol.cut_value - 1e-9

    def test_cut_value_nonnegative(self):
        inst = MaxCutInstance.random(n=10, seed=42)
        sol = goemans_williamson(inst)
        assert sol.cut_value >= 0

    def test_empty_graph(self):
        inst = MaxCutInstance(n=0, adjacency=np.zeros((0, 0)))
        sol = goemans_williamson(inst)
        assert sol.cut_value == 0.0

    def test_single_vertex(self):
        inst = MaxCutInstance(n=1, adjacency=np.zeros((1, 1)))
        sol = goemans_williamson(inst)
        assert sol.cut_value == 0.0

    def test_solution_repr(self):
        inst = MaxCutInstance.random(n=5, seed=42)
        sol = goemans_williamson(inst)
        r = repr(sol)
        assert "MaxCutSolution" in r
        assert "cut=" in r

    def test_partition_valid(self):
        inst = MaxCutInstance.random(n=10, seed=42)
        sol = goemans_williamson(inst)
        assert len(sol.partition) == 10
        assert all(p in (0, 1) for p in sol.partition)
