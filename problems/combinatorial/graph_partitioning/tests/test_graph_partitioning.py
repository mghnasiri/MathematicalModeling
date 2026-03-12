"""Tests for Graph Partitioning problem.

Tests: partition balance, edge cut computation, correctness on small instances.
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

_instance_mod = _load_mod("gp_inst_test", os.path.join(_base, "instance.py"))
_kl_mod = _load_mod("gp_kl_test", os.path.join(_base, "heuristics", "greedy_kl.py"))

GraphPartitioningInstance = _instance_mod.GraphPartitioningInstance
GraphPartitioningSolution = _instance_mod.GraphPartitioningSolution
greedy_kl = _kl_mod.greedy_kl


class TestGraphPartitioningInstance:
    """Test instance creation and utility methods."""

    def test_random_instance(self):
        inst = GraphPartitioningInstance.random(n=10, k=2, density=0.5)
        assert inst.n == 10
        assert inst.k == 2
        assert inst.adjacency.shape == (10, 10)
        # Symmetric
        np.testing.assert_array_equal(inst.adjacency, inst.adjacency.T)

    def test_edge_cut_no_edges(self):
        inst = GraphPartitioningInstance(n=4, k=2, adjacency=np.zeros((4, 4)))
        assert inst.edge_cut([0, 0, 1, 1]) == 0.0

    def test_edge_cut_known(self):
        adj = np.array([
            [0, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
        ], dtype=float)
        inst = GraphPartitioningInstance(n=4, k=2, adjacency=adj)
        # Partition: {0,1} vs {2,3} -> edge (1,2) crosses -> cut = 1
        assert inst.edge_cut([0, 0, 1, 1]) == 1.0
        # Partition: {0,2} vs {1,3} -> edges (0,1), (1,2), (2,3) cross -> cut = 3
        assert inst.edge_cut([0, 1, 0, 1]) == 3.0

    def test_is_balanced(self):
        inst = GraphPartitioningInstance(
            n=4, k=2, adjacency=np.zeros((4, 4)), balance_tolerance=0.0
        )
        assert inst.is_balanced([0, 0, 1, 1])
        assert not inst.is_balanced([0, 0, 0, 1])

    def test_is_balanced_with_tolerance(self):
        inst = GraphPartitioningInstance(
            n=5, k=2, adjacency=np.zeros((5, 5)), balance_tolerance=0.2
        )
        # ideal = 2.5, so floor=2, ceil=3. With tolerance 0.2*5=1: min=1, max=4
        assert inst.is_balanced([0, 0, 0, 1, 1])
        assert inst.is_balanced([0, 0, 0, 0, 1])


class TestGreedyKL:
    """Test Kernighan-Lin heuristic."""

    def test_two_way_partition(self):
        inst = GraphPartitioningInstance.random(n=10, k=2, density=0.3, seed=42)
        sol = greedy_kl(inst, seed=42)
        assert len(sol.partition) == 10
        assert set(sol.partition) <= {0, 1}
        assert sol.edge_cut >= 0

    def test_three_way_partition(self):
        inst = GraphPartitioningInstance.random(n=12, k=3, density=0.3, seed=42)
        sol = greedy_kl(inst, seed=42)
        assert len(sol.partition) == 12
        assert set(sol.partition) <= {0, 1, 2}

    def test_edge_cut_matches(self):
        inst = GraphPartitioningInstance.random(n=10, k=2, seed=42)
        sol = greedy_kl(inst, seed=42)
        computed_cut = inst.edge_cut(sol.partition)
        assert abs(sol.edge_cut - computed_cut) < 1e-9

    def test_disconnected_graph(self):
        """Disconnected graph should have zero cut possible."""
        adj = np.zeros((4, 4))
        inst = GraphPartitioningInstance(n=4, k=2, adjacency=adj)
        sol = greedy_kl(inst)
        assert sol.edge_cut == 0.0

    def test_solution_repr(self):
        inst = GraphPartitioningInstance.random(n=6, k=2, seed=42)
        sol = greedy_kl(inst)
        r = repr(sol)
        assert "GraphPartitioningSolution" in r
        assert "edge_cut" in r

    def test_deterministic(self):
        inst = GraphPartitioningInstance.random(n=10, k=2, seed=42)
        sol1 = greedy_kl(inst, seed=10)
        sol2 = greedy_kl(inst, seed=10)
        assert sol1.partition == sol2.partition
        assert sol1.edge_cut == sol2.edge_cut
