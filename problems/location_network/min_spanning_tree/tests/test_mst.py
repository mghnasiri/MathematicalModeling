"""
Test suite for Minimum Spanning Tree Problem.

Tests cover:
- Instance creation and validation
- Kruskal's algorithm
- Prim's algorithm
- Agreement between methods
"""

from __future__ import annotations

import os
import sys
import pytest
import numpy as np
import importlib.util

_base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_module(name: str, filepath: str):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_inst_mod = _load_module("mst_inst_test", os.path.join(_base_dir, "instance.py"))
_alg_mod = _load_module("mst_alg_test", os.path.join(_base_dir, "exact", "mst_algorithms.py"))

MSTInstance = _inst_mod.MSTInstance
MSTSolution = _inst_mod.MSTSolution
validate_solution = _inst_mod.validate_solution
triangle_graph = _inst_mod.triangle_graph
simple_graph_6 = _inst_mod.simple_graph_6

kruskal = _alg_mod.kruskal
prim = _alg_mod.prim


@pytest.fixture
def inst3():
    return triangle_graph()


@pytest.fixture
def inst6():
    return simple_graph_6()


@pytest.fixture
def random_inst():
    return MSTInstance.random(15, density=0.4, seed=42)


class TestMSTInstance:
    def test_create_triangle(self, inst3):
        assert inst3.n == 3
        assert len(inst3.edges) == 3

    def test_from_matrix(self):
        mat = np.array([
            [0, 3, np.inf],
            [3, 0, 5],
            [np.inf, 5, 0],
        ])
        inst = MSTInstance.from_matrix(mat)
        assert inst.n == 3
        assert len(inst.edges) == 2

    def test_random_connected(self):
        inst = MSTInstance.random(10, seed=42)
        # Should be connected: MST should have n-1 edges
        sol = kruskal(inst)
        assert len(sol.tree_edges) == 9


class TestValidation:
    def test_valid_tree(self, inst3):
        sol = kruskal(inst3)
        valid, errors = validate_solution(inst3, sol)
        assert valid, errors

    def test_wrong_edge_count(self, inst3):
        sol = MSTSolution(
            tree_edges=[(0, 1, 1)],  # Only 1, need 2
            total_weight=1.0)
        valid, errors = validate_solution(inst3, sol)
        assert not valid


class TestKruskal:
    def test_triangle(self, inst3):
        sol = kruskal(inst3)
        assert abs(sol.total_weight - 3.0) < 1e-6
        assert len(sol.tree_edges) == 2

    def test_simple_6(self, inst6):
        sol = kruskal(inst6)
        assert len(sol.tree_edges) == 5
        valid, errors = validate_solution(inst6, sol)
        assert valid, errors

    def test_known_optimal(self, inst6):
        sol = kruskal(inst6)
        # MST of simple6: (1,2,1), (4,5,1), (1,3,2), (3,5,2), (0,2,3) = 9
        assert abs(sol.total_weight - 9.0) < 1e-6

    def test_random_instance(self, random_inst):
        sol = kruskal(random_inst)
        valid, errors = validate_solution(random_inst, sol)
        assert valid, errors


class TestPrim:
    def test_triangle(self, inst3):
        sol = prim(inst3)
        assert abs(sol.total_weight - 3.0) < 1e-6
        assert len(sol.tree_edges) == 2

    def test_simple_6(self, inst6):
        sol = prim(inst6)
        valid, errors = validate_solution(inst6, sol)
        assert valid, errors

    def test_known_optimal(self, inst6):
        sol = prim(inst6)
        assert abs(sol.total_weight - 9.0) < 1e-6

    def test_random_instance(self, random_inst):
        sol = prim(random_inst)
        valid, errors = validate_solution(random_inst, sol)
        assert valid, errors


class TestCrossMethod:
    def test_agree_triangle(self, inst3):
        k = kruskal(inst3)
        p = prim(inst3)
        assert abs(k.total_weight - p.total_weight) < 1e-6

    def test_agree_simple6(self, inst6):
        k = kruskal(inst6)
        p = prim(inst6)
        assert abs(k.total_weight - p.total_weight) < 1e-6

    def test_agree_random(self, random_inst):
        k = kruskal(random_inst)
        p = prim(random_inst)
        assert abs(k.total_weight - p.total_weight) < 1e-6

    def test_both_valid(self, inst6):
        for method in [kruskal, prim]:
            sol = method(inst6)
            valid, errors = validate_solution(inst6, sol)
            assert valid, errors
