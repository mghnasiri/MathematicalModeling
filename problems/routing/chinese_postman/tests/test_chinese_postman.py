"""
Test suite for CPP (Chinese Postman Problem).

Tests cover:
- Instance creation and graph properties
- Eulerian detection
- Exact solver (Eulerian and non-Eulerian cases)
- Solution validation
"""

from __future__ import annotations

import os
import sys
import pytest
import numpy as np
import importlib.util

# ── Module loading ───────────────────────────────────────────────────────────

_base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_mod(name: str, filepath: str):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_instance_mod = _load_mod(
    "cpp_instance_test", os.path.join(_base_dir, "instance.py")
)
_solver_mod = _load_mod(
    "cpp_solver_test",
    os.path.join(_base_dir, "exact", "chinese_postman_solver.py"),
)

ChinesePostmanInstance = _instance_mod.ChinesePostmanInstance
ChinesePostmanSolution = _instance_mod.ChinesePostmanSolution
validate_solution = _instance_mod.validate_solution
eulerian_square = _instance_mod.eulerian_square
non_eulerian_triangle = _instance_mod.non_eulerian_triangle
bridge_graph = _instance_mod.bridge_graph

chinese_postman = _solver_mod.chinese_postman


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def inst_square():
    return eulerian_square()


@pytest.fixture
def inst_triangle():
    return non_eulerian_triangle()


@pytest.fixture
def inst_bridge():
    return bridge_graph()


@pytest.fixture
def random_inst():
    return ChinesePostmanInstance.random(6, edge_prob=0.4, seed=42)


# ── Instance tests ───────────────────────────────────────────────────────────


class TestCPPInstance:
    def test_create_basic(self, inst_square):
        assert inst_square.n_vertices == 4
        assert len(inst_square.edges) == 4
        assert inst_square.adj_matrix.shape == (4, 4)

    def test_eulerian_square(self, inst_square):
        assert inst_square.is_eulerian()
        assert inst_square.odd_degree_vertices() == []

    def test_triangle_is_eulerian(self, inst_triangle):
        assert inst_triangle.is_eulerian()

    def test_bridge_not_eulerian(self, inst_bridge):
        assert not inst_bridge.is_eulerian()
        odd = inst_bridge.odd_degree_vertices()
        assert len(odd) % 2 == 0  # Always even number of odd vertices
        assert len(odd) > 0

    def test_connected(self, inst_square):
        assert inst_square.is_connected()

    def test_total_edge_weight(self, inst_square):
        assert abs(inst_square.total_edge_weight() - 40.0) < 1e-10

    def test_random_instance(self):
        inst = ChinesePostmanInstance.random(8, seed=123)
        assert inst.n_vertices == 8
        assert inst.is_connected()
        assert inst.adj_matrix.shape == (8, 8)

    def test_from_edges(self):
        edges = [(0, 1, 5.0), (1, 2, 3.0), (0, 2, 4.0)]
        inst = ChinesePostmanInstance.from_edges(3, edges, "test")
        assert inst.n_vertices == 3
        assert len(inst.edges) == 3
        assert inst.adj_matrix[0][1] == 5.0

    def test_shortest_paths(self, inst_bridge):
        dist = inst_bridge.shortest_paths()
        assert dist[0][0] == 0.0
        assert dist[0][1] == 4.0
        assert dist[0][2] == 7.0  # 0->1->2
        assert dist[0][3] == 12.0  # 0->1->2->3

    def test_degree(self, inst_bridge):
        # 0--1--2--3, 2--4
        # degrees: 0->1, 1->2, 2->3, 3->1, 4->1
        assert inst_bridge.degree(0) == 1
        assert inst_bridge.degree(2) == 3


# ── Solver tests ─────────────────────────────────────────────────────────────


class TestChinesePostmanSolver:
    def test_eulerian_optimal(self, inst_square):
        sol = chinese_postman(inst_square)
        # Eulerian: total weight = sum of all edges = 40
        assert abs(sol.total_weight - 40.0) < 1e-10
        assert len(sol.duplicated_edges) == 0

    def test_eulerian_tour_valid(self, inst_square):
        sol = chinese_postman(inst_square)
        valid, errors = validate_solution(inst_square, sol)
        assert valid, errors

    def test_triangle_optimal(self, inst_triangle):
        sol = chinese_postman(inst_triangle)
        # K3 is Eulerian, weight = 5 + 7 + 3 = 15
        assert abs(sol.total_weight - 15.0) < 1e-10

    def test_triangle_tour_valid(self, inst_triangle):
        sol = chinese_postman(inst_triangle)
        valid, errors = validate_solution(inst_triangle, sol)
        assert valid, errors

    def test_bridge_needs_duplication(self, inst_bridge):
        sol = chinese_postman(inst_bridge)
        assert len(sol.duplicated_edges) > 0
        assert sol.total_weight > inst_bridge.total_edge_weight()

    def test_bridge_tour_valid(self, inst_bridge):
        sol = chinese_postman(inst_bridge)
        valid, errors = validate_solution(inst_bridge, sol)
        assert valid, errors

    def test_bridge_tour_closed(self, inst_bridge):
        sol = chinese_postman(inst_bridge)
        assert sol.tour[0] == sol.tour[-1]

    def test_random_instance(self, random_inst):
        sol = chinese_postman(random_inst)
        valid, errors = validate_solution(random_inst, sol)
        assert valid, errors
        assert sol.total_weight >= random_inst.total_edge_weight()

    def test_repr(self, inst_square):
        sol = chinese_postman(inst_square)
        r = repr(sol)
        assert "ChinesePostmanSolution" in r
        assert "total_weight" in r

    def test_single_edge(self):
        """Graph with single edge: 2 odd vertices, must traverse twice."""
        inst = ChinesePostmanInstance.from_edges(
            2, [(0, 1, 5.0)], "single_edge"
        )
        sol = chinese_postman(inst)
        assert sol.total_weight >= 10.0  # Must go and return
        assert sol.tour[0] == sol.tour[-1]
