"""
Test suite for Shortest Path Problem.

Tests cover:
- Instance creation and validation
- Dijkstra's algorithm
- Bellman-Ford algorithm
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


_inst_mod = _load_module("sp_inst_test", os.path.join(_base_dir, "instance.py"))
_dj_mod = _load_module("sp_dj_test", os.path.join(_base_dir, "exact", "dijkstra.py"))
_bf_mod = _load_module("sp_bf_test", os.path.join(_base_dir, "exact", "bellman_ford.py"))

ShortestPathInstance = _inst_mod.ShortestPathInstance
ShortestPathSolution = _inst_mod.ShortestPathSolution
validate_solution = _inst_mod.validate_solution
simple_graph_5 = _inst_mod.simple_graph_5
negative_weight_graph = _inst_mod.negative_weight_graph

dijkstra = _dj_mod.dijkstra
dijkstra_all = _dj_mod.dijkstra_all
bellman_ford = _bf_mod.bellman_ford


@pytest.fixture
def inst5():
    return simple_graph_5()


@pytest.fixture
def inst_neg():
    return negative_weight_graph()


class TestShortestPathInstance:
    def test_create_from_edges(self, inst5):
        assert inst5.n == 5
        assert len(inst5.edges) == 7

    def test_from_matrix(self):
        mat = np.array([
            [0, 1, np.inf],
            [np.inf, 0, 2],
            [np.inf, np.inf, 0],
        ])
        inst = ShortestPathInstance.from_matrix(mat)
        assert inst.n == 3
        assert len(inst.edges) == 2

    def test_random_instance(self):
        inst = ShortestPathInstance.random(10, density=0.3, seed=42)
        assert inst.n == 10
        assert len(inst.edges) > 0

    def test_has_negative_weights(self, inst_neg):
        assert inst_neg.has_negative_weights()

    def test_no_negative_weights(self, inst5):
        assert not inst5.has_negative_weights()


class TestValidation:
    def test_valid_path(self, inst5):
        sol = dijkstra(inst5, 0, 4)
        valid, errors = validate_solution(inst5, sol)
        assert valid, errors

    def test_invalid_path(self, inst5):
        sol = ShortestPathSolution(
            source=0, target=4,
            path=[0, 4],  # No direct edge 0->4
            distance=10.0)
        valid, errors = validate_solution(inst5, sol)
        assert not valid


class TestDijkstra:
    def test_simple_shortest_path(self, inst5):
        sol = dijkstra(inst5, 0, 4)
        assert abs(sol.distance - 7.0) < 1e-6
        assert sol.path == [0, 1, 3, 4]

    def test_source_to_self(self, inst5):
        sol = dijkstra(inst5, 0, 0)
        assert sol.distance == 0.0
        assert sol.path == [0]

    def test_all_distances(self, inst5):
        dists, _ = dijkstra_all(inst5, 0)
        assert abs(dists[0] - 0.0) < 1e-6
        assert abs(dists[1] - 2.0) < 1e-6
        assert abs(dists[4] - 7.0) < 1e-6

    def test_no_path(self):
        edges = [(0, 1, 5)]
        inst = ShortestPathInstance.from_edges(3, edges)
        sol = dijkstra(inst, 0, 2)
        assert sol.distance == float("inf")
        assert sol.path == []

    def test_rejects_negative_weights(self, inst_neg):
        with pytest.raises(ValueError, match="non-negative"):
            dijkstra(inst_neg, 0, 4)

    def test_valid_solution(self, inst5):
        sol = dijkstra(inst5, 0, 4)
        valid, errors = validate_solution(inst5, sol)
        assert valid, errors

    def test_random_graph(self):
        inst = ShortestPathInstance.random(10, density=0.4, seed=42)
        sol = dijkstra(inst, 0, 9)
        if sol.path:
            valid, errors = validate_solution(inst, sol)
            assert valid, errors


class TestBellmanFord:
    def test_simple_graph(self, inst5):
        sol = bellman_ford(inst5, 0, 4)
        assert abs(sol.distance - 7.0) < 1e-6
        assert sol.path == [0, 1, 3, 4]

    def test_negative_weights(self, inst_neg):
        sol = bellman_ford(inst_neg, 0, 4)
        valid, errors = validate_solution(inst_neg, sol)
        assert valid, errors
        assert sol.distance < 5.0  # Benefits from negative edges

    def test_agrees_with_dijkstra(self, inst5):
        dj_sol = dijkstra(inst5, 0, 4)
        bf_sol = bellman_ford(inst5, 0, 4)
        assert abs(dj_sol.distance - bf_sol.distance) < 1e-6

    def test_no_path(self):
        edges = [(0, 1, 5)]
        inst = ShortestPathInstance.from_edges(3, edges)
        sol = bellman_ford(inst, 0, 2)
        assert sol.distance == float("inf")

    def test_negative_cycle_detection(self):
        edges = [(0, 1, 1), (1, 2, -3), (2, 0, 1)]
        inst = ShortestPathInstance.from_edges(3, edges)
        with pytest.raises(ValueError, match="[Nn]egative"):
            bellman_ford(inst, 0, 2)

    def test_valid_solution(self, inst_neg):
        sol = bellman_ford(inst_neg, 0, 4)
        valid, errors = validate_solution(inst_neg, sol)
        assert valid, errors


class TestCrossMethod:
    def test_agree_on_nonneg_graph(self, inst5):
        dj = dijkstra(inst5, 0, 4)
        bf = bellman_ford(inst5, 0, 4)
        assert abs(dj.distance - bf.distance) < 1e-6
        assert dj.path == bf.path
