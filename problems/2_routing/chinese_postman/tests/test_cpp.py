"""Tests for Chinese Postman Problem."""
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
_inst_mod = _load_mod("cpp_instance", os.path.join(_base, "instance.py"))
_solver_mod = _load_mod(
    "cpp_solver", os.path.join(_base, "exact", "chinese_postman_solver.py")
)

ChinesePostmanInstance = _inst_mod.ChinesePostmanInstance
ChinesePostmanSolution = _inst_mod.ChinesePostmanSolution
validate_solution = _inst_mod.validate_solution
eulerian_square = _inst_mod.eulerian_square
non_eulerian_triangle = _inst_mod.non_eulerian_triangle
bridge_graph = _inst_mod.bridge_graph
chinese_postman = _solver_mod.chinese_postman


class TestInstance:

    def test_eulerian_square(self):
        inst = eulerian_square()
        assert inst.n_vertices == 4
        assert len(inst.edges) == 4
        assert inst.is_eulerian()

    def test_triangle_eulerian(self):
        inst = non_eulerian_triangle()
        assert inst.n_vertices == 3
        assert inst.is_eulerian()  # K3 has all degree 2

    def test_bridge_not_eulerian(self):
        inst = bridge_graph()
        assert not inst.is_eulerian()
        odd = inst.odd_degree_vertices()
        assert len(odd) % 2 == 0
        assert len(odd) > 0

    def test_total_edge_weight(self):
        inst = eulerian_square()
        assert inst.total_edge_weight() == pytest.approx(40.0)

    def test_connectivity(self):
        inst = eulerian_square()
        assert inst.is_connected()

    def test_random(self):
        inst = ChinesePostmanInstance.random(n_vertices=8, seed=42)
        assert inst.n_vertices == 8
        assert inst.is_connected()

    def test_shortest_paths(self):
        inst = eulerian_square()
        sp = inst.shortest_paths()
        assert sp[0][0] == 0.0
        assert sp[0][2] == pytest.approx(20.0)  # 0->1->2 or 0->3->2


class TestSolver:

    def test_eulerian_optimal(self):
        inst = eulerian_square()
        sol = chinese_postman(inst)
        assert sol.total_weight == pytest.approx(40.0)
        assert len(sol.duplicated_edges) == 0

    def test_triangle_optimal(self):
        inst = non_eulerian_triangle()
        sol = chinese_postman(inst)
        # K3 is Eulerian, cost = 5 + 7 + 3 = 15
        assert sol.total_weight == pytest.approx(15.0)

    def test_bridge_duplicates(self):
        inst = bridge_graph()
        sol = chinese_postman(inst)
        assert sol.total_weight >= inst.total_edge_weight()
        assert len(sol.duplicated_edges) > 0

    def test_tour_is_closed(self):
        inst = eulerian_square()
        sol = chinese_postman(inst)
        assert sol.tour[0] == sol.tour[-1]

    def test_random_instance(self):
        inst = ChinesePostmanInstance.random(n_vertices=6, seed=7)
        sol = chinese_postman(inst)
        assert sol.total_weight >= inst.total_edge_weight()
        assert sol.tour[0] == sol.tour[-1]

    def test_repr(self):
        inst = eulerian_square()
        sol = chinese_postman(inst)
        r = repr(sol)
        assert "total_weight" in r
