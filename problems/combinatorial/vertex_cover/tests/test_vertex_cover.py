"""Tests for Minimum Vertex Cover Problem."""
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
_inst_mod = _load_mod("vc_instance_test", os.path.join(_base, "instance.py"))
_heur_mod = _load_mod(
    "vc_greedy_test", os.path.join(_base, "heuristics", "greedy_vc.py")
)

VertexCoverInstance = _inst_mod.VertexCoverInstance
VertexCoverSolution = _inst_mod.VertexCoverSolution
greedy_edge_cover = _heur_mod.greedy_edge_cover
greedy_degree_cover = _heur_mod.greedy_degree_cover


class TestInstance:

    def test_cycle(self):
        inst = VertexCoverInstance.cycle(6)
        assert inst.n_vertices == 6
        assert inst.n_edges == 6

    def test_star(self):
        inst = VertexCoverInstance.star(4)
        assert inst.n_vertices == 5
        assert inst.n_edges == 4

    def test_is_vertex_cover(self):
        inst = VertexCoverInstance.cycle(4)
        assert inst.is_vertex_cover([0, 2])
        assert not inst.is_vertex_cover([0])

    def test_complete(self):
        inst = VertexCoverInstance.complete(4)
        assert inst.n_edges == 6

    def test_random(self):
        inst = VertexCoverInstance.random(n_vertices=10, seed=42)
        assert inst.n_vertices == 10


class TestGreedyEdge:

    def test_valid_cover(self):
        inst = VertexCoverInstance.cycle(6)
        sol = greedy_edge_cover(inst)
        assert inst.is_vertex_cover(sol.cover)

    def test_2_approx_star(self):
        inst = VertexCoverInstance.star(4)
        sol = greedy_edge_cover(inst)
        assert inst.is_vertex_cover(sol.cover)
        assert sol.size <= 2  # OPT=1, 2*OPT=2

    def test_complete_graph(self):
        inst = VertexCoverInstance.complete(5)
        sol = greedy_edge_cover(inst)
        assert inst.is_vertex_cover(sol.cover)

    def test_random_instance(self):
        inst = VertexCoverInstance.random(n_vertices=15, seed=7)
        sol = greedy_edge_cover(inst)
        assert inst.is_vertex_cover(sol.cover)


class TestGreedyDegree:

    def test_valid_cover(self):
        inst = VertexCoverInstance.cycle(6)
        sol = greedy_degree_cover(inst)
        assert inst.is_vertex_cover(sol.cover)

    def test_star_optimal(self):
        inst = VertexCoverInstance.star(5)
        sol = greedy_degree_cover(inst)
        assert inst.is_vertex_cover(sol.cover)
        assert sol.size == 1  # degree greedy picks center

    def test_degree_vs_edge(self):
        inst = VertexCoverInstance.random(n_vertices=12, seed=42)
        sol_e = greedy_edge_cover(inst)
        sol_d = greedy_degree_cover(inst)
        assert inst.is_vertex_cover(sol_e.cover)
        assert inst.is_vertex_cover(sol_d.cover)

    def test_repr(self):
        inst = VertexCoverInstance.cycle(4)
        sol = greedy_edge_cover(inst)
        assert "size" in repr(sol)
