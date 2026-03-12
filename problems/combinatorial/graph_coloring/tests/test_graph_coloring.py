"""Tests for Graph Coloring Problem."""
from __future__ import annotations

import sys
import os
import importlib.util

import numpy as np
import pytest

def _load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_base = os.path.join(os.path.dirname(__file__), "..")
_inst = _load_mod("gc_instance", os.path.join(_base, "instance.py"))
_greedy = _load_mod("gc_greedy", os.path.join(_base, "heuristics", "greedy_coloring.py"))

GraphColoringInstance = _inst.GraphColoringInstance
greedy_sequential = _greedy.greedy_sequential
greedy_largest_first = _greedy.greedy_largest_first
dsatur = _greedy.dsatur


class TestGraphColoringInstance:
    def test_petersen(self):
        inst = GraphColoringInstance.petersen()
        assert inst.n_vertices == 10
        assert inst.n_edges == 15

    def test_cycle_even(self):
        inst = GraphColoringInstance.cycle(6)
        assert inst.n_vertices == 6
        assert inst.n_edges == 6

    def test_random(self):
        inst = GraphColoringInstance.random(n_vertices=15, density=0.3)
        assert inst.n_vertices == 15


class TestGreedyColoring:
    def test_sequential_valid(self):
        inst = GraphColoringInstance.petersen()
        sol = greedy_sequential(inst)
        assert sol.is_valid
        assert sol.n_colors >= 3

    def test_largest_first_valid(self):
        inst = GraphColoringInstance.petersen()
        sol = greedy_largest_first(inst)
        assert sol.is_valid

    def test_cycle_even_2_colors(self):
        inst = GraphColoringInstance.cycle(6)
        sol = dsatur(inst)
        assert sol.is_valid
        assert sol.n_colors == 2

    def test_cycle_odd_3_colors(self):
        inst = GraphColoringInstance.cycle(5)
        sol = dsatur(inst)
        assert sol.is_valid
        assert sol.n_colors == 3


class TestDSatur:
    def test_dsatur_valid(self):
        inst = GraphColoringInstance.petersen()
        sol = dsatur(inst)
        assert sol.is_valid
        assert sol.n_colors >= 3

    def test_dsatur_bipartite(self):
        # Complete bipartite K_{3,3}
        edges = [(i, 3 + j) for i in range(3) for j in range(3)]
        inst = GraphColoringInstance(n_vertices=6, edges=edges)
        sol = dsatur(inst)
        assert sol.is_valid
        assert sol.n_colors == 2

    def test_dsatur_complete(self):
        # Complete graph K_4: chi = 4
        edges = [(i, j) for i in range(4) for j in range(i + 1, 4)]
        inst = GraphColoringInstance(n_vertices=4, edges=edges)
        sol = dsatur(inst)
        assert sol.is_valid
        assert sol.n_colors == 4

    def test_random_valid(self):
        inst = GraphColoringInstance.random(n_vertices=20, density=0.4)
        sol = dsatur(inst)
        assert sol.is_valid
