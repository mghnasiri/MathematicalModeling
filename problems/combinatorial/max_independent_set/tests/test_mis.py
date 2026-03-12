"""Tests for Maximum Independent Set Problem."""
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
_inst = _load_mod("mis_instance", os.path.join(_base, "instance.py"))
_greedy = _load_mod("mis_greedy", os.path.join(_base, "heuristics", "greedy_mis.py"))
_bb = _load_mod("mis_bb", os.path.join(_base, "exact", "branch_and_bound.py"))

MISInstance = _inst.MISInstance
MISSolution = _inst.MISSolution
greedy_min_degree = _greedy.greedy_min_degree
greedy_random = _greedy.greedy_random
branch_and_bound = _bb.branch_and_bound


class TestMISInstance:
    def test_cycle(self):
        inst = MISInstance.cycle(6)
        assert inst.n_vertices == 6

    def test_is_independent(self):
        inst = MISInstance.cycle(4)  # 0-1-2-3-0
        assert inst.is_independent([0, 2])
        assert not inst.is_independent([0, 1])

    def test_random(self):
        inst = MISInstance.random(n_vertices=15)
        assert inst.n_vertices == 15


class TestGreedy:
    def test_min_degree_valid(self):
        inst = MISInstance.cycle(6)
        sol = greedy_min_degree(inst)
        assert sol.is_valid
        assert sol.size == 3  # floor(6/2)

    def test_random_greedy_valid(self):
        inst = MISInstance.cycle(6)
        sol = greedy_random(inst, n_starts=10, seed=42)
        assert sol.is_valid
        assert sol.size >= 2

    def test_random_instance(self):
        inst = MISInstance.random(n_vertices=20, density=0.3)
        sol = greedy_min_degree(inst)
        assert sol.is_valid
        assert sol.size > 0


class TestBranchAndBound:
    def test_cycle_optimal(self):
        inst = MISInstance.cycle(6)
        sol = branch_and_bound(inst)
        assert sol.is_valid
        assert sol.size == 3

    def test_cycle_odd(self):
        inst = MISInstance.cycle(5)
        sol = branch_and_bound(inst)
        assert sol.is_valid
        assert sol.size == 2  # floor(5/2)

    def test_complete_graph(self):
        edges = [(i, j) for i in range(5) for j in range(i + 1, 5)]
        inst = MISInstance(n_vertices=5, edges=edges)
        sol = branch_and_bound(inst)
        assert sol.size == 1

    def test_empty_graph(self):
        inst = MISInstance(n_vertices=5, edges=[])
        sol = branch_and_bound(inst)
        assert sol.size == 5

    def test_bb_vs_greedy(self):
        inst = MISInstance.random(n_vertices=12, density=0.3, seed=7)
        sol_g = greedy_min_degree(inst)
        sol_bb = branch_and_bound(inst, time_limit_nodes=50000)
        assert sol_bb.size >= sol_g.size
