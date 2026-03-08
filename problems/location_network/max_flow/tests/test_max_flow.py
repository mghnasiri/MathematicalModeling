"""
Test suite for Maximum Flow Problem.

Tests cover:
- Instance creation and validation
- Edmonds-Karp algorithm
- Min-cut correctness
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


_inst_mod = _load_module("mf_inst_test", os.path.join(_base_dir, "instance.py"))
_ek_mod = _load_module("mf_ek_test", os.path.join(_base_dir, "exact", "edmonds_karp.py"))

MaxFlowInstance = _inst_mod.MaxFlowInstance
MaxFlowSolution = _inst_mod.MaxFlowSolution
validate_solution = _inst_mod.validate_solution
simple_flow_4 = _inst_mod.simple_flow_4
two_path_flow = _inst_mod.two_path_flow

edmonds_karp = _ek_mod.edmonds_karp


@pytest.fixture
def inst4():
    return simple_flow_4()


@pytest.fixture
def inst_two_path():
    return two_path_flow()


class TestMaxFlowInstance:
    def test_create_basic(self, inst4):
        assert inst4.n == 4
        assert inst4.source == 0
        assert inst4.sink == 3

    def test_from_edges(self):
        edges = [(0, 1, 5), (1, 2, 3)]
        inst = MaxFlowInstance.from_edges(3, 0, 2, edges)
        assert inst.capacity_matrix[0][1] == 5.0

    def test_random_instance(self):
        inst = MaxFlowInstance.random(8, density=0.3, seed=42)
        assert inst.n == 8
        assert inst.source == 0
        assert inst.sink == 7

    def test_invalid_shape(self):
        with pytest.raises(ValueError):
            MaxFlowInstance(
                n=3, source=0, sink=2,
                edges=[], capacity_matrix=np.zeros((2, 2)))


class TestValidation:
    def test_valid_solution(self, inst4):
        sol = edmonds_karp(inst4)
        valid, errors = validate_solution(inst4, sol)
        assert valid, errors

    def test_capacity_violation(self, inst4):
        flow = np.zeros((4, 4))
        flow[0][1] = 100  # Exceeds capacity of 16
        sol = MaxFlowSolution(max_flow=100, flow_matrix=flow)
        valid, errors = validate_solution(inst4, sol)
        assert not valid


class TestEdmondsKarp:
    def test_simple_flow_4(self, inst4):
        sol = edmonds_karp(inst4)
        assert abs(sol.max_flow - 26.0) < 1e-6

    def test_two_path_flow(self, inst_two_path):
        sol = edmonds_karp(inst_two_path)
        assert abs(sol.max_flow - 5.0) < 1e-6

    def test_valid_solution(self, inst4):
        sol = edmonds_karp(inst4)
        valid, errors = validate_solution(inst4, sol)
        assert valid, errors

    def test_min_cut_exists(self, inst4):
        sol = edmonds_karp(inst4)
        assert sol.min_cut is not None
        s_set, t_set = sol.min_cut
        assert inst4.source in s_set
        assert inst4.sink in t_set
        assert len(s_set) + len(t_set) == inst4.n

    def test_min_cut_equals_max_flow(self, inst4):
        sol = edmonds_karp(inst4)
        s_set, t_set = sol.min_cut
        cut_cap = 0.0
        for u in s_set:
            for v in t_set:
                cut_cap += inst4.capacity_matrix[u][v]
        assert abs(cut_cap - sol.max_flow) < 1e-6

    def test_no_flow_graph(self):
        edges = [(0, 1, 5)]  # No path from 0 to 2
        inst = MaxFlowInstance.from_edges(3, 0, 2, edges)
        sol = edmonds_karp(inst)
        assert abs(sol.max_flow) < 1e-6

    def test_single_edge(self):
        edges = [(0, 1, 7)]
        inst = MaxFlowInstance.from_edges(2, 0, 1, edges)
        sol = edmonds_karp(inst)
        assert abs(sol.max_flow - 7.0) < 1e-6

    def test_random_instance(self):
        inst = MaxFlowInstance.random(10, density=0.3, seed=42)
        sol = edmonds_karp(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, errors
        assert sol.max_flow >= 0


class TestMinCut:
    def test_partition_complete(self, inst4):
        sol = edmonds_karp(inst4)
        s_set, t_set = sol.min_cut
        assert set(s_set) | set(t_set) == set(range(inst4.n))
        assert set(s_set) & set(t_set) == set()

    def test_two_path_min_cut(self, inst_two_path):
        sol = edmonds_karp(inst_two_path)
        s_set, t_set = sol.min_cut
        cut_cap = 0.0
        for u in s_set:
            for v in t_set:
                cut_cap += inst_two_path.capacity_matrix[u][v]
        assert abs(cut_cap - 5.0) < 1e-6
