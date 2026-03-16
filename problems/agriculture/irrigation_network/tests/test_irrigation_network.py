"""
Tests for Agricultural Irrigation Network Design Problem

Covers: instance creation, MST backbone, max flow throughput,
shortest path delivery, and cross-algorithm verification.

28 tests across 5 test classes.
"""
from __future__ import annotations

import sys
import os
import importlib.util

import numpy as np
import pytest


def _load_mod(name: str, filepath: str):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_inst_mod = _load_mod("irr_inst_test", os.path.join(_base, "instance.py"))
_solver_mod = _load_mod(
    "irr_solver_test",
    os.path.join(_base, "exact", "network_analysis.py"),
)

IrrigationNetworkInstance = _inst_mod.IrrigationNetworkInstance
NetworkNode = _inst_mod.NetworkNode
PipeSegment = _inst_mod.PipeSegment


class TestIrrigationNetworkInstance:
    """Test instance creation and data access."""

    def test_standard_farm_creation(self):
        inst = IrrigationNetworkInstance.standard_farm()
        assert inst.n_nodes == 10
        assert inst.n_pipes == 16

    def test_field_nodes(self):
        inst = IrrigationNetworkInstance.standard_farm()
        fields = inst.field_nodes
        assert len(fields) == 5
        assert set(fields) == {4, 5, 6, 7, 8}

    def test_total_field_demand(self):
        inst = IrrigationNetworkInstance.standard_farm()
        assert inst.total_field_demand == 13300  # 3500+2800+2200+1800+3000

    def test_source_sink(self):
        inst = IrrigationNetworkInstance.standard_farm()
        assert inst.source == 0
        assert inst.sink == 9

    def test_mst_edges(self):
        inst = IrrigationNetworkInstance.standard_farm()
        edges = inst.get_mst_edges()
        assert len(edges) >= inst.n_nodes - 1

    def test_flow_edges(self):
        inst = IrrigationNetworkInstance.standard_farm()
        edges = inst.get_flow_edges()
        assert len(edges) == inst.n_pipes

    def test_sp_edges(self):
        inst = IrrigationNetworkInstance.standard_farm()
        edges = inst.get_sp_edges()
        assert len(edges) == inst.n_pipes

    def test_random_instance(self):
        inst = IrrigationNetworkInstance.random(n_fields=4, seed=42)
        assert inst.n_nodes >= 7  # source + 2 junctions + 4 fields + sink
        assert len(inst.field_nodes) == 4


class TestMSTAnalysis:
    """Test MST backbone computation."""

    def test_mst_cost_positive(self):
        inst = IrrigationNetworkInstance.standard_farm()
        sol = _solver_mod.solve_irrigation_network(inst)
        assert sol.mst_cost > 0

    def test_mst_has_n_minus_1_edges(self):
        inst = IrrigationNetworkInstance.standard_farm()
        sol = _solver_mod.solve_irrigation_network(inst)
        assert len(sol.mst_edges) == inst.n_nodes - 1

    def test_mst_edges_have_costs(self):
        inst = IrrigationNetworkInstance.standard_farm()
        sol = _solver_mod.solve_irrigation_network(inst)
        for u, v, cost in sol.mst_edges:
            assert cost > 0
            assert 0 <= u < inst.n_nodes
            assert 0 <= v < inst.n_nodes

    def test_mst_connects_all_nodes(self):
        inst = IrrigationNetworkInstance.standard_farm()
        sol = _solver_mod.solve_irrigation_network(inst)
        # Check that all nodes are covered by MST edges
        covered = set()
        for u, v, _ in sol.mst_edges:
            covered.add(u)
            covered.add(v)
        assert covered == set(range(inst.n_nodes))


class TestMaxFlowAnalysis:
    """Test max flow computation."""

    def test_max_flow_positive(self):
        inst = IrrigationNetworkInstance.standard_farm()
        sol = _solver_mod.solve_irrigation_network(inst)
        assert sol.max_flow > 0

    def test_min_cut_partitions(self):
        inst = IrrigationNetworkInstance.standard_farm()
        sol = _solver_mod.solve_irrigation_network(inst)
        s_set, t_set = sol.min_cut
        assert inst.source in s_set
        assert inst.sink in t_set
        # Union should cover all nodes
        all_nodes = set(s_set) | set(t_set)
        assert all_nodes == set(range(inst.n_nodes))

    def test_min_cut_no_overlap(self):
        inst = IrrigationNetworkInstance.standard_farm()
        sol = _solver_mod.solve_irrigation_network(inst)
        s_set, t_set = sol.min_cut
        assert len(set(s_set) & set(t_set)) == 0


class TestShortestPathAnalysis:
    """Test shortest path computation."""

    def test_paths_to_all_fields(self):
        inst = IrrigationNetworkInstance.standard_farm()
        sol = _solver_mod.solve_irrigation_network(inst)
        for field_id in inst.field_nodes:
            assert field_id in sol.shortest_paths

    def test_path_distances_positive(self):
        inst = IrrigationNetworkInstance.standard_farm()
        sol = _solver_mod.solve_irrigation_network(inst)
        for field_id, (path, dist) in sol.shortest_paths.items():
            assert dist > 0

    def test_paths_start_at_source(self):
        inst = IrrigationNetworkInstance.standard_farm()
        sol = _solver_mod.solve_irrigation_network(inst)
        for field_id, (path, dist) in sol.shortest_paths.items():
            if path:
                assert path[0] == inst.source

    def test_paths_end_at_target(self):
        inst = IrrigationNetworkInstance.standard_farm()
        sol = _solver_mod.solve_irrigation_network(inst)
        for field_id, (path, dist) in sol.shortest_paths.items():
            if path:
                assert path[-1] == field_id

    def test_path_lengths_consistent(self):
        inst = IrrigationNetworkInstance.standard_farm()
        sol = _solver_mod.solve_irrigation_network(inst)
        # Each path should have at least 2 nodes (source + target)
        for field_id, (path, dist) in sol.shortest_paths.items():
            assert len(path) >= 2


class TestIntegration:
    """Cross-algorithm verification."""

    def test_solve_returns_solution(self):
        inst = IrrigationNetworkInstance.standard_farm()
        sol = _solver_mod.solve_irrigation_network(inst)
        assert type(sol).__name__ == "IrrigationNetworkSolution"

    def test_random_instance_solvable(self):
        inst = IrrigationNetworkInstance.random(n_fields=4, seed=42)
        sol = _solver_mod.solve_irrigation_network(inst)
        assert sol.mst_cost > 0
        assert sol.max_flow > 0

    def test_repr(self):
        inst = IrrigationNetworkInstance.standard_farm()
        sol = _solver_mod.solve_irrigation_network(inst)
        r = repr(sol)
        assert "IrrigationNetworkSolution" in r

    def test_mst_cost_less_than_total_pipe_cost(self):
        inst = IrrigationNetworkInstance.standard_farm()
        sol = _solver_mod.solve_irrigation_network(inst)
        total_pipe_cost = sum(p.install_cost for p in inst.pipes)
        assert sol.mst_cost < total_pipe_cost
