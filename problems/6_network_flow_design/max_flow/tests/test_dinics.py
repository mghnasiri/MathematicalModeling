"""
Tests for Dinic's Algorithm (Max Flow).

Run: python -m pytest problems/location_network/max_flow/tests/test_dinics.py -v
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


_inst_mod = _load_module("mf_inst_dinic_test", os.path.join(_base_dir, "instance.py"))
_dinic_mod = _load_module("mf_dinic_test", os.path.join(_base_dir, "exact", "dinics.py"))
_ek_mod = _load_module("mf_ek_dinic_test", os.path.join(_base_dir, "exact", "edmonds_karp.py"))

MaxFlowInstance = _inst_mod.MaxFlowInstance
MaxFlowSolution = _inst_mod.MaxFlowSolution
validate_solution = _inst_mod.validate_solution
simple_flow_4 = _inst_mod.simple_flow_4
two_path_flow = _inst_mod.two_path_flow
dinics = _dinic_mod.dinics
edmonds_karp = _ek_mod.edmonds_karp


class TestDinics:
    """Test Dinic's algorithm for maximum flow."""

    def test_simple4_optimal(self):
        inst = simple_flow_4()
        sol = dinics(inst)
        assert abs(sol.max_flow - 26.0) < 1e-6

    def test_simple4_valid(self):
        inst = simple_flow_4()
        sol = dinics(inst)
        is_valid, errors = validate_solution(inst, sol)
        assert is_valid, f"Invalid: {errors}"

    def test_two_path_optimal(self):
        inst = two_path_flow()
        sol = dinics(inst)
        assert abs(sol.max_flow - 5.0) < 1e-6

    def test_two_path_valid(self):
        inst = two_path_flow()
        sol = dinics(inst)
        is_valid, errors = validate_solution(inst, sol)
        assert is_valid, f"Invalid: {errors}"

    def test_matches_edmonds_karp(self):
        """Dinic's should produce same max flow as Edmonds-Karp."""
        inst = MaxFlowInstance.random(n=8, density=0.4, seed=42)
        sol_dinic = dinics(inst)
        sol_ek = edmonds_karp(inst)
        assert abs(sol_dinic.max_flow - sol_ek.max_flow) < 1e-6

    def test_min_cut_exists(self):
        inst = simple_flow_4()
        sol = dinics(inst)
        assert sol.min_cut is not None
        s_side, t_side = sol.min_cut
        assert inst.source in s_side
        assert inst.sink in t_side
        assert sorted(s_side + t_side) == list(range(inst.n))

    def test_no_flow_possible(self):
        """When no path from source to sink, max flow = 0."""
        inst = MaxFlowInstance.from_edges(
            3, 0, 2,
            [(0, 1, 10)],  # No edge from 1 to 2
            name="no_path",
        )
        sol = dinics(inst)
        assert abs(sol.max_flow - 0.0) < 1e-6

    def test_single_edge(self):
        inst = MaxFlowInstance.from_edges(
            2, 0, 1,
            [(0, 1, 7)],
            name="single_edge",
        )
        sol = dinics(inst)
        assert abs(sol.max_flow - 7.0) < 1e-6

    def test_random_instance_valid(self):
        inst = MaxFlowInstance.random(n=10, density=0.3, seed=42)
        sol = dinics(inst)
        is_valid, errors = validate_solution(inst, sol)
        assert is_valid, f"Invalid: {errors}"

    def test_parallel_edges(self):
        """Multiple edges between same nodes should sum capacities."""
        inst = MaxFlowInstance.from_edges(
            2, 0, 1,
            [(0, 1, 3), (0, 1, 5)],
            name="parallel",
        )
        sol = dinics(inst)
        assert abs(sol.max_flow - 8.0) < 1e-6
