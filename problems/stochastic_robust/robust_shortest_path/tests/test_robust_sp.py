"""Tests for Robust Shortest Path Problem."""
from __future__ import annotations

import sys
import os
import importlib.util

import numpy as np
import pytest

# Load modules via importlib
def _load_mod(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_base = os.path.join(os.path.dirname(__file__), "..")
_inst_mod = _load_mod("rsp_instance", os.path.join(_base, "instance.py"))
_mc_mod = _load_mod("rsp_minmax", os.path.join(_base, "exact", "minmax_cost.py"))
_mr_mod = _load_mod("rsp_regret", os.path.join(_base, "heuristics", "minmax_regret.py"))

RobustSPInstance = _inst_mod.RobustSPInstance
RobustSPSolution = _inst_mod.RobustSPSolution
minmax_cost_enumeration = _mc_mod.minmax_cost_enumeration
minmax_cost_label_setting = _mc_mod.minmax_cost_label_setting
minmax_regret_enumeration = _mr_mod.minmax_regret_enumeration
midpoint_scenario = _mr_mod.midpoint_scenario


def _make_simple_instance():
    """Simple 4-node graph with 2 scenarios.

    Graph: 0 -> 1 -> 3 and 0 -> 2 -> 3
    Scenario 0: path 0-1-3 costs 2+3=5, path 0-2-3 costs 4+2=6
    Scenario 1: path 0-1-3 costs 5+1=6, path 0-2-3 costs 2+3=5
    """
    return RobustSPInstance(
        n_nodes=4,
        edges=[(0, 1), (0, 2), (1, 3), (2, 3)],
        weight_scenarios=np.array([
            [2.0, 4.0, 3.0, 2.0],  # scenario 0
            [5.0, 2.0, 1.0, 3.0],  # scenario 1
        ]),
        source=0,
        target=3,
    )


class TestRobustSPInstance:
    """Test instance creation and helpers."""

    def test_path_cost(self):
        inst = _make_simple_instance()
        # Path 0-1-3 under scenario 0: 2+3=5
        assert inst.path_cost([0, 1, 3], 0) == pytest.approx(5.0)
        # Path 0-2-3 under scenario 1: 2+3=5
        assert inst.path_cost([0, 2, 3], 1) == pytest.approx(5.0)

    def test_max_cost(self):
        inst = _make_simple_instance()
        # Path 0-1-3: max(5, 6) = 6
        assert inst.max_cost([0, 1, 3]) == pytest.approx(6.0)
        # Path 0-2-3: max(6, 5) = 6
        assert inst.max_cost([0, 2, 3]) == pytest.approx(6.0)

    def test_random_instance(self):
        inst = RobustSPInstance.random(n_nodes=6, n_scenarios=4, seed=42)
        assert inst.n_nodes == 6
        assert inst.n_scenarios == 4
        assert inst.n_edges > 0

    def test_adjacency_list(self):
        inst = _make_simple_instance()
        adj = inst.adjacency_list(0)
        assert len(adj[0]) == 2  # edges to 1 and 2


class TestMinMaxCost:
    """Test min-max cost algorithms."""

    def test_enumeration_simple(self):
        inst = _make_simple_instance()
        sol = minmax_cost_enumeration(inst)
        assert len(sol.path) >= 2
        assert sol.path[0] == 0
        assert sol.path[-1] == 3
        assert sol.max_cost <= 6.0 + 1e-6

    def test_label_setting_simple(self):
        inst = _make_simple_instance()
        sol = minmax_cost_label_setting(inst)
        assert len(sol.path) >= 2
        assert sol.path[0] == 0
        assert sol.path[-1] == 3
        assert sol.max_cost <= 6.0 + 1e-6

    def test_label_setting_optimal(self):
        """Both paths have max cost 6, so either is optimal."""
        inst = _make_simple_instance()
        sol = minmax_cost_label_setting(inst)
        assert sol.max_cost == pytest.approx(6.0)

    def test_enumeration_random(self):
        inst = RobustSPInstance.random(n_nodes=6, n_scenarios=3, seed=42)
        sol = minmax_cost_enumeration(inst)
        if sol.path:
            assert sol.path[0] == inst.source
            assert sol.path[-1] == inst.target
            assert sol.max_cost < float("inf")

    def test_label_setting_vs_enumeration(self):
        """Label-setting should be at least as good as enumeration."""
        inst = RobustSPInstance.random(n_nodes=5, n_scenarios=3, seed=7)
        sol_enum = minmax_cost_enumeration(inst)
        sol_label = minmax_cost_label_setting(inst)
        if sol_label.path and sol_enum.path:
            assert sol_label.max_cost <= sol_enum.max_cost + 1e-6


class TestMinMaxRegret:
    """Test min-max regret algorithms."""

    def test_regret_enumeration_simple(self):
        inst = _make_simple_instance()
        sol = minmax_regret_enumeration(inst)
        assert len(sol.path) >= 2
        assert sol.max_regret is not None
        assert sol.max_regret >= 0

    def test_regret_nonnegative(self):
        inst = _make_simple_instance()
        sol = minmax_regret_enumeration(inst)
        # Regret should be >= 0 (path cost - optimal cost)
        assert sol.max_regret >= -1e-6

    def test_midpoint_heuristic(self):
        inst = _make_simple_instance()
        sol = midpoint_scenario(inst)
        assert len(sol.path) >= 2
        assert sol.max_regret is not None

    def test_regret_random(self):
        inst = RobustSPInstance.random(n_nodes=6, n_scenarios=4, seed=42)
        sol = minmax_regret_enumeration(inst)
        if sol.path:
            assert sol.max_regret >= -1e-6
