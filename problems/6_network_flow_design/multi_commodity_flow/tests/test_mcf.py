"""Tests for Multi-Commodity Flow Problem."""
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

_instance_mod = _load_mod("mcf_inst", os.path.join(_base, "instance.py"))
_lp_mod = _load_mod("mcf_lp", os.path.join(_base, "exact", "lp_formulation.py"))

MultiCommodityFlowInstance = _instance_mod.MultiCommodityFlowInstance
Commodity = _instance_mod.Commodity
MultiCommodityFlowSolution = _instance_mod.MultiCommodityFlowSolution
solve_mcf_lp = _lp_mod.solve_mcf_lp


class TestMCFInstance:
    """Tests for MultiCommodityFlowInstance."""

    def test_random_creation(self):
        inst = MultiCommodityFlowInstance.random(n_nodes=5, n_edges=8, n_commodities=2)
        assert inst.n_nodes == 5
        assert len(inst.edges) <= 8
        assert len(inst.commodities) == 2

    def test_capacities_positive(self):
        inst = MultiCommodityFlowInstance.random()
        assert np.all(inst.capacities > 0)

    def test_commodity_source_sink_differ(self):
        inst = MultiCommodityFlowInstance.random()
        for comm in inst.commodities:
            assert comm.source != comm.sink


class TestMCFLP:
    """Tests for LP formulation."""

    def _simple_instance(self):
        """Two-node, one-edge, one-commodity instance."""
        return MultiCommodityFlowInstance(
            n_nodes=2,
            edges=[(0, 1)],
            capacities=np.array([10.0]),
            commodities=[Commodity(source=0, sink=1, demand=5.0)]
        )

    def test_simple_feasible(self):
        inst = self._simple_instance()
        sol = solve_mcf_lp(inst)
        assert sol.feasible

    def test_simple_flow_conservation(self):
        inst = self._simple_instance()
        sol = solve_mcf_lp(inst)
        # Flow on edge 0 for commodity 0 should be exactly 5
        assert abs(sol.flows[0][0] - 5.0) < 1e-6

    def test_capacity_respected(self):
        inst = MultiCommodityFlowInstance(
            n_nodes=2,
            edges=[(0, 1)],
            capacities=np.array([10.0]),
            commodities=[
                Commodity(source=0, sink=1, demand=4.0),
                Commodity(source=0, sink=1, demand=5.0),
            ]
        )
        sol = solve_mcf_lp(inst)
        assert sol.feasible
        total_on_edge = sum(sol.flows[k].get(0, 0.0) for k in range(2))
        assert total_on_edge <= 10.0 + 1e-6

    def test_infeasible_demand_exceeds_capacity(self):
        inst = MultiCommodityFlowInstance(
            n_nodes=2,
            edges=[(0, 1)],
            capacities=np.array([5.0]),
            commodities=[Commodity(source=0, sink=1, demand=10.0)]
        )
        sol = solve_mcf_lp(inst)
        assert not sol.feasible

    def test_multi_path_routing(self):
        """Two parallel paths from 0 to 2."""
        inst = MultiCommodityFlowInstance(
            n_nodes=3,
            edges=[(0, 1), (1, 2), (0, 2)],
            capacities=np.array([5.0, 5.0, 5.0]),
            commodities=[Commodity(source=0, sink=2, demand=8.0)]
        )
        sol = solve_mcf_lp(inst)
        assert sol.feasible

    def test_disconnected_infeasible(self):
        """No path from source to sink."""
        inst = MultiCommodityFlowInstance(
            n_nodes=3,
            edges=[(0, 1)],
            capacities=np.array([10.0]),
            commodities=[Commodity(source=0, sink=2, demand=5.0)]
        )
        sol = solve_mcf_lp(inst)
        assert not sol.feasible

    def test_multiple_commodities_different_paths(self):
        """Two commodities with different source-sinks."""
        inst = MultiCommodityFlowInstance(
            n_nodes=4,
            edges=[(0, 2), (1, 3), (0, 1), (2, 3)],
            capacities=np.array([10.0, 10.0, 10.0, 10.0]),
            commodities=[
                Commodity(source=0, sink=2, demand=3.0),
                Commodity(source=1, sink=3, demand=4.0),
            ]
        )
        sol = solve_mcf_lp(inst)
        assert sol.feasible

    def test_solution_repr(self):
        sol = MultiCommodityFlowSolution(flows={}, total_flow=0.0, feasible=True)
        assert "feasible=True" in repr(sol)

    def test_random_instance_solvable(self):
        """Random instance with generous capacity should often be feasible."""
        inst = MultiCommodityFlowInstance(
            n_nodes=4,
            edges=[(0, 1), (1, 2), (2, 3), (0, 2), (1, 3)],
            capacities=np.array([50.0, 50.0, 50.0, 50.0, 50.0]),
            commodities=[Commodity(source=0, sink=3, demand=5.0)]
        )
        sol = solve_mcf_lp(inst)
        assert sol.feasible
        assert sol.total_flow >= 5.0 - 1e-6
