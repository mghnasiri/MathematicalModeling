"""Tests for Fixed-Charge Network Design Problem."""
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

_instance_mod = _load_mod("nd_inst", os.path.join(_base, "instance.py"))
_greedy_mod = _load_mod("nd_greedy", os.path.join(_base, "heuristics", "greedy_open.py"))

NetworkDesignInstance = _instance_mod.NetworkDesignInstance
NetworkDesignSolution = _instance_mod.NetworkDesignSolution
greedy_open = _greedy_mod.greedy_open


class TestNetworkDesignInstance:
    """Tests for NetworkDesignInstance."""

    def test_random_creation(self):
        inst = NetworkDesignInstance.random(n_nodes=5, n_edges=8)
        assert inst.n_nodes == 5
        assert len(inst.potential_edges) <= 8

    def test_balanced_demands(self):
        inst = NetworkDesignInstance.random()
        # Supply + demand should balance (approximately)
        assert abs(np.sum(inst.demands)) < 1e-9


class TestGreedyOpen:
    """Tests for greedy_open heuristic."""

    def _simple_instance(self):
        """Simple 3-node instance: 0 -> 1 -> 2, supply at 0, demand at 2."""
        return NetworkDesignInstance(
            n_nodes=3,
            potential_edges=[(0, 1), (1, 2)],
            fixed_costs=np.array([10.0, 10.0]),
            unit_costs=np.array([1.0, 1.0]),
            edge_capacities=np.array([20.0, 20.0]),
            demands=np.array([-10.0, 0.0, 10.0])
        )

    def test_simple_feasible(self):
        inst = self._simple_instance()
        sol = greedy_open(inst)
        assert sol.feasible

    def test_simple_both_edges_open(self):
        inst = self._simple_instance()
        sol = greedy_open(inst)
        assert len(sol.open_edges) == 2

    def test_cost_computation(self):
        inst = self._simple_instance()
        sol = greedy_open(inst)
        assert sol.fixed_cost == pytest.approx(20.0)
        assert sol.variable_cost == pytest.approx(20.0)  # 10 units * 1.0/unit * 2 edges
        assert sol.total_cost == pytest.approx(40.0)

    def test_no_demand_no_edges(self):
        inst = NetworkDesignInstance(
            n_nodes=3,
            potential_edges=[(0, 1), (1, 2)],
            fixed_costs=np.array([10.0, 10.0]),
            unit_costs=np.array([1.0, 1.0]),
            edge_capacities=np.array([20.0, 20.0]),
            demands=np.array([0.0, 0.0, 0.0])
        )
        sol = greedy_open(inst)
        assert sol.feasible
        assert sol.total_cost == 0.0

    def test_direct_edge_preferred(self):
        """Direct edge should be cheaper than indirect path."""
        inst = NetworkDesignInstance(
            n_nodes=3,
            potential_edges=[(0, 2), (0, 1), (1, 2)],
            fixed_costs=np.array([5.0, 10.0, 10.0]),
            unit_costs=np.array([1.0, 1.0, 1.0]),
            edge_capacities=np.array([20.0, 20.0, 20.0]),
            demands=np.array([-5.0, 0.0, 5.0])
        )
        sol = greedy_open(inst)
        assert sol.feasible
        # Direct edge (0,2) has fixed cost 5, cheaper than opening two edges
        assert 0 in sol.open_edges

    def test_capacity_constraint(self):
        """Edge capacity limits flow."""
        inst = NetworkDesignInstance(
            n_nodes=2,
            potential_edges=[(0, 1)],
            fixed_costs=np.array([10.0]),
            unit_costs=np.array([1.0]),
            edge_capacities=np.array([5.0]),
            demands=np.array([-10.0, 10.0])
        )
        sol = greedy_open(inst)
        # Cannot route 10 units through capacity 5 edge
        assert not sol.feasible

    def test_random_instance(self):
        inst = NetworkDesignInstance.random()
        sol = greedy_open(inst)
        # Solution should be returned (may or may not be feasible)
        assert hasattr(sol, 'feasible')
        assert hasattr(sol, 'total_cost')

    def test_solution_repr(self):
        sol = NetworkDesignSolution(
            open_edges={0}, flows={0: 5.0}, total_cost=15.0,
            fixed_cost=10.0, variable_cost=5.0, feasible=True
        )
        assert "feasible=True" in repr(sol)
