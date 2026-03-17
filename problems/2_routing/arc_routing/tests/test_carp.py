"""Tests for Capacitated Arc Routing Problem (CARP)."""
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

_instance_mod = _load_mod("carp_inst", os.path.join(_base, "instance.py"))
_ps_mod = _load_mod("carp_ps", os.path.join(_base, "heuristics", "path_scanning.py"))

CARPInstance = _instance_mod.CARPInstance
CARPSolution = _instance_mod.CARPSolution
path_scanning = _ps_mod.path_scanning


class TestCARPInstance:
    """Tests for CARPInstance."""

    def test_random_creation(self):
        inst = CARPInstance.random()
        assert inst.n_nodes == 8
        assert inst.depot == 0

    def test_required_edges(self):
        inst = CARPInstance.random()
        assert inst.n_required > 0
        for idx in inst.required_edges:
            assert inst.demands[idx] > 0

    def test_shortest_paths(self):
        inst = CARPInstance(
            n_nodes=3,
            edges=[(0, 1), (1, 2)],
            costs=np.array([1.0, 2.0]),
            demands=np.array([3.0, 4.0]),
            depot=0, capacity=10.0
        )
        dist = inst.shortest_paths()
        assert dist[0][0] == 0.0
        assert dist[0][1] == 1.0
        assert dist[0][2] == 3.0  # via node 1


class TestPathScanning:
    """Tests for path_scanning heuristic."""

    def _simple_instance(self):
        """Simple triangle: 0-1-2, all required."""
        return CARPInstance(
            n_nodes=3,
            edges=[(0, 1), (0, 2), (1, 2)],
            costs=np.array([1.0, 2.0, 1.5]),
            demands=np.array([3.0, 4.0, 2.0]),
            depot=0, capacity=10.0
        )

    def test_all_required_served(self):
        inst = self._simple_instance()
        sol = path_scanning(inst)
        served = set()
        for route in sol.routes:
            for e_idx, _ in route:
                served.add(e_idx)
        required = set(inst.required_edges)
        assert served == required

    def test_feasible_solution(self):
        inst = self._simple_instance()
        sol = path_scanning(inst)
        assert sol.feasible

    def test_capacity_respected(self):
        inst = self._simple_instance()
        sol = path_scanning(inst)
        for route in sol.routes:
            load = sum(inst.demands[e_idx] for e_idx, _ in route)
            assert load <= inst.capacity + 1e-9

    def test_small_capacity_forces_multiple_routes(self):
        inst = CARPInstance(
            n_nodes=3,
            edges=[(0, 1), (0, 2), (1, 2)],
            costs=np.array([1.0, 2.0, 1.5]),
            demands=np.array([5.0, 5.0, 5.0]),
            depot=0, capacity=6.0
        )
        sol = path_scanning(inst)
        assert sol.feasible
        assert len(sol.routes) >= 3  # each edge needs own route

    def test_no_required_edges(self):
        inst = CARPInstance(
            n_nodes=3,
            edges=[(0, 1), (1, 2)],
            costs=np.array([1.0, 2.0]),
            demands=np.array([0.0, 0.0]),
            depot=0, capacity=10.0
        )
        sol = path_scanning(inst)
        assert sol.feasible
        assert len(sol.routes) == 0
        assert sol.total_cost == 0.0

    def test_single_required_edge(self):
        inst = CARPInstance(
            n_nodes=3,
            edges=[(0, 1), (1, 2)],
            costs=np.array([1.0, 2.0]),
            demands=np.array([3.0, 0.0]),
            depot=0, capacity=10.0
        )
        sol = path_scanning(inst)
        assert sol.feasible
        assert len(sol.routes) == 1

    def test_positive_total_cost(self):
        inst = self._simple_instance()
        sol = path_scanning(inst)
        assert sol.total_cost > 0.0

    def test_solution_repr(self):
        sol = CARPSolution(routes=[], total_cost=0.0, feasible=True)
        assert "feasible=True" in repr(sol)

    def test_random_instance(self):
        inst = CARPInstance.random()
        sol = path_scanning(inst)
        assert hasattr(sol, 'feasible') and hasattr(sol, 'routes')
        # All required edges should be served
        served = set()
        for route in sol.routes:
            for e_idx, _ in route:
                served.add(e_idx)
        assert served == set(inst.required_edges)
