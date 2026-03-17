"""Tests for Simple Assembly Line Balancing Problem (SALBP-1)."""
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

_instance_mod = _load_mod("salbp_inst", os.path.join(_base, "instance.py"))
_rpw_mod = _load_mod("salbp_rpw", os.path.join(_base, "heuristics", "rpw.py"))

SALBPInstance = _instance_mod.SALBPInstance
SALBPSolution = _instance_mod.SALBPSolution
rpw_heuristic = _rpw_mod.rpw_heuristic


class TestSALBPInstance:
    """Tests for SALBPInstance."""

    def test_random_creation(self):
        inst = SALBPInstance.random(n_tasks=8)
        assert inst.n_tasks == 8
        assert len(inst.processing_times) == 8

    def test_successors(self):
        inst = SALBPInstance(
            n_tasks=3, processing_times=np.array([1.0, 2.0, 3.0]),
            precedences=[(0, 1), (1, 2)], cycle_time=10.0
        )
        succ = inst.successors()
        assert succ[0] == [1]
        assert succ[1] == [2]
        assert succ[2] == []

    def test_predecessors(self):
        inst = SALBPInstance(
            n_tasks=3, processing_times=np.array([1.0, 2.0, 3.0]),
            precedences=[(0, 1), (1, 2)], cycle_time=10.0
        )
        pred = inst.predecessors()
        assert pred[0] == []
        assert pred[1] == [0]
        assert pred[2] == [1]


class TestRPW:
    """Tests for RPW heuristic."""

    def _chain_instance(self):
        """Linear chain: 0 -> 1 -> 2 -> 3."""
        return SALBPInstance(
            n_tasks=4,
            processing_times=np.array([3.0, 4.0, 2.0, 5.0]),
            precedences=[(0, 1), (1, 2), (2, 3)],
            cycle_time=10.0
        )

    def test_precedence_respected(self):
        inst = self._chain_instance()
        sol = rpw_heuristic(inst)
        for pred, succ in inst.precedences:
            assert sol.assignment[pred] <= sol.assignment[succ]

    def test_cycle_time_not_exceeded(self):
        inst = self._chain_instance()
        sol = rpw_heuristic(inst)
        for t in sol.station_times:
            assert t <= inst.cycle_time + 1e-9

    def test_feasible(self):
        inst = self._chain_instance()
        sol = rpw_heuristic(inst)
        assert sol.feasible

    def test_all_tasks_assigned(self):
        inst = self._chain_instance()
        sol = rpw_heuristic(inst)
        assert set(sol.assignment.keys()) == set(range(inst.n_tasks))

    def test_single_task(self):
        inst = SALBPInstance(
            n_tasks=1, processing_times=np.array([5.0]),
            precedences=[], cycle_time=10.0
        )
        sol = rpw_heuristic(inst)
        assert sol.n_stations == 1
        assert sol.feasible

    def test_each_task_own_station(self):
        """When cycle time is tight, each task may need its own station."""
        inst = SALBPInstance(
            n_tasks=3, processing_times=np.array([5.0, 5.0, 5.0]),
            precedences=[], cycle_time=5.0
        )
        sol = rpw_heuristic(inst)
        assert sol.n_stations == 3
        assert sol.feasible

    def test_no_precedences(self):
        inst = SALBPInstance(
            n_tasks=4,
            processing_times=np.array([2.0, 3.0, 4.0, 1.0]),
            precedences=[], cycle_time=10.0
        )
        sol = rpw_heuristic(inst)
        assert sol.feasible

    def test_solution_repr(self):
        sol = SALBPSolution(
            assignment={0: 0}, n_stations=1,
            station_times=[5.0], feasible=True
        )
        assert "n_stations=1" in repr(sol)
