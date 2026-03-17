"""Tests for Stochastic Vehicle Routing Problem."""
from __future__ import annotations

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
_inst_mod = _load_mod("svrp_instance", os.path.join(_base, "instance.py"))
_cw_mod = _load_mod("svrp_cw", os.path.join(_base, "heuristics", "chance_constrained_cw.py"))
_sa_mod = _load_mod("svrp_sa", os.path.join(_base, "metaheuristics", "simulated_annealing.py"))

StochasticVRPInstance = _inst_mod.StochasticVRPInstance
StochasticVRPSolution = _inst_mod.StochasticVRPSolution
chance_constrained_savings = _cw_mod.chance_constrained_savings
mean_demand_savings = _cw_mod.mean_demand_savings
simulated_annealing = _sa_mod.simulated_annealing


def _make_simple():
    return StochasticVRPInstance(
        n_customers=5,
        coordinates=np.array([
            [50, 50],   # depot
            [20, 70], [80, 70], [20, 30], [80, 30], [50, 90],
        ], dtype=float),
        demand_scenarios=np.array([
            [8, 6, 10, 7, 5],
            [10, 8, 12, 9, 7],
            [6, 4, 8, 5, 3],
            [12, 10, 14, 11, 9],
        ], dtype=float),
        vehicle_capacity=25.0,
        n_vehicles=3,
        alpha=0.3,
    )


class TestStochasticVRPInstance:

    def test_creation(self):
        inst = _make_simple()
        assert inst.n_customers == 5
        assert inst.n_scenarios == 4

    def test_distance_symmetric(self):
        inst = _make_simple()
        assert inst.distance(0, 1) == pytest.approx(inst.distance(1, 0))

    def test_route_distance(self):
        inst = _make_simple()
        d = inst.route_distance([1, 2])
        assert d > 0

    def test_overflow_probability_bounded(self):
        inst = _make_simple()
        prob = inst.route_overflow_probability([1, 2, 3])
        assert 0.0 <= prob <= 1.0

    def test_random_instance(self):
        inst = StochasticVRPInstance.random(n_customers=8, n_scenarios=10)
        assert inst.n_customers == 8
        assert inst.n_scenarios == 10

    def test_mean_demands(self):
        inst = _make_simple()
        md = inst.mean_demands
        assert len(md) == 5
        assert all(m > 0 for m in md)


class TestClarkeWrightHeuristics:

    def test_cc_savings_covers_all(self):
        inst = _make_simple()
        sol = chance_constrained_savings(inst)
        served = set()
        for r in sol.routes:
            served.update(r)
        assert served == {1, 2, 3, 4, 5}

    def test_cc_savings_distance_positive(self):
        inst = _make_simple()
        sol = chance_constrained_savings(inst)
        assert sol.total_distance > 0

    def test_mean_demand_covers_all(self):
        inst = _make_simple()
        sol = mean_demand_savings(inst)
        served = set()
        for r in sol.routes:
            served.update(r)
        assert served == {1, 2, 3, 4, 5}

    def test_mean_demand_random(self):
        inst = StochasticVRPInstance.random(n_customers=8, n_scenarios=15)
        sol = mean_demand_savings(inst)
        assert sol.n_routes > 0
        assert sol.total_distance > 0


class TestSimulatedAnnealing:

    def test_sa_covers_all(self):
        inst = _make_simple()
        sol = simulated_annealing(inst, max_iterations=1000, seed=42)
        served = set()
        for r in sol.routes:
            served.update(r)
        assert served == {1, 2, 3, 4, 5}

    def test_sa_distance_positive(self):
        inst = _make_simple()
        sol = simulated_annealing(inst, max_iterations=1000, seed=42)
        assert sol.total_distance > 0

    def test_sa_deterministic(self):
        inst = _make_simple()
        sol1 = simulated_annealing(inst, seed=99, max_iterations=500)
        sol2 = simulated_annealing(inst, seed=99, max_iterations=500)
        assert sol1.total_distance == pytest.approx(sol2.total_distance)
