"""Tests for Chance-Constrained Facility Location."""
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
_inst_mod = _load_mod("ccfl_instance", os.path.join(_base, "instance.py"))
_greedy_mod = _load_mod("ccfl_greedy", os.path.join(_base, "heuristics", "greedy_ccfl.py"))
_sa_mod = _load_mod("ccfl_sa", os.path.join(_base, "metaheuristics", "simulated_annealing.py"))

CCFLInstance = _inst_mod.CCFLInstance
CCFLSolution = _inst_mod.CCFLSolution
greedy_open = _greedy_mod.greedy_open
mean_demand_greedy = _greedy_mod.mean_demand_greedy
simulated_annealing = _sa_mod.simulated_annealing


def _make_simple():
    return CCFLInstance(
        n_facilities=3,
        n_customers=4,
        fixed_costs=np.array([100, 80, 120]),
        assignment_costs=np.array([
            [5, 8, 3, 7],
            [4, 6, 9, 2],
            [7, 3, 5, 4],
        ]),
        capacities=np.array([30, 30, 30]),
        demand_scenarios=np.array([
            [8, 6, 7, 5],
            [10, 8, 9, 7],
            [6, 4, 5, 3],
            [12, 10, 11, 9],
        ]),
        alpha=0.3,
    )


class TestCCFLInstance:

    def test_creation(self):
        inst = _make_simple()
        assert inst.n_facilities == 3
        assert inst.n_customers == 4
        assert inst.n_scenarios == 4

    def test_violation_probability_empty(self):
        inst = _make_simple()
        assert inst.capacity_violation_prob(0, []) == 0.0

    def test_violation_probability_bounded(self):
        inst = _make_simple()
        viol = inst.capacity_violation_prob(0, [0, 1, 2, 3])
        assert 0.0 <= viol <= 1.0

    def test_random_instance(self):
        inst = CCFLInstance.random(n_facilities=4, n_customers=6)
        assert inst.n_facilities == 4
        assert inst.n_customers == 6


class TestGreedyHeuristics:

    def test_greedy_open_all_assigned(self):
        inst = _make_simple()
        sol = greedy_open(inst)
        assert len(sol.open_facilities) > 0
        # All customers assigned to open facilities
        for j in range(inst.n_customers):
            assert sol.assignments[j] in sol.open_facilities

    def test_greedy_open_cost_positive(self):
        inst = _make_simple()
        sol = greedy_open(inst)
        assert sol.total_cost > 0

    def test_mean_demand_greedy_feasible(self):
        inst = _make_simple()
        sol = mean_demand_greedy(inst)
        assert len(sol.open_facilities) > 0
        assert sol.total_cost > 0

    def test_random_instance(self):
        inst = CCFLInstance.random(n_facilities=5, n_customers=8, seed=42)
        sol = greedy_open(inst)
        assert len(sol.open_facilities) > 0


class TestSimulatedAnnealing:

    def test_sa_simple(self):
        inst = _make_simple()
        sol = simulated_annealing(inst, max_iterations=1000, seed=42)
        assert len(sol.open_facilities) > 0
        assert sol.total_cost > 0

    def test_sa_all_assigned(self):
        inst = _make_simple()
        sol = simulated_annealing(inst, max_iterations=1000, seed=42)
        for j in range(inst.n_customers):
            assert sol.assignments[j] in sol.open_facilities

    def test_sa_deterministic(self):
        inst = _make_simple()
        sol1 = simulated_annealing(inst, seed=99, max_iterations=500)
        sol2 = simulated_annealing(inst, seed=99, max_iterations=500)
        assert sol1.total_cost == pytest.approx(sol2.total_cost)
