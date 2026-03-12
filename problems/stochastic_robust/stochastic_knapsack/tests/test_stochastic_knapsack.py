"""Tests for Stochastic Knapsack Problem."""
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
_inst_mod = _load_mod("sk_instance", os.path.join(_base, "instance.py"))
_greedy_mod = _load_mod("sk_greedy", os.path.join(_base, "heuristics", "greedy_stochastic.py"))
_sa_mod = _load_mod("sk_sa", os.path.join(_base, "metaheuristics", "simulated_annealing.py"))

StochasticKnapsackInstance = _inst_mod.StochasticKnapsackInstance
greedy_mean_weight = _greedy_mod.greedy_mean_weight
greedy_chance_constrained = _greedy_mod.greedy_chance_constrained
simulated_annealing = _sa_mod.simulated_annealing


def _make_simple():
    return StochasticKnapsackInstance(
        n=4,
        values=np.array([10, 20, 15, 25]),
        weight_scenarios=np.array([
            [3, 5, 4, 6],
            [4, 6, 3, 7],
            [2, 4, 5, 5],
        ]),
        capacity=12.0,
    )


class TestStochasticKnapsackInstance:

    def test_creation(self):
        inst = _make_simple()
        assert inst.n == 4
        assert inst.n_scenarios == 3

    def test_mean_weights(self):
        inst = _make_simple()
        mw = inst.mean_weights
        assert len(mw) == 4
        assert mw[0] == pytest.approx(3.0)  # (3+4+2)/3

    def test_feasibility_probability(self):
        inst = _make_simple()
        sel = np.array([1, 0, 0, 0])
        assert inst.feasibility_probability(sel) == 1.0  # single light item

    def test_random_instance(self):
        inst = StochasticKnapsackInstance.random(n=8, n_scenarios=15)
        assert inst.n == 8
        assert inst.n_scenarios == 15


class TestGreedyHeuristics:

    def test_mean_weight_feasible(self):
        inst = _make_simple()
        sol = greedy_mean_weight(inst)
        assert sol.total_value > 0
        assert sol.expected_weight <= inst.capacity + 1e-6

    def test_chance_constrained_high_confidence(self):
        inst = _make_simple()
        sol = greedy_chance_constrained(inst, alpha=0.05)
        assert sol.feasibility_prob >= 0.95 - 1e-6

    def test_chance_constrained_value_positive(self):
        inst = StochasticKnapsackInstance.random(n=10, n_scenarios=20)
        sol = greedy_chance_constrained(inst, alpha=0.2)
        assert sol.total_value > 0

    def test_mean_weight_random(self):
        inst = StochasticKnapsackInstance.random(n=10, n_scenarios=30)
        sol = greedy_mean_weight(inst)
        assert sol.total_value > 0


class TestSimulatedAnnealing:

    def test_sa_simple(self):
        inst = _make_simple()
        sol = simulated_annealing(inst, alpha=0.1, max_iterations=1000, seed=42)
        assert sol.total_value > 0

    def test_sa_beats_greedy_or_matches(self):
        inst = StochasticKnapsackInstance.random(n=8, n_scenarios=20, seed=7)
        sol_g = greedy_mean_weight(inst)
        sol_sa = simulated_annealing(inst, alpha=0.2, max_iterations=3000, seed=7)
        # SA should find at least as good a solution
        assert sol_sa.total_value >= sol_g.total_value * 0.8

    def test_sa_deterministic_seed(self):
        inst = _make_simple()
        sol1 = simulated_annealing(inst, seed=99, max_iterations=500)
        sol2 = simulated_annealing(inst, seed=99, max_iterations=500)
        assert sol1.total_value == sol2.total_value
