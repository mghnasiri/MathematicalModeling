"""Tests for the Bi-Objective 0-1 Knapsack problem."""
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
_instance_mod = _load_mod("bok_instance", os.path.join(_base, "instance.py"))
_ec_mod = _load_mod(
    "bok_ec", os.path.join(_base, "heuristics", "epsilon_constraint.py")
)

BiObjectiveKnapsackInstance = _instance_mod.BiObjectiveKnapsackInstance
BiObjectiveKnapsackSolution = _instance_mod.BiObjectiveKnapsackSolution
epsilon_constraint = _ec_mod.epsilon_constraint
weighted_sum_pareto = _ec_mod.weighted_sum_pareto


class TestBiObjectiveKnapsackInstance:
    """Tests for instance construction."""

    def test_random_instance(self):
        inst = BiObjectiveKnapsackInstance.random(n=8, seed=1)
        assert inst.n == 8
        assert len(inst.weights) == 8
        assert len(inst.values1) == 8
        assert len(inst.values2) == 8

    def test_feasibility_check(self):
        inst = BiObjectiveKnapsackInstance(
            n=3, weights=np.array([5, 5, 5]),
            values1=np.array([10, 20, 30]),
            values2=np.array([30, 20, 10]),
            capacity=10,
        )
        assert inst.is_feasible(np.array([1, 1, 0]))
        assert not inst.is_feasible(np.array([1, 1, 1]))

    def test_evaluate(self):
        inst = BiObjectiveKnapsackInstance(
            n=3, weights=np.array([1, 1, 1]),
            values1=np.array([10, 20, 30]),
            values2=np.array([30, 20, 10]),
            capacity=3,
        )
        v1, v2 = inst.evaluate(np.array([1, 0, 1]))
        assert v1 == 40.0
        assert v2 == 40.0


class TestEpsilonConstraint:
    """Tests for epsilon-constraint method."""

    def test_finds_solutions(self):
        inst = BiObjectiveKnapsackInstance.random(n=6, seed=10)
        sol = epsilon_constraint(inst, n_points=10)
        assert sol.n_solutions > 0

    def test_pareto_front_non_dominated(self):
        inst = BiObjectiveKnapsackInstance.random(n=6, seed=11)
        sol = epsilon_constraint(inst, n_points=10)
        for i, p1 in enumerate(sol.pareto_front):
            for j, p2 in enumerate(sol.pareto_front):
                if i != j:
                    # No point should dominate another
                    assert not (p2[0] >= p1[0] and p2[1] >= p1[1]
                                and (p2[0] > p1[0] or p2[1] > p1[1]))

    def test_solutions_feasible(self):
        inst = BiObjectiveKnapsackInstance.random(n=6, seed=12)
        sol = epsilon_constraint(inst, n_points=10)
        for x in sol.pareto_solutions:
            assert inst.is_feasible(x)

    def test_objectives_match_solutions(self):
        inst = BiObjectiveKnapsackInstance.random(n=6, seed=13)
        sol = epsilon_constraint(inst, n_points=10)
        for pt, x in zip(sol.pareto_front, sol.pareto_solutions):
            v1, v2 = inst.evaluate(x)
            assert abs(v1 - pt[0]) < 1e-6
            assert abs(v2 - pt[1]) < 1e-6

    def test_handcrafted_two_items(self):
        """Two conflicting items: each should appear in Pareto front."""
        inst = BiObjectiveKnapsackInstance(
            n=2, weights=np.array([5, 5]),
            values1=np.array([100, 1]),
            values2=np.array([1, 100]),
            capacity=5,
        )
        sol = epsilon_constraint(inst, n_points=10)
        assert sol.n_solutions >= 2


class TestWeightedSumPareto:
    """Tests for weighted-sum scalarization."""

    def test_finds_solutions(self):
        inst = BiObjectiveKnapsackInstance.random(n=8, seed=20)
        sol = weighted_sum_pareto(inst, n_weights=11)
        assert sol.n_solutions > 0

    def test_solutions_feasible(self):
        inst = BiObjectiveKnapsackInstance.random(n=8, seed=21)
        sol = weighted_sum_pareto(inst, n_weights=11)
        for x in sol.pareto_solutions:
            assert inst.is_feasible(x)

    def test_front_sorted_by_v1(self):
        inst = BiObjectiveKnapsackInstance.random(n=8, seed=22)
        sol = weighted_sum_pareto(inst, n_weights=11)
        for i in range(len(sol.pareto_front) - 1):
            assert sol.pareto_front[i][0] <= sol.pareto_front[i + 1][0]

    def test_n_solutions_matches_front(self):
        inst = BiObjectiveKnapsackInstance.random(n=6, seed=23)
        sol = weighted_sum_pareto(inst, n_weights=11)
        assert sol.n_solutions == len(sol.pareto_front)
        assert sol.n_solutions == len(sol.pareto_solutions)
