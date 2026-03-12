"""Tests for the Multi-Objective TSP problem."""
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
_instance_mod = _load_mod("motsp_instance", os.path.join(_base, "instance.py"))
_ws_mod = _load_mod(
    "motsp_ws", os.path.join(_base, "heuristics", "weighted_sum_nn.py")
)

MultiObjectiveTSPInstance = _instance_mod.MultiObjectiveTSPInstance
MultiObjectiveTSPSolution = _instance_mod.MultiObjectiveTSPSolution
weighted_sum_nn = _ws_mod.weighted_sum_nn


class TestMultiObjectiveTSPInstance:
    """Tests for instance construction."""

    def test_random_instance(self):
        inst = MultiObjectiveTSPInstance.random(n=5, n_objectives=2, seed=1)
        assert inst.n == 5
        assert inst.n_objectives == 2
        assert len(inst.distance_matrices) == 2
        assert inst.distance_matrices[0].shape == (5, 5)

    def test_tour_cost_valid(self):
        inst = MultiObjectiveTSPInstance.random(n=4, n_objectives=2, seed=2)
        tour = [0, 1, 2, 3]
        c0 = inst.tour_cost(tour, 0)
        c1 = inst.tour_cost(tour, 1)
        assert c0 > 0
        assert c1 > 0

    def test_evaluate_returns_all_objectives(self):
        inst = MultiObjectiveTSPInstance.random(n=4, n_objectives=3, seed=3)
        tour = [0, 1, 2, 3]
        costs = inst.evaluate(tour)
        assert len(costs) == 3
        assert all(c > 0 for c in costs)


class TestWeightedSumNN:
    """Tests for weighted-sum nearest neighbor."""

    def test_finds_solutions(self):
        inst = MultiObjectiveTSPInstance.random(n=6, n_objectives=2, seed=10)
        sol = weighted_sum_nn(inst, n_weights=5)
        assert sol.n_solutions > 0

    def test_tours_are_valid_permutations(self):
        inst = MultiObjectiveTSPInstance.random(n=6, n_objectives=2, seed=11)
        sol = weighted_sum_nn(inst, n_weights=5)
        for tour in sol.pareto_tours:
            assert sorted(tour) == list(range(inst.n))

    def test_front_non_dominated(self):
        inst = MultiObjectiveTSPInstance.random(n=6, n_objectives=2, seed=12)
        sol = weighted_sum_nn(inst, n_weights=7)
        for i, p1 in enumerate(sol.pareto_front):
            for j, p2 in enumerate(sol.pareto_front):
                if i != j:
                    assert not (all(p2[k] <= p1[k] for k in range(2))
                                and any(p2[k] < p1[k] for k in range(2)))

    def test_costs_match_tours(self):
        inst = MultiObjectiveTSPInstance.random(n=5, n_objectives=2, seed=13)
        sol = weighted_sum_nn(inst, n_weights=5)
        for pt, tour in zip(sol.pareto_front, sol.pareto_tours):
            actual = inst.evaluate(tour)
            for k in range(inst.n_objectives):
                assert abs(pt[k] - actual[k]) < 1e-6

    def test_sorted_by_first_objective(self):
        inst = MultiObjectiveTSPInstance.random(n=6, n_objectives=2, seed=14)
        sol = weighted_sum_nn(inst, n_weights=7)
        for i in range(len(sol.pareto_front) - 1):
            assert sol.pareto_front[i][0] <= sol.pareto_front[i + 1][0] + 1e-9

    def test_n_solutions_consistent(self):
        inst = MultiObjectiveTSPInstance.random(n=6, n_objectives=2, seed=15)
        sol = weighted_sum_nn(inst, n_weights=5)
        assert sol.n_solutions == len(sol.pareto_front)
        assert sol.n_solutions == len(sol.pareto_tours)

    def test_small_instance_four_cities(self):
        inst = MultiObjectiveTSPInstance.random(n=4, n_objectives=2, seed=16)
        sol = weighted_sum_nn(inst, n_weights=5)
        assert sol.n_solutions >= 1
        for tour in sol.pareto_tours:
            assert len(tour) == 4

    def test_three_objectives(self):
        inst = MultiObjectiveTSPInstance.random(n=5, n_objectives=3, seed=17)
        sol = weighted_sum_nn(inst, n_weights=5)
        assert sol.n_solutions >= 1
        for pt in sol.pareto_front:
            assert len(pt) == 3
