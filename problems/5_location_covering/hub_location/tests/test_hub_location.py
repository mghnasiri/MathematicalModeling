"""Tests for the Hub Location problem (p-hub median)."""
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
_instance_mod = _load_mod("hub_instance", os.path.join(_base, "instance.py"))
_greedy_mod = _load_mod(
    "hub_greedy", os.path.join(_base, "heuristics", "greedy_hub.py")
)

HubLocationInstance = _instance_mod.HubLocationInstance
HubLocationSolution = _instance_mod.HubLocationSolution
greedy_hub = _greedy_mod.greedy_hub
enumeration_hub = _greedy_mod.enumeration_hub


class TestHubLocationInstance:
    """Tests for instance construction and cost computation."""

    def test_random_instance_shape(self):
        inst = HubLocationInstance.random(n=5, p=2, seed=1)
        assert inst.n == 5
        assert inst.p == 2
        assert inst.flows.shape == (5, 5)
        assert inst.distances.shape == (5, 5)

    def test_zero_diagonal_flows(self):
        inst = HubLocationInstance.random(n=6, p=2, seed=2)
        for i in range(inst.n):
            assert inst.flows[i, i] == 0.0

    def test_symmetric_distances(self):
        inst = HubLocationInstance.random(n=5, p=2, seed=3)
        np.testing.assert_allclose(inst.distances, inst.distances.T, atol=1e-10)

    def test_transport_cost_nonnegative(self):
        inst = HubLocationInstance.random(n=5, p=2, seed=4)
        hubs = [0, 1]
        assignments = [0, 1, 0, 1, 0]
        cost = inst.transport_cost(hubs, assignments)
        assert cost >= 0.0


class TestGreedyHub:
    """Tests for the greedy hub heuristic."""

    def test_correct_number_of_hubs(self):
        inst = HubLocationInstance.random(n=8, p=3, seed=10)
        sol = greedy_hub(inst)
        assert len(sol.hubs) == 3

    def test_hubs_are_valid_nodes(self):
        inst = HubLocationInstance.random(n=8, p=3, seed=11)
        sol = greedy_hub(inst)
        for h in sol.hubs:
            assert 0 <= h < inst.n

    def test_all_nodes_assigned(self):
        inst = HubLocationInstance.random(n=8, p=3, seed=12)
        sol = greedy_hub(inst)
        assert len(sol.assignments) == inst.n
        for a in sol.assignments:
            assert a in sol.hubs

    def test_objective_matches_cost(self):
        inst = HubLocationInstance.random(n=6, p=2, seed=13)
        sol = greedy_hub(inst)
        recomputed = inst.transport_cost(sol.hubs, sol.assignments)
        assert abs(sol.objective - recomputed) < 1e-6

    def test_single_hub(self):
        inst = HubLocationInstance.random(n=5, p=1, seed=14)
        sol = greedy_hub(inst)
        assert len(sol.hubs) == 1
        assert all(a == sol.hubs[0] for a in sol.assignments)


class TestEnumerationHub:
    """Tests for exhaustive enumeration (small instances)."""

    def test_optimal_small_instance(self):
        inst = HubLocationInstance.random(n=5, p=2, seed=20)
        sol = enumeration_hub(inst)
        assert len(sol.hubs) == 2
        # Verify no other combination is better
        from itertools import combinations
        for combo in combinations(range(5), 2):
            hubs = list(combo)
            assignments = [min(hubs, key=lambda h: inst.distances[i, h])
                           for i in range(5)]
            cost = inst.transport_cost(hubs, assignments)
            assert cost >= sol.objective - 1e-6

    def test_greedy_within_reasonable_gap(self):
        inst = HubLocationInstance.random(n=7, p=2, seed=21)
        sol_greedy = greedy_hub(inst)
        sol_opt = enumeration_hub(inst)
        # Greedy should be within 50% of optimal for small instances
        assert sol_greedy.objective <= sol_opt.objective * 1.5

    def test_greedy_finds_optimal_trivial(self):
        """When p == n, all nodes are hubs — cost should be minimal."""
        inst = HubLocationInstance.random(n=4, p=4, seed=22)
        sol = greedy_hub(inst)
        assert len(sol.hubs) == 4
        assert set(sol.hubs) == {0, 1, 2, 3}
