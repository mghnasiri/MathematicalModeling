"""
Tests for TSP Tabu Search.

Run: python -m pytest problems/routing/tsp/tests/test_tsp_ts.py -v
"""

from __future__ import annotations

import os
import sys
import pytest
import numpy as np
import importlib.util

_base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_module(name: str, filepath: str):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_instance_mod = _load_module(
    "tsp_inst_ts_test", os.path.join(_base_dir, "instance.py")
)
_ts_mod = _load_module(
    "tsp_ts_test", os.path.join(_base_dir, "metaheuristics", "tabu_search.py")
)

TSPInstance = _instance_mod.TSPInstance
TSPSolution = _instance_mod.TSPSolution
validate_tour = _instance_mod.validate_tour
small4 = _instance_mod.small4
small5 = _instance_mod.small5
gr17 = _instance_mod.gr17
tabu_search = _ts_mod.tabu_search


class TestTSPTabuSearch:
    """Test Tabu Search for TSP."""

    def test_returns_valid_tour(self):
        inst = TSPInstance.random(n=10, seed=42)
        sol = tabu_search(inst, max_iterations=200, seed=42)
        is_valid, _ = validate_tour(inst, sol.tour)
        assert is_valid

    def test_distance_correct(self):
        inst = TSPInstance.random(n=10, seed=42)
        sol = tabu_search(inst, max_iterations=200, seed=42)
        assert abs(sol.distance - inst.tour_distance(sol.tour)) < 1e-6

    def test_small4_quality(self):
        """TS should find a good solution on 4-city instance."""
        inst = small4()
        sol = tabu_search(inst, max_iterations=200, seed=42)
        assert sol.distance <= 8.0 + 1e-6

    def test_small5_optimal(self):
        """TS should find optimal on 5-city instance."""
        inst = small5()
        sol = tabu_search(inst, max_iterations=500, seed=42)
        assert abs(sol.distance - 19.0) < 1e-6

    def test_gr17_quality(self):
        """TS should find good solution for gr17 (optimal=2016)."""
        inst = gr17()
        sol = tabu_search(inst, max_iterations=5000, seed=42)
        assert sol.distance <= 2200  # Within 10% of optimal

    def test_deterministic_with_seed(self):
        inst = TSPInstance.random(n=10, seed=42)
        sol_a = tabu_search(inst, max_iterations=200, seed=123)
        sol_b = tabu_search(inst, max_iterations=200, seed=123)
        assert abs(sol_a.distance - sol_b.distance) < 1e-6
        assert sol_a.tour == sol_b.tour

    def test_tiny_instance(self):
        inst = TSPInstance.from_distance_matrix([[0, 5], [5, 0]])
        sol = tabu_search(inst, max_iterations=10, seed=42)
        assert sorted(sol.tour) == [0, 1]

    def test_three_cities(self):
        inst = TSPInstance.from_distance_matrix([
            [0, 1, 2], [1, 0, 3], [2, 3, 0],
        ])
        sol = tabu_search(inst, max_iterations=10, seed=42)
        assert sorted(sol.tour) == [0, 1, 2]

    def test_time_limit(self):
        inst = TSPInstance.random(n=20, seed=42)
        sol = tabu_search(inst, time_limit=2.0, seed=42)
        is_valid, _ = validate_tour(inst, sol.tour)
        assert is_valid

    def test_custom_tabu_tenure(self):
        inst = TSPInstance.random(n=15, seed=42)
        sol = tabu_search(
            inst, max_iterations=200, tabu_tenure=10, seed=42,
        )
        is_valid, _ = validate_tour(inst, sol.tour)
        assert is_valid

    def test_improves_over_iterations(self):
        inst = TSPInstance.random(n=20, seed=42)
        sol_short = tabu_search(inst, max_iterations=50, seed=42)
        sol_long = tabu_search(inst, max_iterations=2000, seed=42)
        assert sol_long.distance <= sol_short.distance + 1e-6
