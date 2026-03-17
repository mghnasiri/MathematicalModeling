"""
Tests for CVRP Tabu Search.

Run: python -m pytest problems/routing/cvrp/tests/test_cvrp_ts.py -v
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
    "cvrp_inst_ts_test", os.path.join(_base_dir, "instance.py")
)
_ts_mod = _load_module(
    "cvrp_ts_test", os.path.join(_base_dir, "metaheuristics", "tabu_search.py")
)
_cw_mod = _load_module(
    "cvrp_cw_ts_test", os.path.join(_base_dir, "heuristics", "clarke_wright.py")
)

CVRPInstance = _instance_mod.CVRPInstance
CVRPSolution = _instance_mod.CVRPSolution
validate_solution = _instance_mod.validate_solution
small6 = _instance_mod.small6
christofides1 = _instance_mod.christofides1
medium12 = _instance_mod.medium12
tabu_search = _ts_mod.tabu_search
clarke_wright = _cw_mod.clarke_wright_savings


class TestCVRPTabuSearch:
    """Test Tabu Search for CVRP."""

    def test_returns_valid_solution(self):
        inst = small6()
        sol = tabu_search(inst, max_iterations=500, seed=42)
        is_valid, errors = validate_solution(inst, sol)
        assert is_valid, f"Invalid: {errors}"

    def test_distance_correct(self):
        inst = small6()
        sol = tabu_search(inst, max_iterations=500, seed=42)
        actual_dist = inst.total_distance(sol.routes)
        assert abs(sol.distance - actual_dist) < 1e-6

    def test_christofides1_valid(self):
        inst = christofides1()
        sol = tabu_search(inst, max_iterations=500, seed=42)
        is_valid, errors = validate_solution(inst, sol)
        assert is_valid, f"Invalid: {errors}"

    def test_no_worse_than_clarke_wright(self):
        inst = medium12()
        cw_sol = clarke_wright(inst)
        ts_sol = tabu_search(inst, max_iterations=1000, seed=42)
        assert ts_sol.distance <= cw_sol.distance + 1e-6

    def test_deterministic_with_seed(self):
        inst = small6()
        sol_a = tabu_search(inst, max_iterations=300, seed=123)
        sol_b = tabu_search(inst, max_iterations=300, seed=123)
        assert abs(sol_a.distance - sol_b.distance) < 1e-6

    def test_all_customers_served(self):
        inst = CVRPInstance.random(n=15, seed=42)
        sol = tabu_search(inst, max_iterations=500, seed=42)
        all_customers = [c for r in sol.routes for c in r]
        assert sorted(all_customers) == list(range(1, 16))

    def test_capacity_respected(self):
        inst = CVRPInstance.random(n=15, seed=42)
        sol = tabu_search(inst, max_iterations=500, seed=42)
        for route in sol.routes:
            demand = inst.route_demand(route)
            assert demand <= inst.capacity + 1e-10

    def test_time_limit(self):
        inst = CVRPInstance.random(n=15, seed=42)
        sol = tabu_search(inst, time_limit=2.0, seed=42)
        is_valid, _ = validate_solution(inst, sol)
        assert is_valid

    def test_medium_instance(self):
        inst = medium12()
        sol = tabu_search(inst, max_iterations=1000, seed=42)
        is_valid, errors = validate_solution(inst, sol)
        assert is_valid, f"Invalid: {errors}"

    def test_custom_tabu_tenure(self):
        inst = small6()
        sol = tabu_search(inst, max_iterations=300, tabu_tenure=3, seed=42)
        is_valid, _ = validate_solution(inst, sol)
        assert is_valid
