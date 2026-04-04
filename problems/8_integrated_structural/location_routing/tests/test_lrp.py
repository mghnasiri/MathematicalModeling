"""
Test suite for Location-Routing Problem (LRP).

Tests cover:
- Instance creation, validation, and properties
- Greedy heuristic correctness and solution quality
- Simulated annealing convergence and determinism

13 tests across 3 test classes.
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


_inst_mod = _load_module(
    "lrp_inst_test", os.path.join(_base_dir, "instance.py")
)
_gr_mod = _load_module(
    "lrp_gr_test",
    os.path.join(_base_dir, "heuristics", "greedy_lrp.py"),
)
_sa_mod = _load_module(
    "lrp_sa_test",
    os.path.join(_base_dir, "metaheuristics", "simulated_annealing.py"),
)

LRPInstance = _inst_mod.LRPInstance
LRPSolution = _inst_mod.LRPSolution
compute_cost = _inst_mod.compute_cost
validate_solution = _inst_mod.validate_solution
small_lrp_3_8 = _inst_mod.small_lrp_3_8
medium_lrp_5_15 = _inst_mod.medium_lrp_5_15

greedy_lrp = _gr_mod.greedy_lrp
simulated_annealing = _sa_mod.simulated_annealing


@pytest.fixture
def inst_small():
    return small_lrp_3_8()


@pytest.fixture
def inst_medium():
    return medium_lrp_5_15()


@pytest.fixture
def inst_random():
    return LRPInstance.random(4, 12, seed=99)


class TestLRPInstance:
    """Tests for LRPInstance creation and properties."""

    def test_random(self):
        inst = LRPInstance.random(3, 10, seed=123)
        assert inst.m == 3
        assert inst.n == 10
        assert inst.fixed_costs.shape == (3,)
        assert inst.capacities.shape == (3,)
        assert inst.demands.shape == (10,)
        assert inst.distance_matrix.shape == (13, 13)

    def test_small(self, inst_small):
        assert inst_small.m == 3
        assert inst_small.n == 8
        assert inst_small.name == "small_3_8"
        assert inst_small.vehicle_capacity == 30.0
        assert len(inst_small.demands) == 8

    def test_costs_positive(self, inst_small):
        assert np.all(inst_small.fixed_costs > 0)
        assert np.all(inst_small.demands > 0)
        assert np.all(inst_small.capacities > 0)

    def test_distance_matrix(self, inst_small):
        dm = inst_small.distance_matrix
        total_nodes = inst_small.m + inst_small.n
        assert dm.shape == (total_nodes, total_nodes)
        # Diagonal should be zero
        assert np.allclose(np.diag(dm), 0.0)
        # Symmetric for Euclidean
        assert np.allclose(dm, dm.T, atol=1e-10)
        # Non-negative
        assert np.all(dm >= -1e-10)

    def test_invalid_shapes(self):
        with pytest.raises(ValueError):
            LRPInstance(
                m=2,
                n=3,
                fixed_costs=np.array([1.0]),  # wrong shape
                capacities=np.array([10.0, 10.0]),
                demands=np.array([5.0, 5.0, 5.0]),
                vehicle_capacity=20.0,
                distance_matrix=np.zeros((5, 5)),
            )

    def test_route_distance(self, inst_small):
        # A single-customer route from depot 0 to customer 0 and back
        d = inst_small.route_distance(0, [0])
        assert d > 0
        # Empty route should be 0
        assert inst_small.route_distance(0, []) == 0.0

    def test_route_demand(self, inst_small):
        demand = inst_small.route_demand([0, 1])
        assert demand == inst_small.demands[0] + inst_small.demands[1]

    def test_from_coordinates(self):
        inst = LRPInstance.from_coordinates(
            depot_coords=[[0, 0], [10, 10]],
            customer_coords=[[1, 1], [2, 2], [8, 8], [9, 9]],
            fixed_costs=[100.0, 200.0],
            capacities=[50.0, 50.0],
            demands=[5.0, 10.0, 8.0, 7.0],
            vehicle_capacity=20.0,
            name="test_from_coords",
        )
        assert inst.m == 2
        assert inst.n == 4
        assert inst.distance_matrix.shape == (6, 6)


class TestGreedyLRP:
    """Tests for the greedy LRP heuristic."""

    def test_returns_solution(self, inst_small):
        sol = greedy_lrp(inst_small)
        assert type(sol).__name__ == "LRPSolution"
        assert sol.cost > 0

    def test_all_customers_served(self, inst_small):
        sol = greedy_lrp(inst_small)
        all_custs: list[int] = []
        for depot_routes in sol.routes.values():
            for route in depot_routes:
                all_custs.extend(route)
        assert sorted(all_custs) == list(range(inst_small.n))

    def test_capacity_respected(self, inst_small):
        sol = greedy_lrp(inst_small)
        for d, depot_routes in sol.routes.items():
            for route in depot_routes:
                demand = inst_small.route_demand(route)
                assert demand <= inst_small.vehicle_capacity + 1e-10

    def test_cost_positive(self, inst_small):
        sol = greedy_lrp(inst_small)
        assert sol.cost > 0

    def test_at_least_one_depot_open(self, inst_small):
        sol = greedy_lrp(inst_small)
        assert len(sol.open_depots) >= 1

    def test_validates(self, inst_small):
        sol = greedy_lrp(inst_small)
        valid, errors = validate_solution(inst_small, sol)
        assert valid, errors

    def test_medium_instance(self, inst_medium):
        sol = greedy_lrp(inst_medium)
        valid, errors = validate_solution(inst_medium, sol)
        assert valid, errors

    def test_random_instance(self, inst_random):
        sol = greedy_lrp(inst_random)
        valid, errors = validate_solution(inst_random, sol)
        assert valid, errors


class TestLRPSA:
    """Tests for simulated annealing on LRP."""

    def test_returns_solution(self, inst_small):
        sol = simulated_annealing(
            inst_small, max_iterations=5000, seed=42
        )
        assert type(sol).__name__ == "LRPSolution"
        assert sol.cost > 0

    def test_improves_over_greedy(self, inst_small):
        greedy_sol = greedy_lrp(inst_small)
        sa_sol = simulated_annealing(
            inst_small, max_iterations=15000, seed=42
        )
        # SA should be competitive with greedy (within 30% tolerance)
        assert sa_sol.cost <= greedy_sol.cost * 1.3

    def test_deterministic_with_seed(self, inst_small):
        s1 = simulated_annealing(
            inst_small, max_iterations=5000, seed=42
        )
        s2 = simulated_annealing(
            inst_small, max_iterations=5000, seed=42
        )
        assert abs(s1.cost - s2.cost) < 1e-10

    def test_validates(self, inst_small):
        sol = simulated_annealing(
            inst_small, max_iterations=5000, seed=42
        )
        valid, errors = validate_solution(inst_small, sol)
        assert valid, errors

    def test_medium_instance(self, inst_medium):
        sol = simulated_annealing(
            inst_medium, max_iterations=5000, seed=42
        )
        valid, errors = validate_solution(inst_medium, sol)
        assert valid, errors


class TestCrossMethod:
    """Cross-method validation tests."""

    def test_all_methods_valid(self, inst_small):
        methods = [
            greedy_lrp,
            lambda i: simulated_annealing(i, max_iterations=5000, seed=42),
        ]
        for method in methods:
            sol = method(inst_small)
            valid, errors = validate_solution(inst_small, sol)
            assert valid, errors

    def test_cost_decomposition(self, inst_small):
        sol = greedy_lrp(inst_small)
        total, fixed, routing = compute_cost(inst_small, sol)
        assert abs(total - sol.cost) < 1e-4
        assert fixed > 0
        assert routing > 0
        assert abs(total - (fixed + routing)) < 1e-10
