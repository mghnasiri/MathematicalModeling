"""
Tests for Farm-to-Market Delivery Routing Problem

Covers: instance creation, deterministic CVRP, stochastic VRP,
route validity, and method comparison.

30 tests across 6 test classes.
"""
from __future__ import annotations

import sys
import os
import importlib.util

import numpy as np
import pytest


def _load_mod(name: str, filepath: str):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_inst_mod = _load_mod("fd_inst_test", os.path.join(_base, "instance.py"))
_det_mod = _load_mod(
    "fd_det_test",
    os.path.join(_base, "heuristics", "deterministic_routing.py"),
)
_sto_mod = _load_mod(
    "fd_sto_test",
    os.path.join(_base, "metaheuristics", "stochastic_routing.py"),
)

FarmDeliveryInstance = _inst_mod.FarmDeliveryInstance
DeliveryPoint = _inst_mod.DeliveryPoint


class TestDeliveryPoint:
    """Test DeliveryPoint dataclass."""

    def test_creation(self):
        dp = DeliveryPoint("Test Market", "farmers_market", 450)
        assert dp.name == "Test Market"
        assert dp.point_type == "farmers_market"
        assert dp.base_demand_kg == 450


class TestFarmDeliveryInstance:
    """Test instance creation and data generation."""

    def test_quebec_cooperative_creation(self):
        inst = FarmDeliveryInstance.quebec_cooperative()
        assert inst.n_customers == 15
        assert inst.truck_capacity_kg == 2000.0

    def test_coordinate_generation_shape(self):
        inst = FarmDeliveryInstance.quebec_cooperative()
        coords = inst.generate_coordinates(seed=42)
        assert coords.shape == (16, 2)  # depot + 15 customers

    def test_distance_matrix_shape(self):
        inst = FarmDeliveryInstance.quebec_cooperative()
        coords = inst.generate_coordinates(seed=42)
        dist = inst.generate_distance_matrix(coords)
        assert dist.shape == (16, 16)

    def test_distance_matrix_symmetric(self):
        inst = FarmDeliveryInstance.quebec_cooperative()
        coords = inst.generate_coordinates(seed=42)
        dist = inst.generate_distance_matrix(coords)
        np.testing.assert_array_almost_equal(dist, dist.T)

    def test_distance_matrix_diagonal_zero(self):
        inst = FarmDeliveryInstance.quebec_cooperative()
        coords = inst.generate_coordinates(seed=42)
        dist = inst.generate_distance_matrix(coords)
        np.testing.assert_array_almost_equal(np.diag(dist), 0)

    def test_demand_scenarios_shape(self):
        inst = FarmDeliveryInstance.quebec_cooperative(n_demand_scenarios=50)
        scenarios = inst.generate_demand_scenarios(seed=42)
        assert scenarios.shape == (50, 15)

    def test_demand_scenarios_positive(self):
        inst = FarmDeliveryInstance.quebec_cooperative()
        scenarios = inst.generate_demand_scenarios(seed=42)
        assert np.all(scenarios >= 10.0)

    def test_random_instance(self):
        inst = FarmDeliveryInstance.random(n_customers=8, seed=42)
        assert inst.n_customers == 8

    def test_individual_demands_within_capacity(self):
        inst = FarmDeliveryInstance.quebec_cooperative()
        for dp in inst.delivery_points:
            assert dp.base_demand_kg <= inst.truck_capacity_kg


class TestClarkeWrightDelivery:
    """Test deterministic Clarke-Wright solution."""

    def test_returns_solution(self):
        inst = FarmDeliveryInstance.quebec_cooperative()
        sol = _det_mod.clarke_wright_delivery(inst, seed=42)
        assert type(sol).__name__ == "FarmDeliverySolution"

    def test_positive_distance(self):
        inst = FarmDeliveryInstance.quebec_cooperative()
        sol = _det_mod.clarke_wright_delivery(inst, seed=42)
        assert sol.total_distance > 0

    def test_routes_cover_all_customers(self):
        inst = FarmDeliveryInstance.quebec_cooperative()
        sol = _det_mod.clarke_wright_delivery(inst, seed=42)
        all_customers = set()
        for route in sol.routes:
            all_customers.update(route)
        assert all_customers == set(range(1, inst.n_customers + 1))

    def test_no_duplicate_customers(self):
        inst = FarmDeliveryInstance.quebec_cooperative()
        sol = _det_mod.clarke_wright_delivery(inst, seed=42)
        all_customers = []
        for route in sol.routes:
            all_customers.extend(route)
        assert len(all_customers) == len(set(all_customers))

    def test_at_least_one_vehicle(self):
        inst = FarmDeliveryInstance.quebec_cooperative()
        sol = _det_mod.clarke_wright_delivery(inst, seed=42)
        assert sol.n_vehicles >= 1

    def test_method_label(self):
        inst = FarmDeliveryInstance.quebec_cooperative()
        sol = _det_mod.clarke_wright_delivery(inst, seed=42)
        assert sol.method == "Clarke-Wright"


class TestSweepDelivery:
    """Test deterministic sweep algorithm."""

    def test_returns_solution(self):
        inst = FarmDeliveryInstance.quebec_cooperative()
        sol = _det_mod.sweep_delivery(inst, seed=42)
        assert type(sol).__name__ == "FarmDeliverySolution"

    def test_positive_distance(self):
        inst = FarmDeliveryInstance.quebec_cooperative()
        sol = _det_mod.sweep_delivery(inst, seed=42)
        assert sol.total_distance > 0

    def test_routes_cover_all_customers(self):
        inst = FarmDeliveryInstance.quebec_cooperative()
        sol = _det_mod.sweep_delivery(inst, seed=42)
        all_customers = set()
        for route in sol.routes:
            all_customers.update(route)
        assert all_customers == set(range(1, inst.n_customers + 1))

    def test_method_label(self):
        inst = FarmDeliveryInstance.quebec_cooperative()
        sol = _det_mod.sweep_delivery(inst, seed=42)
        assert sol.method == "Sweep"


class TestStochasticDelivery:
    """Test stochastic VRP solutions."""

    def test_cc_returns_solution(self):
        inst = FarmDeliveryInstance.quebec_cooperative()
        sol = _sto_mod.chance_constrained_delivery(inst, seed=42)
        assert type(sol).__name__ == "FarmDeliverySolution"

    def test_cc_positive_distance(self):
        inst = FarmDeliveryInstance.quebec_cooperative()
        sol = _sto_mod.chance_constrained_delivery(inst, seed=42)
        assert sol.total_distance > 0

    def test_cc_has_expected_cost(self):
        inst = FarmDeliveryInstance.quebec_cooperative()
        sol = _sto_mod.chance_constrained_delivery(inst, seed=42)
        assert sol.expected_cost is not None
        assert sol.expected_cost > 0

    def test_cc_has_overflow_prob(self):
        inst = FarmDeliveryInstance.quebec_cooperative()
        sol = _sto_mod.chance_constrained_delivery(inst, seed=42)
        assert sol.max_overflow_prob is not None

    def test_sa_returns_solution(self):
        inst = FarmDeliveryInstance.quebec_cooperative()
        sol = _sto_mod.stochastic_sa_delivery(inst, max_iterations=1000, seed=42)
        assert type(sol).__name__ == "FarmDeliverySolution"

    def test_sa_positive_distance(self):
        inst = FarmDeliveryInstance.quebec_cooperative()
        sol = _sto_mod.stochastic_sa_delivery(inst, max_iterations=1000, seed=42)
        assert sol.total_distance > 0


class TestMethodComparison:
    """Compare deterministic vs stochastic methods."""

    def test_both_cover_all_customers(self):
        inst = FarmDeliveryInstance.quebec_cooperative()
        sol_det = _det_mod.clarke_wright_delivery(inst, seed=42)
        sol_sto = _sto_mod.chance_constrained_delivery(inst, seed=42)

        det_custs = set()
        for route in sol_det.routes:
            det_custs.update(route)
        sto_custs = set()
        for route in sol_sto.routes:
            sto_custs.update(route)

        expected = set(range(1, inst.n_customers + 1))
        assert det_custs == expected
        assert sto_custs == expected

    def test_repr(self):
        inst = FarmDeliveryInstance.quebec_cooperative()
        sol = _det_mod.clarke_wright_delivery(inst, seed=42)
        r = repr(sol)
        assert "FarmDeliverySolution" in r
