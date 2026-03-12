"""
Test suite for MDVRP (Multi-Depot Vehicle Routing Problem).

Tests cover:
- Instance creation and validation
- Nearest depot assignment + Clarke-Wright heuristic
- Solution validation (capacity, completeness, depot assignment)
"""

from __future__ import annotations

import os
import sys
import pytest
import numpy as np
import importlib.util

# ── Module loading ───────────────────────────────────────────────────────────

_base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_mod(name: str, filepath: str):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_instance_mod = _load_mod(
    "mdvrp_instance_test", os.path.join(_base_dir, "instance.py")
)
_nd_mod = _load_mod(
    "mdvrp_nd_test",
    os.path.join(_base_dir, "heuristics", "nearest_depot.py"),
)

MDVRPInstance = _instance_mod.MDVRPInstance
MDVRPSolution = _instance_mod.MDVRPSolution
validate_solution = _instance_mod.validate_solution
small_mdvrp4 = _instance_mod.small_mdvrp4
medium_mdvrp8 = _instance_mod.medium_mdvrp8

nearest_depot_cw = _nd_mod.nearest_depot_cw


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def inst4():
    return small_mdvrp4()


@pytest.fixture
def inst8():
    return medium_mdvrp8()


@pytest.fixture
def random_inst():
    return MDVRPInstance.random(10, n_depots=3, capacity=60.0, seed=42)


# ── Instance tests ───────────────────────────────────────────────────────────


class TestMDVRPInstance:
    def test_create_basic(self, inst4):
        assert inst4.n_customers == 4
        assert inst4.n_depots == 2
        assert inst4.capacity == 40.0
        assert inst4.demands.shape == (4,)
        assert inst4.distance_matrix.shape == (6, 6)

    def test_node_indices(self, inst4):
        assert inst4.depot_node(0) == 0
        assert inst4.depot_node(1) == 1
        assert inst4.customer_node(0) == 2
        assert inst4.customer_node(3) == 5

    def test_random_instance(self):
        inst = MDVRPInstance.random(12, n_depots=3, seed=123)
        assert inst.n_customers == 12
        assert inst.n_depots == 3
        assert inst.distance_matrix.shape == (15, 15)
        assert np.all(inst.demands <= inst.capacity)

    def test_route_distance(self, inst4):
        d = inst4.route_distance(0, [0])
        d_node = inst4.depot_node(0)
        c_node = inst4.customer_node(0)
        expected = (
            inst4.distance_matrix[d_node][c_node]
            + inst4.distance_matrix[c_node][d_node]
        )
        assert abs(d - expected) < 1e-10

    def test_empty_route(self, inst4):
        assert inst4.route_distance(0, []) == 0.0

    def test_route_demand(self, inst4):
        demand = inst4.route_demand([0, 1])
        assert abs(demand - 25.0) < 1e-10  # 10 + 15

    def test_invalid_demands_shape(self):
        with pytest.raises(ValueError):
            MDVRPInstance(
                n_customers=3, n_depots=2, capacity=30.0,
                demands=np.array([10.0, 10.0]),
                depot_coords=np.zeros((2, 2)),
                customer_coords=np.zeros((3, 2)),
                distance_matrix=np.zeros((5, 5)),
            )

    def test_demand_exceeds_capacity(self):
        with pytest.raises(ValueError):
            MDVRPInstance(
                n_customers=2, n_depots=1, capacity=5.0,
                demands=np.array([3.0, 10.0]),
                depot_coords=np.zeros((1, 2)),
                customer_coords=np.zeros((2, 2)),
                distance_matrix=np.zeros((3, 3)),
            )


# ── Validation tests ─────────────────────────────────────────────────────────


class TestValidation:
    def test_valid_solution(self, inst4):
        sol = nearest_depot_cw(inst4)
        valid, errors = validate_solution(inst4, sol)
        assert valid, errors

    def test_missing_customer(self, inst4):
        sol = MDVRPSolution(
            depot_routes={0: [[0]], 1: [[2]]}, distance=0.0
        )
        valid, errors = validate_solution(inst4, sol)
        assert not valid

    def test_duplicate_customer(self, inst4):
        sol = MDVRPSolution(
            depot_routes={0: [[0, 1, 0]], 1: [[2, 3]]}, distance=0.0
        )
        valid, errors = validate_solution(inst4, sol)
        assert not valid


# ── Nearest depot + CW tests ────────────────────────────────────────────────


class TestNearestDepotCW:
    def test_feasible_solution_small(self, inst4):
        sol = nearest_depot_cw(inst4)
        valid, errors = validate_solution(inst4, sol)
        assert valid, errors

    def test_feasible_solution_medium(self, inst8):
        sol = nearest_depot_cw(inst8)
        valid, errors = validate_solution(inst8, sol)
        assert valid, errors

    def test_all_customers_visited(self, inst8):
        sol = nearest_depot_cw(inst8)
        all_customers = []
        for routes in sol.depot_routes.values():
            for route in routes:
                all_customers.extend(route)
        assert sorted(all_customers) == list(range(inst8.n_customers))

    def test_capacity_respected(self, inst8):
        sol = nearest_depot_cw(inst8)
        for depot_idx, routes in sol.depot_routes.items():
            for route in routes:
                demand = inst8.route_demand(route)
                assert demand <= inst8.capacity + 1e-10

    def test_distance_matches(self, inst4):
        sol = nearest_depot_cw(inst4)
        expected = inst4.total_distance(sol.depot_routes)
        assert abs(sol.distance - expected) < 1e-10

    def test_nearest_assignment(self, inst4):
        """Customers near depot 0 should be assigned to depot 0."""
        sol = nearest_depot_cw(inst4)
        # Customer 0 (10,60) is near depot 0 (20,50)
        # Customer 3 (85,40) is near depot 1 (80,50)
        depot0_customers = []
        depot1_customers = []
        for route in sol.depot_routes.get(0, []):
            depot0_customers.extend(route)
        for route in sol.depot_routes.get(1, []):
            depot1_customers.extend(route)
        assert 0 in depot0_customers
        assert 3 in depot1_customers

    def test_random_instance(self, random_inst):
        sol = nearest_depot_cw(random_inst)
        valid, errors = validate_solution(random_inst, sol)
        assert valid, errors

    def test_repr(self, inst4):
        sol = nearest_depot_cw(inst4)
        r = repr(sol)
        assert "MDVRPSolution" in r
        assert "distance" in r
