"""
Test suite for VRPPD (VRP with Pickup and Delivery).

Tests cover:
- Instance creation and validation
- Cheapest insertion heuristic
- Solution validation (precedence, capacity, completeness)
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
    "vrppd_instance_test", os.path.join(_base_dir, "instance.py")
)
_ins_mod = _load_mod(
    "vrppd_insertion_test",
    os.path.join(_base_dir, "heuristics", "insertion_vrppd.py"),
)

VRPPDInstance = _instance_mod.VRPPDInstance
VRPPDSolution = _instance_mod.VRPPDSolution
validate_solution = _instance_mod.validate_solution
small_vrppd3 = _instance_mod.small_vrppd3
medium_vrppd5 = _instance_mod.medium_vrppd5

cheapest_insertion_vrppd = _ins_mod.cheapest_insertion_vrppd


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def inst3():
    return small_vrppd3()


@pytest.fixture
def inst5():
    return medium_vrppd5()


@pytest.fixture
def random_inst():
    return VRPPDInstance.random(6, capacity=40.0, seed=42)


# ── Instance tests ───────────────────────────────────────────────────────────


class TestVRPPDInstance:
    def test_create_basic(self, inst3):
        assert inst3.n_requests == 3
        assert inst3.capacity == 30.0
        assert inst3.loads.shape == (3,)
        assert inst3.distance_matrix.shape == (7, 7)

    def test_pickup_delivery_nodes(self, inst3):
        assert inst3.pickups == [1, 2, 3]
        assert inst3.deliveries == [4, 5, 6]

    def test_pickup_of(self, inst3):
        assert inst3.pickup_of(0) == 1
        assert inst3.pickup_of(1) == 2
        assert inst3.pickup_of(2) == 3

    def test_delivery_of(self, inst3):
        assert inst3.delivery_of(0) == 4
        assert inst3.delivery_of(1) == 5
        assert inst3.delivery_of(2) == 6

    def test_is_pickup_delivery(self, inst3):
        assert inst3.is_pickup(1)
        assert inst3.is_pickup(3)
        assert not inst3.is_pickup(4)
        assert inst3.is_delivery(4)
        assert inst3.is_delivery(6)
        assert not inst3.is_delivery(1)

    def test_random_instance(self):
        inst = VRPPDInstance.random(8, capacity=60.0, seed=123)
        assert inst.n_requests == 8
        assert inst.capacity == 60.0
        assert inst.distance_matrix.shape == (17, 17)
        assert np.all(inst.loads <= inst.capacity)
        assert np.all(inst.loads > 0)

    def test_route_distance(self, inst3):
        d = inst3.route_distance([1, 4])
        expected = (
            inst3.distance_matrix[0][1]
            + inst3.distance_matrix[1][4]
            + inst3.distance_matrix[4][0]
        )
        assert abs(d - expected) < 1e-10

    def test_empty_route(self, inst3):
        assert inst3.route_distance([]) == 0.0
        feasible, _ = inst3.route_feasible([])
        assert feasible

    def test_invalid_loads_shape(self):
        with pytest.raises(ValueError):
            VRPPDInstance(
                n_requests=3,
                capacity=30.0,
                loads=np.array([10.0, 10.0]),
                distance_matrix=np.zeros((7, 7)),
            )

    def test_load_exceeds_capacity(self):
        with pytest.raises(ValueError):
            VRPPDInstance(
                n_requests=2,
                capacity=5.0,
                loads=np.array([3.0, 10.0]),
                distance_matrix=np.zeros((5, 5)),
            )


# ── Route feasibility tests ─────────────────────────────────────────────────


class TestRouteFeasibility:
    def test_valid_route(self, inst3):
        feasible, msg = inst3.route_feasible([1, 4])
        assert feasible, msg

    def test_delivery_before_pickup(self, inst3):
        feasible, msg = inst3.route_feasible([4, 1])
        assert not feasible
        assert "before pickup" in msg.lower() or "Delivery" in msg

    def test_capacity_exceeded(self):
        inst = VRPPDInstance(
            n_requests=2,
            capacity=15.0,
            loads=np.array([10.0, 10.0]),
            distance_matrix=np.zeros((5, 5)),
        )
        # Both pickups before any delivery => load = 20 > 15
        feasible, msg = inst.route_feasible([1, 2, 3, 4])
        assert not feasible


# ── Validation tests ─────────────────────────────────────────────────────────


class TestValidation:
    def test_valid_solution(self, inst3):
        sol = cheapest_insertion_vrppd(inst3)
        valid, errors = validate_solution(inst3, sol)
        assert valid, errors

    def test_missing_node(self, inst3):
        sol = VRPPDSolution(routes=[[1, 4, 2, 5]], distance=0.0)
        valid, errors = validate_solution(inst3, sol)
        assert not valid

    def test_pickup_delivery_different_routes(self, inst3):
        sol = VRPPDSolution(
            routes=[[1, 2, 5], [3, 4, 6]], distance=0.0
        )
        valid, errors = validate_solution(inst3, sol)
        assert not valid  # pickup 1 in route 0, delivery 4 in route 1


# ── Cheapest insertion tests ─────────────────────────────────────────────────


class TestCheapestInsertion:
    def test_feasible_solution_small(self, inst3):
        sol = cheapest_insertion_vrppd(inst3)
        valid, errors = validate_solution(inst3, sol)
        assert valid, errors

    def test_feasible_solution_medium(self, inst5):
        sol = cheapest_insertion_vrppd(inst5)
        valid, errors = validate_solution(inst5, sol)
        assert valid, errors

    def test_all_nodes_visited(self, inst5):
        sol = cheapest_insertion_vrppd(inst5)
        all_nodes = [n for r in sol.routes for n in r]
        assert sorted(all_nodes) == list(range(1, 2 * inst5.n_requests + 1))

    def test_precedence_maintained(self, inst5):
        sol = cheapest_insertion_vrppd(inst5)
        for route in sol.routes:
            seen_pickups = set()
            for node in route:
                if inst5.is_pickup(node):
                    seen_pickups.add(inst5.request_of_node(node))
                elif inst5.is_delivery(node):
                    req = inst5.request_of_node(node)
                    assert req in seen_pickups, (
                        f"Delivery for request {req} before pickup"
                    )

    def test_distance_matches(self, inst3):
        sol = cheapest_insertion_vrppd(inst3)
        expected = inst3.total_distance(sol.routes)
        assert abs(sol.distance - expected) < 1e-10

    def test_random_instance(self, random_inst):
        sol = cheapest_insertion_vrppd(random_inst)
        valid, errors = validate_solution(random_inst, sol)
        assert valid, errors

    def test_repr(self, inst3):
        sol = cheapest_insertion_vrppd(inst3)
        r = repr(sol)
        assert "VRPPDSolution" in r
        assert "distance" in r
