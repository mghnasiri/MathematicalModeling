"""Tests for Dial-a-Ride Problem."""
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
_inst_mod = _load_mod("darp_instance_test", os.path.join(_base, "instance.py"))
_heur_mod = _load_mod(
    "darp_insertion_test",
    os.path.join(_base, "heuristics", "insertion_darp.py"),
)

DARPInstance = _inst_mod.DARPInstance
DARPSolution = _inst_mod.DARPSolution
DARPRequest = _inst_mod.DARPRequest
validate_solution = _inst_mod.validate_solution
small_darp3 = _inst_mod.small_darp3
cheapest_insertion_darp = _heur_mod.cheapest_insertion_darp


class TestInstance:

    def test_creation(self):
        inst = small_darp3()
        assert inst.n_requests == 3
        assert inst.n_vehicles == 2
        assert inst.n_nodes == 8

    def test_depot_nodes(self):
        inst = small_darp3()
        assert inst.depot_start == 0
        assert inst.depot_end == 7

    def test_pickup_delivery_nodes(self):
        inst = small_darp3()
        assert inst.pickup_node(0) == 1
        assert inst.pickup_node(2) == 3
        assert inst.delivery_node(0) == 4
        assert inst.delivery_node(2) == 6

    def test_distance_symmetric(self):
        inst = small_darp3()
        assert inst.distance(1, 2) == pytest.approx(inst.distance(2, 1))

    def test_random_instance(self):
        inst = DARPInstance.random(n_requests=5, n_vehicles=2, seed=42)
        assert inst.n_requests == 5
        assert inst.n_vehicles == 2
        assert inst.n_nodes == 12

    def test_route_distance(self):
        inst = small_darp3()
        d = inst.route_distance([0, 1, 4, 7])
        assert d > 0


class TestInsertion:

    def test_all_served(self):
        inst = small_darp3()
        sol = cheapest_insertion_darp(inst, seed=42)
        assert sol.n_served == 3

    def test_distance_positive(self):
        inst = small_darp3()
        sol = cheapest_insertion_darp(inst, seed=42)
        assert sol.total_distance > 0

    def test_routes_start_end_depot(self):
        inst = small_darp3()
        sol = cheapest_insertion_darp(inst, seed=42)
        for route in sol.routes:
            assert route[0] == inst.depot_start
            assert route[-1] == inst.depot_end

    def test_precedence_maintained(self):
        inst = small_darp3()
        sol = cheapest_insertion_darp(inst, seed=42)
        for route in sol.routes:
            pickup_seen = set()
            for node in route:
                for req_idx in range(inst.n_requests):
                    if node == inst.pickup_node(req_idx):
                        pickup_seen.add(req_idx)
                    elif node == inst.delivery_node(req_idx):
                        assert req_idx in pickup_seen

    def test_random_instance(self):
        inst = DARPInstance.random(n_requests=6, n_vehicles=3, seed=7)
        sol = cheapest_insertion_darp(inst, seed=7)
        assert sol.n_served == 6

    def test_validate_solution(self):
        inst = small_darp3()
        sol = cheapest_insertion_darp(inst, seed=42)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Validation errors: {errors}"

    def test_repr(self):
        inst = small_darp3()
        sol = cheapest_insertion_darp(inst, seed=42)
        r = repr(sol)
        assert "distance" in r
        assert "served" in r
