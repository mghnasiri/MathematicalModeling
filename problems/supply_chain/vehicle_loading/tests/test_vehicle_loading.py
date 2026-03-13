"""Tests for Vehicle Loading problem.

Tests: capacity constraints, all items loaded, correctness on small instances.
"""
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

_instance_mod = _load_mod("vl_inst_test", os.path.join(_base, "instance.py"))
_ffd_mod = _load_mod("vl_ffd_test", os.path.join(_base, "heuristics", "greedy_ffd.py"))

VehicleLoadingInstance = _instance_mod.VehicleLoadingInstance
VehicleLoadingSolution = _instance_mod.VehicleLoadingSolution
greedy_ffd = _ffd_mod.greedy_ffd
greedy_ffd_volume = _ffd_mod.greedy_ffd_volume


class TestVehicleLoadingInstance:
    """Test instance creation and validation."""

    def test_random_instance(self):
        inst = VehicleLoadingInstance.random(n_items=10, seed=42)
        assert inst.n_items == 10
        assert len(inst.weights) == 10
        assert len(inst.volumes) == 10

    def test_validate_valid(self):
        inst = VehicleLoadingInstance(
            n_items=3,
            weights=np.array([10.0, 20.0, 30.0]),
            volumes=np.array([10.0, 20.0, 30.0]),
            weight_capacity=50.0,
            volume_capacity=50.0,
        )
        # Two vehicles: [0,1] (w=30,v=30) and [2] (w=30,v=30)
        assert inst.validate_loading([[0, 1], [2]])

    def test_validate_weight_exceeded(self):
        inst = VehicleLoadingInstance(
            n_items=2,
            weights=np.array([30.0, 30.0]),
            volumes=np.array([10.0, 10.0]),
            weight_capacity=50.0,
            volume_capacity=50.0,
        )
        # One vehicle with both items: weight=60 > 50
        assert not inst.validate_loading([[0, 1]])

    def test_validate_missing_item(self):
        inst = VehicleLoadingInstance(
            n_items=3,
            weights=np.array([10.0, 20.0, 30.0]),
            volumes=np.array([10.0, 20.0, 30.0]),
            weight_capacity=100.0,
            volume_capacity=100.0,
        )
        # Only items 0 and 1 loaded
        assert not inst.validate_loading([[0, 1]])

    def test_validate_duplicate_item(self):
        inst = VehicleLoadingInstance(
            n_items=2,
            weights=np.array([10.0, 20.0]),
            volumes=np.array([10.0, 20.0]),
            weight_capacity=100.0,
            volume_capacity=100.0,
        )
        assert not inst.validate_loading([[0, 1], [0]])


class TestGreedyFFD:
    """Test First-Fit Decreasing heuristic."""

    def test_all_items_loaded(self):
        inst = VehicleLoadingInstance.random(n_items=15, seed=42)
        sol = greedy_ffd(inst)
        all_items = set()
        for v in sol.vehicle_assignments:
            all_items.update(v)
        assert all_items == set(range(inst.n_items))

    def test_capacity_respected(self):
        inst = VehicleLoadingInstance.random(n_items=15, seed=42)
        sol = greedy_ffd(inst)
        assert inst.validate_loading(sol.vehicle_assignments)

    def test_single_item(self):
        inst = VehicleLoadingInstance(
            n_items=1,
            weights=np.array([10.0]),
            volumes=np.array([10.0]),
            weight_capacity=50.0,
            volume_capacity=50.0,
        )
        sol = greedy_ffd(inst)
        assert sol.n_vehicles == 1
        assert sol.vehicle_assignments == [[0]]

    def test_each_item_needs_own_vehicle(self):
        inst = VehicleLoadingInstance(
            n_items=3,
            weights=np.array([90.0, 80.0, 70.0]),
            volumes=np.array([10.0, 10.0, 10.0]),
            weight_capacity=100.0,
            volume_capacity=100.0,
        )
        sol = greedy_ffd(inst)
        assert sol.n_vehicles == 3

    def test_volume_constraint_binding(self):
        """Volume forces separation even when weight fits."""
        inst = VehicleLoadingInstance(
            n_items=2,
            weights=np.array([10.0, 10.0]),
            volumes=np.array([60.0, 60.0]),
            weight_capacity=100.0,
            volume_capacity=100.0,
        )
        sol = greedy_ffd(inst)
        assert sol.n_vehicles == 2

    def test_ffd_volume_variant(self):
        inst = VehicleLoadingInstance.random(n_items=10, seed=42)
        sol = greedy_ffd_volume(inst)
        assert inst.validate_loading(sol.vehicle_assignments)

    def test_solution_repr(self):
        inst = VehicleLoadingInstance.random(n_items=5, seed=42)
        sol = greedy_ffd(inst)
        r = repr(sol)
        assert "VehicleLoadingSolution" in r
        assert "vehicles=" in r
