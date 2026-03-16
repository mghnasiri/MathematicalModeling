"""
Tests for Agriculture Cold Storage & Packaging Optimization Problem

Covers: instance creation, bin packing methods, cutting stock methods,
solution quality, and lower bound comparisons.

32 tests across 5 test classes.
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
_inst_mod = _load_mod("cs_inst_test", os.path.join(_base, "instance.py"))
_solver_mod = _load_mod(
    "cs_solver_test",
    os.path.join(_base, "heuristics", "packing_solver.py"),
)

ColdStorageInstance = _inst_mod.ColdStorageInstance
PackagingInstance = _inst_mod.PackagingInstance
ProduceLot = _inst_mod.ProduceLot
SheetType = _inst_mod.SheetType


class TestColdStorageInstance:
    """Test cold storage instance creation."""

    def test_packing_house_creation(self):
        inst = ColdStorageInstance.packing_house()
        assert inst.n_lots == 20
        assert inst.storage_capacity_kg == 1000.0

    def test_total_weight(self):
        inst = ColdStorageInstance.packing_house()
        assert inst.total_weight > 0

    def test_weights_within_capacity(self):
        inst = ColdStorageInstance.packing_house()
        for lot in inst.lots:
            assert lot.weight_kg <= inst.storage_capacity_kg

    def test_get_weights(self):
        inst = ColdStorageInstance.packing_house()
        weights = inst.get_weights()
        assert len(weights) == 20
        assert np.all(weights > 0)

    def test_custom_instance(self):
        lots = [
            ProduceLot("lot-0", "tomato", 300),
            ProduceLot("lot-1", "berry", 100),
        ]
        inst = ColdStorageInstance(lots=lots, storage_capacity_kg=500)
        assert inst.n_lots == 2
        assert inst.total_weight == 400

    def test_reproducible(self):
        inst1 = ColdStorageInstance.packing_house(seed=42)
        inst2 = ColdStorageInstance.packing_house(seed=42)
        w1 = inst1.get_weights()
        w2 = inst2.get_weights()
        np.testing.assert_array_equal(w1, w2)


class TestPackagingInstance:
    """Test packaging instance creation."""

    def test_standard_creation(self):
        inst = PackagingInstance.standard()
        assert inst.n_types == 4
        assert inst.roll_length_cm == 200.0

    def test_sheet_types(self):
        inst = PackagingInstance.standard()
        names = [s.name for s in inst.sheet_types]
        assert "large" in names
        assert "small" in names

    def test_get_lengths(self):
        inst = PackagingInstance.standard()
        lengths = inst.get_lengths()
        assert len(lengths) == 4
        assert np.all(lengths > 0)

    def test_get_demands(self):
        inst = PackagingInstance.standard()
        demands = inst.get_demands()
        assert len(demands) == 4
        assert np.all(demands > 0)

    def test_sheets_fit_in_roll(self):
        inst = PackagingInstance.standard()
        for s in inst.sheet_types:
            assert s.length_cm <= inst.roll_length_cm


class TestColdStorageSolving:
    """Test cold storage bin packing solutions."""

    def test_all_methods_return_results(self):
        inst = ColdStorageInstance.packing_house()
        results = _solver_mod.solve_cold_storage(inst)
        for method in ["FF", "FFD", "BFD", "GA"]:
            assert method in results

    def test_at_least_one_unit(self):
        inst = ColdStorageInstance.packing_house()
        results = _solver_mod.solve_cold_storage(inst)
        for sol in results.values():
            assert sol.n_units >= 1

    def test_ffd_le_ff(self):
        inst = ColdStorageInstance.packing_house()
        results = _solver_mod.solve_cold_storage(inst)
        assert results["FFD"].n_units <= results["FF"].n_units

    def test_all_lots_assigned(self):
        inst = ColdStorageInstance.packing_house()
        results = _solver_mod.solve_cold_storage(inst)
        for sol in results.values():
            all_items = set()
            for bin_items in sol.bins:
                all_items.update(bin_items)
            assert len(all_items) == inst.n_lots

    def test_no_bin_exceeds_capacity(self):
        inst = ColdStorageInstance.packing_house()
        weights = inst.get_weights()
        results = _solver_mod.solve_cold_storage(inst)
        for sol in results.values():
            for bin_items in sol.bins:
                bin_weight = sum(weights[j] for j in bin_items)
                assert bin_weight <= inst.storage_capacity_kg + 0.01

    def test_lower_bound(self):
        inst = ColdStorageInstance.packing_house()
        results = _solver_mod.solve_cold_storage(inst)
        lb = int(np.ceil(inst.total_weight / inst.storage_capacity_kg))
        for sol in results.values():
            assert sol.n_units >= lb

    def test_method_labels(self):
        inst = ColdStorageInstance.packing_house()
        results = _solver_mod.solve_cold_storage(inst)
        for name, sol in results.items():
            assert sol.method == name


class TestPackagingSolving:
    """Test packaging film cutting stock solutions."""

    def test_all_methods_return_results(self):
        inst = PackagingInstance.standard()
        results = _solver_mod.solve_packaging(inst)
        assert "Greedy" in results
        assert "FFD" in results

    def test_at_least_one_roll(self):
        inst = PackagingInstance.standard()
        results = _solver_mod.solve_packaging(inst)
        for sol in results.values():
            assert sol.n_rolls >= 1

    def test_waste_non_negative(self):
        inst = PackagingInstance.standard()
        results = _solver_mod.solve_packaging(inst)
        for sol in results.values():
            assert sol.waste_cm >= 0
            assert sol.waste_pct >= 0

    def test_waste_pct_reasonable(self):
        inst = PackagingInstance.standard()
        results = _solver_mod.solve_packaging(inst)
        for sol in results.values():
            assert sol.waste_pct < 50  # Less than 50% waste

    def test_patterns_present(self):
        inst = PackagingInstance.standard()
        results = _solver_mod.solve_packaging(inst)
        for sol in results.values():
            assert len(sol.patterns) >= 1

    def test_method_labels(self):
        inst = PackagingInstance.standard()
        results = _solver_mod.solve_packaging(inst)
        for name, sol in results.items():
            assert sol.method == name


class TestIntegration:
    """Integration tests combining both problems."""

    def test_both_problems_solvable(self):
        cs = ColdStorageInstance.packing_house()
        pkg = PackagingInstance.standard()
        cs_results = _solver_mod.solve_cold_storage(cs)
        pkg_results = _solver_mod.solve_packaging(pkg)
        assert len(cs_results) >= 3
        assert len(pkg_results) >= 2

    def test_repr_cold_storage(self):
        inst = ColdStorageInstance.packing_house()
        results = _solver_mod.solve_cold_storage(inst)
        for sol in results.values():
            r = repr(sol)
            assert "ColdStorageSolution" in r

    def test_repr_packaging(self):
        inst = PackagingInstance.standard()
        results = _solver_mod.solve_packaging(inst)
        for sol in results.values():
            r = repr(sol)
            assert "PackagingSolution" in r
