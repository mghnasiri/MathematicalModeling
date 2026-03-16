"""
Tests for Crop Rotation and Land Allocation Problem

Covers: instance creation, LP allocation, Pareto front,
constraint satisfaction, and solution quality.

30 tests across 5 test classes.
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
_inst_mod = _load_mod("cr_inst_test", os.path.join(_base, "instance.py"))
_solver_mod = _load_mod(
    "cr_solver_test",
    os.path.join(_base, "exact", "lp_allocation.py"),
)

CropRotationInstance = _inst_mod.CropRotationInstance
FieldProfile = _inst_mod.FieldProfile


class TestCropRotationInstance:
    """Test instance creation and data access."""

    def test_standard_farm_creation(self):
        inst = CropRotationInstance.standard_farm()
        assert inst.n_fields == 6
        assert inst.n_crops == 5

    def test_total_hectares(self):
        inst = CropRotationInstance.standard_farm()
        assert inst.total_hectares == 145

    def test_yield_by_soil_quality(self):
        inst = CropRotationInstance.standard_farm()
        # High soil should yield more than low
        corn_high = inst.get_yield("corn", 0)  # North Ridge = high
        corn_low = inst.get_yield("corn", 3)   # West Hill = low
        assert corn_high > corn_low

    def test_revenue_per_ha(self):
        inst = CropRotationInstance.standard_farm()
        rev = inst.get_revenue_per_ha("corn", 0)
        assert rev > 0

    def test_rotation_forbidden(self):
        inst = CropRotationInstance.standard_farm()
        assert len(inst.rotation_forbidden) == 3
        assert (0, "corn") in inst.rotation_forbidden

    def test_random_instance(self):
        inst = CropRotationInstance.random(n_fields=4, n_crops=3, seed=42)
        assert inst.n_fields == 4
        assert inst.n_crops == 3

    def test_crops_list(self):
        inst = CropRotationInstance.standard_farm()
        assert "corn" in inst.crops
        assert "soybeans" in inst.crops


class TestLPAllocation:
    """Test LP-based crop allocation."""

    def test_returns_solution(self):
        inst = CropRotationInstance.standard_farm()
        sol = _solver_mod.solve_lp_allocation(inst)
        assert type(sol).__name__ == "CropRotationSolution"

    def test_success(self):
        inst = CropRotationInstance.standard_farm()
        sol = _solver_mod.solve_lp_allocation(inst)
        assert sol.success is True

    def test_positive_revenue(self):
        inst = CropRotationInstance.standard_farm()
        sol = _solver_mod.solve_lp_allocation(inst)
        assert sol.total_revenue > 0

    def test_all_fields_allocated(self):
        inst = CropRotationInstance.standard_farm()
        sol = _solver_mod.solve_lp_allocation(inst)
        assert len(sol.allocation.field_crops) == inst.n_fields

    def test_allocated_crops_valid(self):
        inst = CropRotationInstance.standard_farm()
        sol = _solver_mod.solve_lp_allocation(inst)
        for crop in sol.allocation.field_crops.values():
            assert crop in inst.crops

    def test_water_within_budget(self):
        inst = CropRotationInstance.standard_farm()
        sol = _solver_mod.solve_lp_allocation(inst)
        assert sol.total_water <= inst.water_budget + 1.0

    def test_labor_within_budget(self):
        inst = CropRotationInstance.standard_farm()
        sol = _solver_mod.solve_lp_allocation(inst)
        assert sol.total_labor <= inst.labor_budget + 1.0

    def test_rotation_constraints_respected(self):
        inst = CropRotationInstance.standard_farm()
        sol = _solver_mod.solve_lp_allocation(inst)
        for field_idx, forbidden_crop in inst.rotation_forbidden:
            assigned = sol.allocation.field_crops[field_idx]
            assert assigned != forbidden_crop, \
                f"Field {field_idx} assigned forbidden crop {forbidden_crop}"

    def test_method_label(self):
        inst = CropRotationInstance.standard_farm()
        sol = _solver_mod.solve_lp_allocation(inst)
        assert "LP" in sol.method

    def test_repr(self):
        inst = CropRotationInstance.standard_farm()
        sol = _solver_mod.solve_lp_allocation(inst)
        r = repr(sol)
        assert "CropRotationSolution" in r


class TestParetoFront:
    """Test multi-objective Pareto front."""

    def test_returns_solution(self):
        inst = CropRotationInstance.standard_farm()
        sol = _solver_mod.solve_pareto_front(inst, n_points=10)
        assert type(sol).__name__ == "ParetoFrontSolution"

    def test_at_least_two_points(self):
        inst = CropRotationInstance.standard_farm()
        sol = _solver_mod.solve_pareto_front(inst, n_points=10)
        assert sol.n_points >= 2

    def test_points_and_allocations_match(self):
        inst = CropRotationInstance.standard_farm()
        sol = _solver_mod.solve_pareto_front(inst, n_points=10)
        assert len(sol.points) == len(sol.allocations)

    def test_nitrogen_range(self):
        inst = CropRotationInstance.standard_farm()
        sol = _solver_mod.solve_pareto_front(inst, n_points=10)
        min_n, max_n = sol.nitrogen_range
        assert max_n > min_n

    def test_sorted_by_revenue(self):
        inst = CropRotationInstance.standard_farm()
        sol = _solver_mod.solve_pareto_front(inst, n_points=10)
        revenues = [rev for rev, _ in sol.points]
        assert revenues == sorted(revenues)

    def test_pareto_efficient(self):
        """No point should be dominated by another."""
        inst = CropRotationInstance.standard_farm()
        sol = _solver_mod.solve_pareto_front(inst, n_points=15)
        for i, (rev_i, nit_i) in enumerate(sol.points):
            for j, (rev_j, nit_j) in enumerate(sol.points):
                if i != j:
                    # j should not dominate i
                    assert not (rev_j >= rev_i + 1e-3 and nit_j >= nit_i + 1e-3)

    def test_repr(self):
        inst = CropRotationInstance.standard_farm()
        sol = _solver_mod.solve_pareto_front(inst, n_points=10)
        r = repr(sol)
        assert "ParetoFrontSolution" in r


class TestConstraintValidation:
    """Test specific constraint properties."""

    def test_diversity_constraint(self):
        """No single crop should exceed max_crop_fraction of total area."""
        inst = CropRotationInstance.standard_farm()
        sol = _solver_mod.solve_lp_allocation(inst)
        max_area = inst.max_crop_fraction * inst.total_hectares
        crop_areas = {}
        for f, crop in sol.allocation.field_crops.items():
            frac = sol.allocation.fractions[f]
            ha = inst.fields[f].hectares * frac
            crop_areas[crop] = crop_areas.get(crop, 0) + ha
        for crop, area in crop_areas.items():
            assert area <= max_area + 1.0, \
                f"Crop {crop} uses {area:.0f} ha, max = {max_area:.0f}"


class TestRandomInstances:
    """Test with random instances."""

    def test_small_random_lp(self):
        inst = CropRotationInstance.random(n_fields=3, n_crops=2, seed=42)
        sol = _solver_mod.solve_lp_allocation(inst)
        assert sol.success

    def test_medium_random_lp(self):
        inst = CropRotationInstance.random(n_fields=6, n_crops=4, seed=99)
        sol = _solver_mod.solve_lp_allocation(inst)
        assert sol.success

    def test_random_pareto(self):
        inst = CropRotationInstance.random(n_fields=4, n_crops=3, seed=42)
        sol = _solver_mod.solve_pareto_front(inst, n_points=8)
        assert sol.n_points >= 1
