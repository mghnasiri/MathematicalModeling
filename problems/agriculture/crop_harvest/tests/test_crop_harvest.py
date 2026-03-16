"""
Tests for Perishable Crop Harvest Planning Problem

Covers: instance creation, critical fractile, marginal allocation,
independent+scale, budget feasibility, and solution quality.

37 tests across 6 test classes.
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
_inst_mod = _load_mod("ch_inst_test", os.path.join(_base, "instance.py"))
_heur_mod = _load_mod(
    "ch_heur_test",
    os.path.join(_base, "heuristics", "harvest_planning.py"),
)

CropHarvestInstance = _inst_mod.CropHarvestInstance
CropHarvestSolution = _inst_mod.CropHarvestSolution
CropProfile = _inst_mod.CropProfile


class TestCropProfile:
    """Test CropProfile dataclass and properties."""

    def test_critical_fractile_range(self):
        crop = CropProfile("Tomato", 1.50, 4.00, 0.50, 500.0, 100.0)
        cf = crop.critical_fractile
        assert 0 < cf < 1

    def test_overage_underage_costs(self):
        crop = CropProfile("Tomato", 1.50, 4.00, 0.50, 500.0, 100.0)
        assert crop.overage_cost == pytest.approx(1.0)  # 1.50 - 0.50
        assert crop.underage_cost == pytest.approx(2.5)  # 4.00 - 1.50

    def test_critical_fractile_high_margin(self):
        # High margin => high critical fractile (order more)
        crop = CropProfile("Herbs", 2.50, 12.00, 0.80, 80.0, 25.0)
        assert crop.critical_fractile > 0.8

    def test_critical_fractile_low_margin(self):
        # Low margin => lower critical fractile
        crop = CropProfile("Zucchini", 0.50, 1.80, 0.10, 400.0, 120.0)
        assert crop.critical_fractile < 0.85


class TestCropHarvestInstance:
    """Test instance creation and scenario generation."""

    def test_quebec_farm_creation(self):
        inst = CropHarvestInstance.quebec_farm()
        assert inst.n_crops == 8
        assert inst.daily_labor_budget == 2500.0

    def test_quebec_farm_crop_names(self):
        inst = CropHarvestInstance.quebec_farm()
        names = [c.name for c in inst.crops]
        assert "Tomatoes" in names
        assert "Strawberries" in names
        assert "Herbs" in names

    def test_scenario_generation_shape(self):
        inst = CropHarvestInstance.quebec_farm(n_scenarios=50)
        scenarios = inst.generate_scenarios(seed=42)
        assert scenarios.shape == (50, 8)

    def test_scenario_generation_non_negative(self):
        inst = CropHarvestInstance.quebec_farm(n_scenarios=100)
        scenarios = inst.generate_scenarios(seed=42)
        assert np.all(scenarios >= 0)

    def test_scenario_generation_reproducible(self):
        inst = CropHarvestInstance.quebec_farm()
        s1 = inst.generate_scenarios(seed=42)
        s2 = inst.generate_scenarios(seed=42)
        np.testing.assert_array_equal(s1, s2)

    def test_scenario_generation_different_seeds(self):
        inst = CropHarvestInstance.quebec_farm()
        s1 = inst.generate_scenarios(seed=42)
        s2 = inst.generate_scenarios(seed=99)
        assert not np.allclose(s1, s2)

    def test_random_instance(self):
        inst = CropHarvestInstance.random(n_crops=5, seed=42)
        assert inst.n_crops == 5
        assert inst.daily_labor_budget > 0

    def test_random_instance_valid_crops(self):
        inst = CropHarvestInstance.random(n_crops=3, seed=42)
        for crop in inst.crops:
            assert crop.selling_price > crop.unit_cost
            assert crop.unit_cost > crop.salvage_value
            assert crop.demand_mean > 0
            assert crop.demand_std > 0

    def test_custom_instance(self):
        crops = [
            CropProfile("A", 1.0, 3.0, 0.2, 100.0, 20.0),
            CropProfile("B", 2.0, 5.0, 0.5, 200.0, 50.0),
        ]
        inst = CropHarvestInstance(crops=crops, n_scenarios=30, daily_labor_budget=500.0)
        assert inst.n_crops == 2
        assert inst.n_scenarios == 30


class TestCriticalFractileHarvest:
    """Test unconstrained critical fractile solution."""

    def test_returns_solution(self):
        inst = CropHarvestInstance.quebec_farm()
        sol = _heur_mod.critical_fractile_harvest(inst, seed=42)
        assert type(sol).__name__ == "CropHarvestSolution"

    def test_all_quantities_positive(self):
        inst = CropHarvestInstance.quebec_farm()
        sol = _heur_mod.critical_fractile_harvest(inst, seed=42)
        for q in sol.harvest_quantities:
            assert q > 0

    def test_service_levels_valid(self):
        inst = CropHarvestInstance.quebec_farm()
        sol = _heur_mod.critical_fractile_harvest(inst, seed=42)
        for sl in sol.service_levels:
            assert 0 < sl <= 1.0

    def test_positive_profits(self):
        inst = CropHarvestInstance.quebec_farm()
        sol = _heur_mod.critical_fractile_harvest(inst, seed=42)
        assert sol.total_expected_profit > 0

    def test_total_profit_is_sum(self):
        inst = CropHarvestInstance.quebec_farm()
        sol = _heur_mod.critical_fractile_harvest(inst, seed=42)
        assert sol.total_expected_profit == pytest.approx(
            sum(sol.expected_profits), rel=1e-4
        )

    def test_method_label(self):
        inst = CropHarvestInstance.quebec_farm()
        sol = _heur_mod.critical_fractile_harvest(inst, seed=42)
        assert "Critical Fractile" in sol.method

    def test_reproducible(self):
        inst = CropHarvestInstance.quebec_farm()
        sol1 = _heur_mod.critical_fractile_harvest(inst, seed=42)
        sol2 = _heur_mod.critical_fractile_harvest(inst, seed=42)
        assert sol1.total_expected_profit == pytest.approx(
            sol2.total_expected_profit
        )


class TestMarginalAllocationHarvest:
    """Test budget-constrained marginal allocation."""

    def test_returns_solution(self):
        inst = CropHarvestInstance.quebec_farm()
        sol = _heur_mod.marginal_allocation_harvest(inst, seed=42)
        assert type(sol).__name__ == "CropHarvestSolution"

    def test_budget_feasible(self):
        inst = CropHarvestInstance.quebec_farm()
        sol = _heur_mod.marginal_allocation_harvest(inst, seed=42)
        assert sol.total_harvest_cost <= inst.daily_labor_budget + 1.0

    def test_budget_feasibility_flag(self):
        inst = CropHarvestInstance.quebec_farm()
        sol = _heur_mod.marginal_allocation_harvest(inst, seed=42)
        assert sol.budget_feasible == True

    def test_positive_profit(self):
        inst = CropHarvestInstance.quebec_farm()
        sol = _heur_mod.marginal_allocation_harvest(inst, seed=42)
        assert sol.total_expected_profit > 0

    def test_quantities_non_negative(self):
        inst = CropHarvestInstance.quebec_farm()
        sol = _heur_mod.marginal_allocation_harvest(inst, seed=42)
        for q in sol.harvest_quantities:
            assert q >= 0

    def test_method_label(self):
        inst = CropHarvestInstance.quebec_farm()
        sol = _heur_mod.marginal_allocation_harvest(inst, seed=42)
        assert "Marginal" in sol.method


class TestIndependentScaleHarvest:
    """Test independent-then-scale solution."""

    def test_returns_solution(self):
        inst = CropHarvestInstance.quebec_farm()
        sol = _heur_mod.independent_scale_harvest(inst, seed=42)
        assert type(sol).__name__ == "CropHarvestSolution"

    def test_budget_feasible(self):
        inst = CropHarvestInstance.quebec_farm()
        sol = _heur_mod.independent_scale_harvest(inst, seed=42)
        assert sol.total_harvest_cost <= inst.daily_labor_budget + 1.0

    def test_positive_profit(self):
        inst = CropHarvestInstance.quebec_farm()
        sol = _heur_mod.independent_scale_harvest(inst, seed=42)
        assert sol.total_expected_profit > 0

    def test_method_label(self):
        inst = CropHarvestInstance.quebec_farm()
        sol = _heur_mod.independent_scale_harvest(inst, seed=42)
        assert "Independent" in sol.method


class TestSolutionComparison:
    """Compare different solution methods."""

    def test_unconstrained_profit_ge_constrained(self):
        inst = CropHarvestInstance.quebec_farm()
        sol_cf = _heur_mod.critical_fractile_harvest(inst, seed=42)
        sol_ma = _heur_mod.marginal_allocation_harvest(inst, seed=42)
        # Unconstrained profit >= constrained (modulo budget)
        assert sol_cf.total_expected_profit >= sol_ma.total_expected_profit - 1.0

    def test_marginal_vs_independent(self):
        inst = CropHarvestInstance.quebec_farm()
        sol_ma = _heur_mod.marginal_allocation_harvest(inst, seed=42)
        sol_is = _heur_mod.independent_scale_harvest(inst, seed=42)
        # Both should produce positive profit
        assert sol_ma.total_expected_profit > 0
        assert sol_is.total_expected_profit > 0

    def test_small_instance_all_methods(self):
        crops = [
            CropProfile("A", 1.0, 3.0, 0.2, 100.0, 20.0),
            CropProfile("B", 2.0, 6.0, 0.5, 80.0, 15.0),
        ]
        inst = CropHarvestInstance(
            crops=crops, n_scenarios=30, daily_labor_budget=300.0,
        )
        sol_cf = _heur_mod.critical_fractile_harvest(inst, seed=42)
        sol_ma = _heur_mod.marginal_allocation_harvest(inst, step=2.0, seed=42)
        sol_is = _heur_mod.independent_scale_harvest(inst, seed=42)
        assert sol_cf.total_expected_profit > 0
        assert sol_ma.total_expected_profit > 0
        assert sol_is.total_expected_profit > 0

    def test_repr(self):
        inst = CropHarvestInstance.quebec_farm()
        sol = _heur_mod.critical_fractile_harvest(inst, seed=42)
        r = repr(sol)
        assert "CropHarvestSolution" in r
        assert "profit" in r
