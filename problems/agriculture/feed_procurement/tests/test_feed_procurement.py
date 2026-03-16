"""
Tests for Agricultural Feed & Fertilizer Procurement Planning Problem

Covers: instance creation, EOQ baseline, Silver-Meal heuristic,
Wagner-Whitin optimal DP, cost comparisons.

31 tests across 6 test classes.
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
_inst_mod = _load_mod("fp_inst_test", os.path.join(_base, "instance.py"))
_heur_mod = _load_mod(
    "fp_heur_test",
    os.path.join(_base, "heuristics", "procurement_planning.py"),
)

FeedProcurementInstance = _inst_mod.FeedProcurementInstance
FarmInputProfile = _inst_mod.FarmInputProfile


class TestFarmInputProfile:
    """Test FarmInputProfile dataclass."""

    def test_total_demand(self):
        inp = FarmInputProfile(
            "Feed", "tons", 150.0, 2.0,
            np.array([75, 72, 65, 55, 45, 40, 42, 45, 50, 58, 70, 80]),
        )
        assert inp.total_annual_demand == pytest.approx(697.0)

    def test_avg_monthly_demand(self):
        inp = FarmInputProfile(
            "Feed", "tons", 150.0, 2.0,
            np.array([75, 72, 65, 55, 45, 40, 42, 45, 50, 58, 70, 80]),
        )
        assert inp.avg_monthly_demand == pytest.approx(697.0 / 12.0, rel=1e-4)

    def test_active_months_all(self):
        inp = FarmInputProfile(
            "Feed", "tons", 150.0, 2.0,
            np.array([75, 72, 65, 55, 45, 40, 42, 45, 50, 58, 70, 80]),
        )
        assert inp.n_active_months == 12

    def test_active_months_seasonal(self):
        inp = FarmInputProfile(
            "Fertilizer", "tons", 200.0, 5.0,
            np.array([0, 0, 15, 45, 60, 40, 25, 10, 0, 0, 0, 0]),
        )
        assert inp.n_active_months == 6

    def test_avg_monthly_demand_seasonal(self):
        inp = FarmInputProfile(
            "Fertilizer", "tons", 200.0, 5.0,
            np.array([0, 0, 15, 45, 60, 40, 25, 10, 0, 0, 0, 0]),
        )
        # Average over non-zero months only
        assert inp.avg_monthly_demand == pytest.approx(195.0 / 6.0, rel=1e-4)


class TestFeedProcurementInstance:
    """Test instance creation."""

    def test_quebec_farm_creation(self):
        inst = FeedProcurementInstance.quebec_dairy_farm()
        assert inst.n_inputs == 3
        assert inst.horizon == 12

    def test_quebec_farm_input_names(self):
        inst = FeedProcurementInstance.quebec_dairy_farm()
        names = [inp.name for inp in inst.inputs]
        assert any("Feed" in n for n in names)
        assert any("Fertilizer" in n for n in names)
        assert any("Seeds" in n for n in names)

    def test_random_instance(self):
        inst = FeedProcurementInstance.random(n_inputs=3, seed=42)
        assert inst.n_inputs == 3
        assert inst.horizon == 12

    def test_custom_instance(self):
        inputs = [
            FarmInputProfile("A", "kg", 50.0, 1.0, np.array([10] * 12)),
        ]
        inst = FeedProcurementInstance(inputs=inputs)
        assert inst.n_inputs == 1


class TestEOQProcurement:
    """Test EOQ baseline solution."""

    def test_returns_solution(self):
        inst = FeedProcurementInstance.quebec_dairy_farm()
        sol = _heur_mod.eoq_procurement(inst)
        assert type(sol).__name__ == "FeedProcurementSolution"

    def test_all_inputs_solved(self):
        inst = FeedProcurementInstance.quebec_dairy_farm()
        sol = _heur_mod.eoq_procurement(inst)
        assert len(sol.input_solutions) == 3

    def test_positive_costs(self):
        inst = FeedProcurementInstance.quebec_dairy_farm()
        sol = _heur_mod.eoq_procurement(inst)
        assert sol.total_cost > 0
        for isol in sol.input_solutions.values():
            assert isol.total_cost > 0

    def test_orders_placed(self):
        inst = FeedProcurementInstance.quebec_dairy_farm()
        sol = _heur_mod.eoq_procurement(inst)
        for isol in sol.input_solutions.values():
            assert isol.num_orders >= 1

    def test_method_label(self):
        inst = FeedProcurementInstance.quebec_dairy_farm()
        sol = _heur_mod.eoq_procurement(inst)
        assert sol.method == "EOQ"


class TestSilverMealProcurement:
    """Test Silver-Meal heuristic solution."""

    def test_returns_solution(self):
        inst = FeedProcurementInstance.quebec_dairy_farm()
        sol = _heur_mod.silver_meal_procurement(inst)
        assert type(sol).__name__ == "FeedProcurementSolution"

    def test_all_inputs_solved(self):
        inst = FeedProcurementInstance.quebec_dairy_farm()
        sol = _heur_mod.silver_meal_procurement(inst)
        assert len(sol.input_solutions) == 3

    def test_positive_costs(self):
        inst = FeedProcurementInstance.quebec_dairy_farm()
        sol = _heur_mod.silver_meal_procurement(inst)
        assert sol.total_cost > 0

    def test_method_label(self):
        inst = FeedProcurementInstance.quebec_dairy_farm()
        sol = _heur_mod.silver_meal_procurement(inst)
        assert sol.method == "Silver-Meal"


class TestWagnerWhitinProcurement:
    """Test Wagner-Whitin optimal DP solution."""

    def test_returns_solution(self):
        inst = FeedProcurementInstance.quebec_dairy_farm()
        sol = _heur_mod.wagner_whitin_procurement(inst)
        assert type(sol).__name__ == "FeedProcurementSolution"

    def test_all_inputs_solved(self):
        inst = FeedProcurementInstance.quebec_dairy_farm()
        sol = _heur_mod.wagner_whitin_procurement(inst)
        assert len(sol.input_solutions) == 3

    def test_positive_costs(self):
        inst = FeedProcurementInstance.quebec_dairy_farm()
        sol = _heur_mod.wagner_whitin_procurement(inst)
        assert sol.total_cost > 0

    def test_method_label(self):
        inst = FeedProcurementInstance.quebec_dairy_farm()
        sol = _heur_mod.wagner_whitin_procurement(inst)
        assert sol.method == "Wagner-Whitin"


class TestCostComparison:
    """Compare cost across methods."""

    def test_ww_le_eoq(self):
        """Wagner-Whitin (optimal) should be no worse than EOQ."""
        inst = FeedProcurementInstance.quebec_dairy_farm()
        eoq_sol = _heur_mod.eoq_procurement(inst)
        ww_sol = _heur_mod.wagner_whitin_procurement(inst)
        assert ww_sol.total_cost <= eoq_sol.total_cost + 1.0

    def test_all_methods_produce_results(self):
        """All three methods should produce valid results."""
        inst = FeedProcurementInstance.quebec_dairy_farm()
        sm_sol = _heur_mod.silver_meal_procurement(inst)
        ww_sol = _heur_mod.wagner_whitin_procurement(inst)
        assert sm_sol.total_cost > 0
        assert ww_sol.total_cost > 0

    def test_total_cost_is_sum(self):
        inst = FeedProcurementInstance.quebec_dairy_farm()
        sol = _heur_mod.wagner_whitin_procurement(inst)
        sum_cost = sum(isol.total_cost for isol in sol.input_solutions.values())
        assert sol.total_cost == pytest.approx(sum_cost, rel=1e-6)

    def test_repr(self):
        inst = FeedProcurementInstance.quebec_dairy_farm()
        sol = _heur_mod.eoq_procurement(inst)
        r = repr(sol)
        assert "FeedProcurementSolution" in r
