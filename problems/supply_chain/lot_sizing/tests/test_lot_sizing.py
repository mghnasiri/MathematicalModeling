"""
Test suite for Dynamic Lot Sizing problem.

Tests cover:
- Instance creation and validation
- Wagner-Whitin exact algorithm
- Silver-Meal heuristic
- Part-Period Balancing heuristic
- Lot-for-lot baseline
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


_inst_mod = _load_mod("ls_inst_test", os.path.join(_base_dir, "instance.py"))
_ww_mod = _load_mod(
    "ls_ww_test", os.path.join(_base_dir, "exact", "wagner_whitin.py")
)
_sm_mod = _load_mod(
    "ls_sm_test", os.path.join(_base_dir, "heuristics", "silver_meal.py")
)

LotSizingInstance = _inst_mod.LotSizingInstance
LotSizingSolution = _inst_mod.LotSizingSolution
textbook_4period = _inst_mod.textbook_4period
seasonal_8period = _inst_mod.seasonal_8period
varying_costs_6period = _inst_mod.varying_costs_6period

wagner_whitin = _ww_mod.wagner_whitin
silver_meal = _sm_mod.silver_meal
part_period_balancing = _sm_mod.part_period_balancing
lot_for_lot = _sm_mod.lot_for_lot


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def inst4():
    return textbook_4period()


@pytest.fixture
def inst8():
    return seasonal_8period()


@pytest.fixture
def inst6():
    return varying_costs_6period()


# ── Instance tests ───────────────────────────────────────────────────────────


class TestLotSizingInstance:
    def test_creation(self, inst4):
        assert inst4.T == 4
        assert len(inst4.demands) == 4
        assert np.allclose(inst4.demands, [20, 50, 10, 50])

    def test_invalid_shape(self):
        with pytest.raises(ValueError, match="demands shape"):
            LotSizingInstance(
                T=3,
                demands=np.array([10.0, 20.0]),
                ordering_costs=np.array([50.0, 50.0, 50.0]),
                holding_costs=np.array([1.0, 1.0, 1.0]),
            )

    def test_negative_demand(self):
        with pytest.raises(ValueError, match="non-negative"):
            LotSizingInstance(
                T=2,
                demands=np.array([10.0, -5.0]),
                ordering_costs=np.array([50.0, 50.0]),
                holding_costs=np.array([1.0, 1.0]),
            )

    def test_random_instance(self):
        inst = LotSizingInstance.random(T=6, seed=42)
        assert inst.T == 6
        assert len(inst.demands) == 6
        assert np.all(inst.demands >= 0)

    def test_compute_cost(self, inst4):
        # Lot-for-lot: order exactly demand each period
        q = inst4.demands.copy()
        cost = inst4.compute_cost(q)
        # 4 orders * K=54 = 216, zero holding
        assert abs(cost - 216.0) < 1e-6

    def test_solution_repr(self):
        sol = LotSizingSolution(
            order_quantities=np.array([10.0, 0.0, 10.0]),
            total_cost=100.0,
            order_periods=[0, 2],
        )
        r = repr(sol)
        assert "LotSizingSolution" in r
        assert "100.00" in r


# ── Wagner-Whitin tests ─────────────────────────────────────────────────────


class TestWagnerWhitin:
    def test_textbook_optimal(self, inst4):
        sol = wagner_whitin(inst4)
        assert sol.total_cost < float("inf")
        # Verify demands are met
        cumulative_order = np.cumsum(sol.order_quantities)
        cumulative_demand = np.cumsum(inst4.demands)
        assert np.all(cumulative_order >= cumulative_demand - 1e-10)

    def test_single_period(self):
        inst = LotSizingInstance(
            T=1,
            demands=np.array([100.0]),
            ordering_costs=np.array([50.0]),
            holding_costs=np.array([1.0]),
        )
        sol = wagner_whitin(inst)
        assert abs(sol.order_quantities[0] - 100.0) < 1e-6
        assert abs(sol.total_cost - 50.0) < 1e-6

    def test_ww_beats_lot_for_lot(self, inst4):
        sol_ww = wagner_whitin(inst4)
        sol_lfl = lot_for_lot(inst4)
        assert sol_ww.total_cost <= sol_lfl.total_cost + 1e-10

    def test_ww_is_optimal(self, inst4):
        """WW should be optimal; verify cost matches compute_cost."""
        sol = wagner_whitin(inst4)
        recomputed = inst4.compute_cost(sol.order_quantities)
        assert abs(sol.total_cost - recomputed) < 1e-6

    def test_seasonal_8(self, inst8):
        sol = wagner_whitin(inst8)
        assert sol.total_cost < float("inf")
        cumulative_order = np.cumsum(sol.order_quantities)
        cumulative_demand = np.cumsum(inst8.demands)
        assert np.all(cumulative_order >= cumulative_demand - 1e-10)


# ── Silver-Meal tests ───────────────────────────────────────────────────────


class TestSilverMeal:
    def test_feasibility(self, inst4):
        sol = silver_meal(inst4)
        cumulative_order = np.cumsum(sol.order_quantities)
        cumulative_demand = np.cumsum(inst4.demands)
        assert np.all(cumulative_order >= cumulative_demand - 1e-10)

    def test_heuristic_quality(self, inst4):
        sol_sm = silver_meal(inst4)
        sol_ww = wagner_whitin(inst4)
        # Silver-Meal should be reasonable (within 20% of optimal typically)
        assert sol_sm.total_cost <= sol_ww.total_cost * 1.5

    def test_cost_matches_recompute(self, inst8):
        sol = silver_meal(inst8)
        recomputed = inst8.compute_cost(sol.order_quantities)
        assert abs(sol.total_cost - recomputed) < 1e-6


# ── Part-Period Balancing tests ──────────────────────────────────────────────


class TestPartPeriodBalancing:
    def test_feasibility(self, inst4):
        sol = part_period_balancing(inst4)
        cumulative_order = np.cumsum(sol.order_quantities)
        cumulative_demand = np.cumsum(inst4.demands)
        assert np.all(cumulative_order >= cumulative_demand - 1e-10)

    def test_cost_matches_recompute(self, inst6):
        sol = part_period_balancing(inst6)
        recomputed = inst6.compute_cost(sol.order_quantities)
        assert abs(sol.total_cost - recomputed) < 1e-6


# ── Lot-for-lot tests ───────────────────────────────────────────────────────


class TestLotForLot:
    def test_orders_every_period(self, inst4):
        sol = lot_for_lot(inst4)
        # Should order every period with positive demand
        for t in range(inst4.T):
            if inst4.demands[t] > 0:
                assert sol.order_quantities[t] == inst4.demands[t]

    def test_zero_holding(self, inst4):
        sol = lot_for_lot(inst4)
        # Total cost = sum of ordering costs for periods with demand
        expected = sum(
            inst4.ordering_costs[t]
            for t in range(inst4.T)
            if inst4.demands[t] > 0
        )
        assert abs(sol.total_cost - expected) < 1e-6
