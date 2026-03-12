"""
Test suite for Wagner-Whitin Lot Sizing Problem.

Tests cover:
- Instance creation and validation
- Wagner-Whitin DP correctness on small instances
- Solution cost verification
- Edge cases (single period, zero demand)
"""

from __future__ import annotations

import os
import sys
import pytest
import numpy as np
import importlib.util

# -- Module loading ------------------------------------------------------------

_base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_mod(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_inst_mod = _load_mod("ww_instance_test", os.path.join(_base_dir, "instance.py"))
_dp_mod = _load_mod(
    "ww_dp_test", os.path.join(_base_dir, "exact", "wagner_whitin_dp.py")
)

WagnerWhitinInstance = _inst_mod.WagnerWhitinInstance
WagnerWhitinSolution = _inst_mod.WagnerWhitinSolution
textbook_4 = _inst_mod.textbook_4
seasonal_8 = _inst_mod.seasonal_8
single_period = _inst_mod.single_period

wagner_whitin_dp = _dp_mod.wagner_whitin_dp


# -- Fixtures ------------------------------------------------------------------


@pytest.fixture
def inst4():
    return textbook_4()


@pytest.fixture
def inst8():
    return seasonal_8()


@pytest.fixture
def inst1():
    return single_period()


# -- Instance tests ------------------------------------------------------------


class TestWagnerWhitinInstance:
    def test_create_basic(self, inst4):
        assert inst4.T == 4
        assert inst4.demands.shape == (4,)

    def test_random_instance(self):
        inst = WagnerWhitinInstance.random(10, seed=42)
        assert inst.T == 10
        assert np.all(inst.demands >= 0)

    def test_invalid_shape(self):
        with pytest.raises(ValueError):
            WagnerWhitinInstance(
                T=3,
                demands=np.array([10.0, 20.0]),
                ordering_costs=np.full(3, 50.0),
                holding_costs=np.full(3, 1.0),
            )

    def test_negative_demand(self):
        with pytest.raises(ValueError):
            WagnerWhitinInstance(
                T=2,
                demands=np.array([10.0, -5.0]),
                ordering_costs=np.full(2, 50.0),
                holding_costs=np.full(2, 1.0),
            )

    def test_compute_cost(self, inst4):
        # Order everything in period 0
        q = np.array([130.0, 0.0, 0.0, 0.0])
        cost = inst4.compute_cost(q)
        # K[0] + h[0]*110 + h[1]*60 + h[2]*50 = 54 + 110 + 60 + 50 = 274
        assert abs(cost - 274.0) < 1e-10


# -- DP tests ------------------------------------------------------------------


class TestWagnerWhitinDP:
    def test_textbook_optimal(self, inst4):
        sol = wagner_whitin_dp(inst4)
        # Verified cost must match
        verified = inst4.compute_cost(sol.order_quantities)
        assert abs(sol.total_cost - verified) < 1e-6

    def test_textbook_beats_lot_for_lot(self, inst4):
        sol = wagner_whitin_dp(inst4)
        # Lot-for-lot: order every period
        lfl = np.array([20.0, 50.0, 10.0, 50.0])
        lfl_cost = inst4.compute_cost(lfl)
        assert sol.total_cost <= lfl_cost + 1e-10

    def test_textbook_beats_single_order(self, inst4):
        sol = wagner_whitin_dp(inst4)
        single = np.array([130.0, 0.0, 0.0, 0.0])
        single_cost = inst4.compute_cost(single)
        assert sol.total_cost <= single_cost + 1e-10

    def test_single_period(self, inst1):
        sol = wagner_whitin_dp(inst1)
        assert len(sol.order_periods) == 1
        assert sol.order_periods[0] == 0
        assert abs(sol.order_quantities[0] - 30.0) < 1e-10
        assert abs(sol.total_cost - 50.0) < 1e-10

    def test_seasonal_feasible(self, inst8):
        sol = wagner_whitin_dp(inst8)
        # Verify all demand is met
        inventory = 0.0
        for t in range(inst8.T):
            inventory += sol.order_quantities[t] - inst8.demands[t]
            assert inventory >= -1e-10

    def test_zio_property(self, inst4):
        """Orders should only occur when inventory is zero."""
        sol = wagner_whitin_dp(inst4)
        inventory = 0.0
        for t in range(inst4.T):
            if sol.order_quantities[t] > 1e-10:
                assert inventory < 1e-10, f"Order at t={t} with inventory={inventory}"
            inventory += sol.order_quantities[t] - inst4.demands[t]

    def test_random_instance(self):
        inst = WagnerWhitinInstance.random(6, seed=99)
        sol = wagner_whitin_dp(inst)
        verified = inst.compute_cost(sol.order_quantities)
        assert abs(sol.total_cost - verified) < 1e-6

    def test_repr(self, inst4):
        sol = wagner_whitin_dp(inst4)
        r = repr(sol)
        assert "WagnerWhitinSolution" in r
