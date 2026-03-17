"""
Test suite for Safety Stock Optimization.

Tests cover:
- Instance creation and validation
- Analytical safety stock computation
- sigma_DDLT formula verification
- Edge cases (zero variability, single item)
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


_inst_mod = _load_mod("ss_instance_test", os.path.join(_base_dir, "instance.py"))
_exact_mod = _load_mod(
    "ss_exact_test", os.path.join(_base_dir, "exact", "analytical_ss.py")
)

SafetyStockInstance = _inst_mod.SafetyStockInstance
SafetyStockSolution = _inst_mod.SafetyStockSolution
basic_3items = _inst_mod.basic_3items
single_item = _inst_mod.single_item
zero_lt_variability = _inst_mod.zero_lt_variability

analytical_safety_stock = _exact_mod.analytical_safety_stock
compute_sigma_ddlt = _exact_mod.compute_sigma_ddlt
safety_stock_fill_rate = _exact_mod.safety_stock_fill_rate


# -- Fixtures ------------------------------------------------------------------


@pytest.fixture
def inst3():
    return basic_3items()


@pytest.fixture
def inst1():
    return single_item()


@pytest.fixture
def inst_det_lt():
    return zero_lt_variability()


# -- Instance tests ------------------------------------------------------------


class TestSafetyStockInstance:
    def test_create_basic(self, inst3):
        assert inst3.n == 3
        assert inst3.mean_demands.shape == (3,)

    def test_random_instance(self):
        inst = SafetyStockInstance.random(8, seed=42)
        assert inst.n == 8
        assert np.all(inst.mean_demands >= 0)

    def test_invalid_shape(self):
        with pytest.raises(ValueError):
            SafetyStockInstance(
                n=2,
                mean_demands=np.array([10.0]),
                std_demands=np.array([5.0, 3.0]),
                mean_lead_times=np.array([1.0, 2.0]),
                std_lead_times=np.array([0.0, 0.0]),
                holding_costs=np.array([1.0, 2.0]),
            )

    def test_invalid_service_level(self):
        with pytest.raises(ValueError):
            SafetyStockInstance(
                n=1,
                mean_demands=np.array([10.0]),
                std_demands=np.array([2.0]),
                mean_lead_times=np.array([1.0]),
                std_lead_times=np.array([0.0]),
                holding_costs=np.array([1.0]),
                service_level=0.0,
            )


# -- Sigma DDLT tests ---------------------------------------------------------


class TestSigmaDDLT:
    def test_deterministic_lead_time(self):
        # sigma_L = 0 => sigma_DDLT = sigma_D * sqrt(L)
        sigma = compute_sigma_ddlt(100.0, 20.0, 4.0, 0.0)
        expected = 20.0 * np.sqrt(4.0)
        assert abs(sigma - expected) < 1e-10

    def test_deterministic_demand(self):
        # sigma_D = 0 => sigma_DDLT = D * sigma_L
        sigma = compute_sigma_ddlt(100.0, 0.0, 4.0, 1.5)
        expected = 100.0 * 1.5
        assert abs(sigma - expected) < 1e-10

    def test_both_variability(self):
        sigma = compute_sigma_ddlt(100.0, 20.0, 4.0, 1.0)
        expected = np.sqrt(4.0 * 400.0 + 10000.0 * 1.0)
        assert abs(sigma - expected) < 1e-10


# -- Analytical safety stock tests --------------------------------------------


class TestAnalyticalSafetyStock:
    def test_positive_safety_stocks(self, inst3):
        sol = analytical_safety_stock(inst3)
        assert np.all(sol.safety_stocks > 0)

    def test_reorder_point_above_mean_ddlt(self, inst1):
        sol = analytical_safety_stock(inst1)
        mean_ddlt = inst1.mean_demands[0] * inst1.mean_lead_times[0]
        assert sol.reorder_points[0] > mean_ddlt

    def test_higher_service_more_stock(self):
        inst_low = SafetyStockInstance(
            n=1, mean_demands=np.array([100.0]),
            std_demands=np.array([20.0]),
            mean_lead_times=np.array([2.0]),
            std_lead_times=np.array([0.5]),
            holding_costs=np.array([1.0]),
            service_level=0.90,
        )
        inst_high = SafetyStockInstance(
            n=1, mean_demands=np.array([100.0]),
            std_demands=np.array([20.0]),
            mean_lead_times=np.array([2.0]),
            std_lead_times=np.array([0.5]),
            holding_costs=np.array([1.0]),
            service_level=0.99,
        )
        sol_low = analytical_safety_stock(inst_low)
        sol_high = analytical_safety_stock(inst_high)
        assert sol_high.safety_stocks[0] > sol_low.safety_stocks[0]

    def test_zero_lt_variability(self, inst_det_lt):
        sol = analytical_safety_stock(inst_det_lt)
        for i in range(inst_det_lt.n):
            # sigma_DDLT = sigma_D * sqrt(L) when sigma_L = 0
            expected_sigma = (inst_det_lt.std_demands[i]
                              * np.sqrt(inst_det_lt.mean_lead_times[i]))
            from scipy.stats import norm
            z = norm.ppf(inst_det_lt.service_level)
            expected_ss = z * expected_sigma
            assert abs(sol.safety_stocks[i] - expected_ss) < 1e-6

    def test_cost_positive(self, inst3):
        sol = analytical_safety_stock(inst3)
        assert sol.total_holding_cost > 0

    def test_fill_rate_method(self, inst3):
        sol = safety_stock_fill_rate(inst3, target_fill_rate=0.95)
        assert np.all(sol.safety_stocks > 0)

    def test_repr(self, inst1):
        sol = analytical_safety_stock(inst1)
        r = repr(sol)
        assert "SafetyStockSolution" in r
