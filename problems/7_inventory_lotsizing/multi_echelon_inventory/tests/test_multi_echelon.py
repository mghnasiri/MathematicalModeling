"""
Test suite for Multi-Echelon Inventory problem.

Tests cover:
- Instance creation and validation
- Echelon lead time computation
- Demand during lead time statistics
- Echelon base-stock policy
- Powers-of-two policy
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


_inst_mod = _load_mod("mei_inst_test", os.path.join(_base_dir, "instance.py"))
_bs_mod = _load_mod(
    "mei_bs_test", os.path.join(_base_dir, "heuristics", "base_stock.py")
)

MultiEchelonInstance = _inst_mod.MultiEchelonInstance
MultiEchelonSolution = _inst_mod.MultiEchelonSolution
serial_2echelon = _inst_mod.serial_2echelon
serial_3echelon = _inst_mod.serial_3echelon
high_variance_3echelon = _inst_mod.high_variance_3echelon

echelon_base_stock = _bs_mod.echelon_base_stock
powers_of_two = _bs_mod.powers_of_two


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def inst2():
    return serial_2echelon()


@pytest.fixture
def inst3():
    return serial_3echelon()


@pytest.fixture
def inst_hv():
    return high_variance_3echelon()


# ── Instance tests ───────────────────────────────────────────────────────────


class TestMultiEchelonInstance:
    def test_creation(self, inst2):
        assert inst2.L == 2
        assert len(inst2.holding_costs) == 2
        assert inst2.mean_demand == 50.0
        assert inst2.std_demand == 10.0

    def test_invalid_shape(self):
        with pytest.raises(ValueError, match="holding_costs shape"):
            MultiEchelonInstance(
                L=3,
                holding_costs=np.array([10.0, 5.0]),
                ordering_costs=np.array([100.0, 150.0, 200.0]),
                lead_times=np.array([1.0, 2.0, 3.0]),
                mean_demand=100.0,
                std_demand=20.0,
            )

    def test_invalid_demand(self):
        with pytest.raises(ValueError, match="mean_demand"):
            MultiEchelonInstance(
                L=2,
                holding_costs=np.array([10.0, 5.0]),
                ordering_costs=np.array([100.0, 150.0]),
                lead_times=np.array([1.0, 2.0]),
                mean_demand=-10.0,
                std_demand=5.0,
            )

    def test_invalid_service_level(self):
        with pytest.raises(ValueError, match="service_level"):
            MultiEchelonInstance(
                L=2,
                holding_costs=np.array([10.0, 5.0]),
                ordering_costs=np.array([100.0, 150.0]),
                lead_times=np.array([1.0, 2.0]),
                mean_demand=100.0,
                std_demand=20.0,
                service_level=1.5,
            )

    def test_random_instance(self):
        inst = MultiEchelonInstance.random(L=4, seed=42)
        assert inst.L == 4
        assert len(inst.holding_costs) == 4
        assert inst.mean_demand > 0

    def test_echelon_lead_time(self, inst3):
        # Echelon 0: LT=1
        assert abs(inst3.echelon_lead_time(0) - 1.0) < 1e-10
        # Echelon 1: LT=1+2=3
        assert abs(inst3.echelon_lead_time(1) - 3.0) < 1e-10
        # Echelon 2: LT=1+2+3=6
        assert abs(inst3.echelon_lead_time(2) - 6.0) < 1e-10

    def test_demand_during_lead_time(self, inst3):
        mu, sigma = inst3.demand_during_lead_time(0)
        # Echelon 0: LT=1, mu=100*1=100, sigma=25*sqrt(1)=25
        assert abs(mu - 100.0) < 1e-10
        assert abs(sigma - 25.0) < 1e-10

        mu2, sigma2 = inst3.demand_during_lead_time(2)
        # Echelon 2: LT=6, mu=100*6=600, sigma=25*sqrt(6)
        assert abs(mu2 - 600.0) < 1e-10
        assert abs(sigma2 - 25.0 * np.sqrt(6.0)) < 1e-6

    def test_solution_repr(self):
        sol = MultiEchelonSolution(
            base_stock_levels=np.array([100.0, 200.0]),
            safety_stocks=np.array([30.0, 45.0]),
            total_holding_cost=500.0,
            total_cost=750.0,
        )
        r = repr(sol)
        assert "MultiEchelonSolution" in r
        assert "750.00" in r


# ── Echelon base-stock tests ────────────────────────────────────────────────


class TestEchelonBaseStock:
    def test_base_stocks_positive(self, inst3):
        sol = echelon_base_stock(inst3)
        assert np.all(sol.base_stock_levels > 0)

    def test_base_stocks_increasing(self, inst3):
        """Upstream echelons should have higher base-stock (more demand coverage)."""
        sol = echelon_base_stock(inst3)
        for i in range(1, inst3.L):
            assert sol.base_stock_levels[i] > sol.base_stock_levels[i - 1]

    def test_safety_stocks_positive(self, inst3):
        sol = echelon_base_stock(inst3)
        assert np.all(sol.safety_stocks > 0)

    def test_total_cost_positive(self, inst2):
        sol = echelon_base_stock(inst2)
        assert sol.total_cost > 0
        assert sol.total_holding_cost > 0

    def test_higher_service_more_safety(self):
        inst_low = MultiEchelonInstance(
            L=2,
            holding_costs=np.array([10.0, 5.0]),
            ordering_costs=np.array([100.0, 150.0]),
            lead_times=np.array([1.0, 2.0]),
            mean_demand=50.0,
            std_demand=10.0,
            service_level=0.90,
        )
        inst_high = MultiEchelonInstance(
            L=2,
            holding_costs=np.array([10.0, 5.0]),
            ordering_costs=np.array([100.0, 150.0]),
            lead_times=np.array([1.0, 2.0]),
            mean_demand=50.0,
            std_demand=10.0,
            service_level=0.99,
        )
        sol_low = echelon_base_stock(inst_low)
        sol_high = echelon_base_stock(inst_high)
        assert np.all(sol_high.safety_stocks > sol_low.safety_stocks)


# ── Powers-of-two tests ─────────────────────────────────────────────────────


class TestPowersOfTwo:
    def test_base_stocks_positive(self, inst3):
        sol = powers_of_two(inst3)
        assert np.all(sol.base_stock_levels > 0)

    def test_total_cost_positive(self, inst2):
        sol = powers_of_two(inst2)
        assert sol.total_cost > 0

    def test_high_variance(self, inst_hv):
        sol = powers_of_two(inst_hv)
        assert np.all(sol.safety_stocks > 0)
        assert sol.total_cost > 0

    def test_safety_stocks_positive(self, inst3):
        sol = powers_of_two(inst3)
        assert np.all(sol.safety_stocks > 0)
