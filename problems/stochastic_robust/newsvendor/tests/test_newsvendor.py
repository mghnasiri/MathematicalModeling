"""Tests for the Newsvendor Problem."""
from __future__ import annotations

import sys
import os
import importlib.util

import numpy as np
import pytest

# Load modules via importlib to avoid naming collisions
def _load_mod(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_base = os.path.join(os.path.dirname(__file__), "..")
_inst_mod = _load_mod("nv_instance", os.path.join(_base, "instance.py"))
_cf_mod = _load_mod("nv_critical_fractile", os.path.join(_base, "exact", "critical_fractile.py"))
_mp_mod = _load_mod("nv_multi_product", os.path.join(_base, "heuristics", "multi_product.py"))

NewsvendorInstance = _inst_mod.NewsvendorInstance
NewsvendorSolution = _inst_mod.NewsvendorSolution
critical_fractile = _cf_mod.critical_fractile
grid_search = _cf_mod.grid_search
MultiProductInstance = _mp_mod.MultiProductInstance
marginal_allocation = _mp_mod.marginal_allocation
independent_then_scale = _mp_mod.independent_then_scale


class TestNewsvendorInstance:
    """Test instance creation and cost computation."""

    def test_critical_fractile_value(self):
        inst = NewsvendorInstance(
            unit_cost=5.0, selling_price=10.0, salvage_value=2.0,
            demand_scenarios=np.array([50, 60, 70, 80, 90]),
        )
        # c_u = 10 - 5 = 5, c_o = 5 - 2 = 3, CF = 5/8 = 0.625
        assert abs(inst.critical_fractile - 0.625) < 1e-10

    def test_overage_underage_costs(self):
        inst = NewsvendorInstance(
            unit_cost=6.0, selling_price=15.0, salvage_value=1.0,
            demand_scenarios=np.array([100]),
        )
        assert inst.overage_cost == 5.0
        assert inst.underage_cost == 9.0

    def test_expected_cost_at_extremes(self):
        inst = NewsvendorInstance(
            unit_cost=5.0, selling_price=10.0, salvage_value=2.0,
            demand_scenarios=np.array([100]),
        )
        # Order exactly demand: zero cost
        assert inst.expected_cost(100) == 0.0
        # Order too many: overage cost
        assert inst.expected_cost(110) == pytest.approx(10 * 3.0)  # 10 * c_o
        # Order too few: underage cost
        assert inst.expected_cost(90) == pytest.approx(10 * 5.0)  # 10 * c_u

    def test_expected_profit(self):
        inst = NewsvendorInstance(
            unit_cost=5.0, selling_price=10.0, salvage_value=0.0,
            demand_scenarios=np.array([100]),
        )
        # Q=100, D=100: profit = 10*100 - 5*100 = 500
        assert inst.expected_profit(100) == pytest.approx(500.0)

    def test_uniform_probabilities_default(self):
        inst = NewsvendorInstance(
            unit_cost=5.0, selling_price=10.0, salvage_value=0.0,
            demand_scenarios=np.array([10, 20, 30, 40, 50]),
        )
        assert len(inst.probabilities) == 5
        assert abs(sum(inst.probabilities) - 1.0) < 1e-10

    def test_random_instance(self):
        inst = NewsvendorInstance.random(n_scenarios=100, seed=7)
        assert inst.n_scenarios == 100
        assert inst.selling_price > inst.unit_cost > inst.salvage_value


class TestCriticalFractile:
    """Test the critical fractile exact solution."""

    def test_symmetric_demand(self):
        """With symmetric demand and CF=0.5, Q* should be median."""
        inst = NewsvendorInstance(
            unit_cost=5.0, selling_price=10.0, salvage_value=0.0,
            demand_scenarios=np.array([10, 20, 30, 40, 50]),
        )
        # CF = 5/(5+5) = 0.5, should pick median = 30
        sol = critical_fractile(inst)
        assert sol.order_quantity == pytest.approx(30.0)

    def test_high_underage_cost(self):
        """High underage cost -> order more (high service level)."""
        inst = NewsvendorInstance(
            unit_cost=2.0, selling_price=20.0, salvage_value=1.0,
            demand_scenarios=np.array([10, 20, 30, 40, 50]),
        )
        # CF = 18/19 ≈ 0.947 -> should order near max
        sol = critical_fractile(inst)
        assert sol.order_quantity >= 40.0

    def test_high_overage_cost(self):
        """High overage cost -> order less."""
        inst = NewsvendorInstance(
            unit_cost=9.0, selling_price=10.0, salvage_value=0.0,
            demand_scenarios=np.array([10, 20, 30, 40, 50]),
        )
        # CF = 1/10 = 0.1 -> should order near min
        sol = critical_fractile(inst)
        assert sol.order_quantity <= 20.0

    def test_matches_grid_search(self):
        """Critical fractile and grid search should give similar results."""
        inst = NewsvendorInstance(
            unit_cost=5.0, selling_price=12.0, salvage_value=1.0,
            demand_scenarios=np.arange(10, 101, 1.0),
        )
        sol_cf = critical_fractile(inst)
        sol_gs = grid_search(inst, n_points=5000)
        assert abs(sol_cf.expected_cost - sol_gs.expected_cost) < 2.0


class TestMultiProduct:
    """Test multi-product newsvendor heuristics."""

    def _make_instance(self):
        products = [
            NewsvendorInstance(
                unit_cost=5.0, selling_price=12.0, salvage_value=1.0,
                demand_scenarios=np.array([20, 30, 40, 50, 60]),
            ),
            NewsvendorInstance(
                unit_cost=3.0, selling_price=8.0, salvage_value=0.5,
                demand_scenarios=np.array([15, 25, 35, 45, 55]),
            ),
        ]
        return MultiProductInstance(products=products, budget=250.0)

    def test_marginal_allocation_budget(self):
        mp = self._make_instance()
        sol = marginal_allocation(mp, step=5.0)
        assert sol.total_cost <= mp.budget + 1e-6
        assert all(q >= 0 for q in sol.order_quantities)

    def test_independent_scale_budget(self):
        mp = self._make_instance()
        sol = independent_then_scale(mp)
        assert sol.total_cost <= mp.budget + 1e-6

    def test_marginal_positive_profit(self):
        mp = self._make_instance()
        sol = marginal_allocation(mp, step=5.0)
        assert sol.total_expected_profit > 0
