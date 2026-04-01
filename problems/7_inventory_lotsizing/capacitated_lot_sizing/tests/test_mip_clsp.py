"""
Test suite for CLSP MIP solver.

Tests cover:
- Optimal solution on small handcrafted instances
- Feasibility and constraint satisfaction
- Comparison with heuristics (MIP should be at least as good)
- Edge cases (single period, zero demand, tight capacity)
- Random instances
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


_inst_mod = _load_mod("cls_inst_mip_test", os.path.join(_base_dir, "instance.py"))
_mip_mod = _load_mod(
    "cls_mip_test", os.path.join(_base_dir, "exact", "mip_clsp.py")
)
_greedy_mod = _load_mod(
    "cls_greedy_mip_test", os.path.join(_base_dir, "heuristics", "greedy_clsp.py")
)
_cls_mod = _load_mod(
    "cls_cls_mip_test", os.path.join(_base_dir, "heuristics", "greedy_cls.py")
)

CapLotSizingInstance = _inst_mod.CapLotSizingInstance
CapLotSizingSolution = _inst_mod.CapLotSizingSolution
tight_capacity_6 = _inst_mod.tight_capacity_6
loose_capacity_4 = _inst_mod.loose_capacity_4
variable_costs_8 = _inst_mod.variable_costs_8

mip_clsp = _mip_mod.mip_clsp
greedy_lot_sizing = _greedy_mod.greedy_lot_sizing
forward_greedy = _cls_mod.forward_greedy


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def tight6():
    return tight_capacity_6()


@pytest.fixture
def loose4():
    return loose_capacity_4()


@pytest.fixture
def var8():
    return variable_costs_8()


@pytest.fixture
def single_period():
    return CapLotSizingInstance(
        T=1,
        demands=np.array([50.0]),
        capacities=np.array([100.0]),
        fixed_costs=np.array([200.0]),
        variable_costs=np.array([3.0]),
        holding_costs=np.array([1.0]),
        name="single_1",
    )


@pytest.fixture
def two_period_known():
    """2-period instance with known optimal: produce all in period 0."""
    return CapLotSizingInstance(
        T=2,
        demands=np.array([30.0, 20.0]),
        capacities=np.array([60.0, 60.0]),
        fixed_costs=np.array([100.0, 100.0]),
        variable_costs=np.array([0.0, 0.0]),
        holding_costs=np.array([1.0, 1.0]),
        name="known_2",
    )


# ── Feasibility tests ───────────────────────────────────────────────────────


class TestMIPFeasibility:
    def test_feasible_tight6(self, tight6):
        sol = mip_clsp(tight6)
        assert tight6.is_feasible(sol.production)

    def test_feasible_loose4(self, loose4):
        sol = mip_clsp(loose4)
        assert loose4.is_feasible(sol.production)

    def test_feasible_var8(self, var8):
        sol = mip_clsp(var8)
        assert var8.is_feasible(sol.production)

    def test_capacity_respected(self, tight6):
        sol = mip_clsp(tight6)
        assert np.all(sol.production <= tight6.capacities + 1e-6)

    def test_demand_satisfied(self, tight6):
        sol = mip_clsp(tight6)
        cum_prod = np.cumsum(sol.production)
        cum_dem = np.cumsum(tight6.demands)
        assert np.all(cum_prod >= cum_dem - 1e-6)


# ── Optimality tests ────────────────────────────────────────────────────────


class TestMIPOptimality:
    def test_cost_matches_recompute(self, tight6):
        sol = mip_clsp(tight6)
        recomputed = tight6.compute_cost(sol.production)
        assert abs(sol.total_cost - recomputed) < 1e-4

    def test_beats_greedy(self, tight6):
        mip_sol = mip_clsp(tight6)
        greedy_sol = greedy_lot_sizing(tight6)
        assert mip_sol.total_cost <= greedy_sol.total_cost + 1e-4

    def test_beats_forward_greedy(self, var8):
        mip_sol = mip_clsp(var8)
        fg_sol = forward_greedy(var8)
        assert mip_sol.total_cost <= fg_sol.total_cost + 1e-4

    def test_known_optimal_single(self, single_period):
        """Single period: must produce exactly demand."""
        sol = mip_clsp(single_period)
        assert abs(sol.production[0] - 50.0) < 1e-4
        expected_cost = 200.0 + 3.0 * 50.0  # fixed + variable, no holding
        assert abs(sol.total_cost - expected_cost) < 1e-4

    def test_known_two_period(self, two_period_known):
        """Two periods: cheaper to produce all in period 0 (one setup)."""
        sol = mip_clsp(two_period_known)
        # Optimal: produce 50 in period 0 (one setup = 100, holding 20*1 = 20)
        # vs produce 30+20 separately (two setups = 200, no holding)
        # So optimal = 100 + 20 = 120 < 200
        assert sol.total_cost < 200.0 - 1e-4
        assert abs(sol.total_cost - 120.0) < 1e-4

    def test_loose_matches_uncapacitated(self, loose4):
        """Loose capacity should give same result as uncapacitated problem."""
        sol = mip_clsp(loose4)
        # With loose capacity and K=54, h=1, zero variable costs:
        # Optimal is to consolidate orders when holding < setup
        assert sol.total_cost < loose4.compute_cost(loose4.demands) + 1e-4


# ── Edge case tests ──────────────────────────────────────────────────────────


class TestMIPEdgeCases:
    def test_zero_demand_period(self):
        """Period with zero demand should not require production."""
        inst = CapLotSizingInstance(
            T=3,
            demands=np.array([20.0, 0.0, 30.0]),
            capacities=np.array([60.0, 60.0, 60.0]),
            fixed_costs=np.full(3, 100.0),
            variable_costs=np.full(3, 0.0),
            holding_costs=np.full(3, 1.0),
        )
        sol = mip_clsp(inst)
        assert inst.is_feasible(sol.production)

    def test_all_zero_demand(self):
        """All zero demand: optimal cost should be zero."""
        inst = CapLotSizingInstance(
            T=3,
            demands=np.zeros(3),
            capacities=np.full(3, 100.0),
            fixed_costs=np.full(3, 100.0),
            variable_costs=np.full(3, 1.0),
            holding_costs=np.full(3, 1.0),
        )
        sol = mip_clsp(inst)
        assert abs(sol.total_cost) < 1e-4
        assert np.allclose(sol.production, 0.0, atol=1e-4)

    def test_production_periods_populated(self, tight6):
        sol = mip_clsp(tight6)
        for t in sol.production_periods:
            assert sol.production[t] > 1e-6

    def test_random_instance(self):
        inst = CapLotSizingInstance.random(T=8, seed=123)
        sol = mip_clsp(inst)
        assert inst.is_feasible(sol.production)
        assert sol.total_cost < float("inf")

    def test_repr(self, tight6):
        sol = mip_clsp(tight6)
        r = repr(sol)
        assert "CapLotSizingSolution" in r
