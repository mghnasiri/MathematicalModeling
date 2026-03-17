"""
Test suite for Capacitated Lot Sizing Problem.

Tests cover:
- Instance creation and validation
- Feasibility checking
- Lot-for-lot baseline
- Forward pass greedy heuristic
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


_inst_mod = _load_mod("cls_inst_test", os.path.join(_base_dir, "instance.py"))
_greedy_mod = _load_mod(
    "cls_greedy_test", os.path.join(_base_dir, "heuristics", "greedy_cls.py")
)

CapLotSizingInstance = _inst_mod.CapLotSizingInstance
CapLotSizingSolution = _inst_mod.CapLotSizingSolution
tight_capacity_6 = _inst_mod.tight_capacity_6
loose_capacity_4 = _inst_mod.loose_capacity_4
variable_costs_8 = _inst_mod.variable_costs_8

lot_for_lot = _greedy_mod.lot_for_lot
forward_greedy = _greedy_mod.forward_greedy


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


# ── Instance tests ───────────────────────────────────────────────────────────


class TestCapLotSizingInstance:
    def test_creation(self, tight6):
        assert tight6.T == 6
        assert len(tight6.demands) == 6
        assert np.allclose(tight6.capacities, 40.0)

    def test_invalid_shape(self):
        with pytest.raises(ValueError, match="demands shape"):
            CapLotSizingInstance(
                T=3,
                demands=np.array([10.0, 20.0]),
                capacities=np.array([50.0, 50.0, 50.0]),
                fixed_costs=np.array([100.0, 100.0, 100.0]),
                variable_costs=np.array([1.0, 1.0, 1.0]),
                holding_costs=np.array([1.0, 1.0, 1.0]),
            )

    def test_negative_demand(self):
        with pytest.raises(ValueError, match="non-negative"):
            CapLotSizingInstance(
                T=2,
                demands=np.array([10.0, -5.0]),
                capacities=np.array([50.0, 50.0]),
                fixed_costs=np.array([100.0, 100.0]),
                variable_costs=np.array([1.0, 1.0]),
                holding_costs=np.array([1.0, 1.0]),
            )

    def test_random_instance(self):
        inst = CapLotSizingInstance.random(T=5, seed=42)
        assert inst.T == 5
        assert len(inst.demands) == 5
        assert np.all(inst.capacities > 0)

    def test_feasibility_check(self, tight6):
        # Lot-for-lot should be feasible
        assert tight6.is_feasible(tight6.demands)
        # Zero production is infeasible (unmet demand)
        assert not tight6.is_feasible(np.zeros(6))

    def test_compute_cost(self, loose4):
        production = loose4.demands.copy()
        cost = loose4.compute_cost(production)
        # variable_costs are 0, so cost = 4 * 54 = 216 (fixed) + 0 (holding)
        assert abs(cost - 216.0) < 1e-6

    def test_solution_repr(self):
        sol = CapLotSizingSolution(
            production=np.array([50.0, 0.0, 30.0]),
            total_cost=250.0,
            production_periods=[0, 2],
        )
        r = repr(sol)
        assert "CapLotSizingSolution" in r
        assert "250.00" in r


# ── Lot-for-lot tests ───────────────────────────────────────────────────────


class TestLotForLot:
    def test_produces_exact_demand(self, tight6):
        sol = lot_for_lot(tight6)
        assert np.allclose(sol.production, tight6.demands)

    def test_feasibility(self, tight6):
        sol = lot_for_lot(tight6)
        assert tight6.is_feasible(sol.production)

    def test_cost_matches_recompute(self, tight6):
        sol = lot_for_lot(tight6)
        recomputed = tight6.compute_cost(sol.production)
        assert abs(sol.total_cost - recomputed) < 1e-6

    def test_infeasible_capacity(self):
        inst = CapLotSizingInstance(
            T=2,
            demands=np.array([100.0, 50.0]),
            capacities=np.array([50.0, 50.0]),
            fixed_costs=np.array([100.0, 100.0]),
            variable_costs=np.array([1.0, 1.0]),
            holding_costs=np.array([1.0, 1.0]),
        )
        with pytest.raises(ValueError, match="exceeds capacity"):
            lot_for_lot(inst)


# ── Forward greedy tests ────────────────────────────────────────────────────


class TestForwardGreedy:
    def test_feasibility(self, tight6):
        sol = forward_greedy(tight6)
        assert tight6.is_feasible(sol.production)

    def test_capacity_respected(self, tight6):
        sol = forward_greedy(tight6)
        assert np.all(sol.production <= tight6.capacities + 1e-10)

    def test_demand_satisfied(self, tight6):
        sol = forward_greedy(tight6)
        cumulative_prod = np.cumsum(sol.production)
        cumulative_dem = np.cumsum(tight6.demands)
        assert np.all(cumulative_prod >= cumulative_dem - 1e-10)

    def test_greedy_beats_lfl_loose(self, loose4):
        """With loose capacity, greedy should do at least as well as lot-for-lot."""
        sol_greedy = forward_greedy(loose4)
        sol_lfl = lot_for_lot(loose4)
        assert sol_greedy.total_cost <= sol_lfl.total_cost + 1e-10

    def test_cost_matches_recompute(self, var8):
        sol = forward_greedy(var8)
        recomputed = var8.compute_cost(sol.production)
        assert abs(sol.total_cost - recomputed) < 1e-6

    def test_variable_costs_8(self, var8):
        sol = forward_greedy(var8)
        assert sol.total_cost < float("inf")
        assert var8.is_feasible(sol.production)
