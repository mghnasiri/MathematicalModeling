"""
Test suite for Capacitated Lot Sizing Problem (CLSP).

Tests cover:
- Instance creation and validation
- Greedy heuristic and lot-for-lot
- Feasibility checks
- Edge cases
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


_inst_mod = _load_mod("cls_instance_test", os.path.join(_base_dir, "instance.py"))
_greedy_mod = _load_mod(
    "cls_greedy_test", os.path.join(_base_dir, "heuristics", "greedy_clsp.py")
)

CapLotSizingInstance = _inst_mod.CapLotSizingInstance
CapLotSizingSolution = _inst_mod.CapLotSizingSolution
tight_capacity_6 = _inst_mod.tight_capacity_6
loose_capacity_4 = _inst_mod.loose_capacity_4
variable_costs_8 = _inst_mod.variable_costs_8

greedy_lot_sizing = _greedy_mod.greedy_lot_sizing
lot_for_lot = _greedy_mod.lot_for_lot


# -- Fixtures ------------------------------------------------------------------


@pytest.fixture
def inst6():
    return tight_capacity_6()


@pytest.fixture
def inst4():
    return loose_capacity_4()


@pytest.fixture
def inst8():
    return variable_costs_8()


# -- Instance tests ------------------------------------------------------------


class TestCapLotSizingInstance:
    def test_create_basic(self, inst6):
        assert inst6.T == 6
        assert inst6.demands.shape == (6,)
        assert inst6.capacities.shape == (6,)

    def test_random_instance(self):
        inst = CapLotSizingInstance.random(8, seed=42)
        assert inst.T == 8
        assert np.all(inst.demands >= 0)

    def test_invalid_shape(self):
        with pytest.raises(ValueError):
            CapLotSizingInstance(
                T=3,
                demands=np.array([10.0, 20.0]),
                capacities=np.full(3, 50.0),
                fixed_costs=np.full(3, 100.0),
                variable_costs=np.full(3, 1.0),
                holding_costs=np.full(3, 1.0),
            )

    def test_is_feasible(self, inst6):
        prod = inst6.demands.copy()
        assert inst6.is_feasible(prod)

    def test_infeasible_over_capacity(self, inst6):
        prod = inst6.capacities + 10.0
        assert not inst6.is_feasible(prod)


# -- Greedy tests --------------------------------------------------------------


class TestGreedyLotSizing:
    def test_feasible_tight(self, inst6):
        sol = greedy_lot_sizing(inst6)
        assert inst6.is_feasible(sol.production)

    def test_feasible_loose(self, inst4):
        sol = greedy_lot_sizing(inst4)
        assert inst4.is_feasible(sol.production)

    def test_cost_matches(self, inst6):
        sol = greedy_lot_sizing(inst6)
        verified = inst6.compute_cost(sol.production)
        assert abs(sol.total_cost - verified) < 1e-6

    def test_beats_lot_for_lot_loose(self, inst4):
        greedy_sol = greedy_lot_sizing(inst4)
        lfl_sol = lot_for_lot(inst4)
        assert greedy_sol.total_cost <= lfl_sol.total_cost + 1e-10

    def test_variable_costs(self, inst8):
        sol = greedy_lot_sizing(inst8)
        assert inst8.is_feasible(sol.production)

    def test_random_instance(self):
        inst = CapLotSizingInstance.random(10, capacity_factor=2.0, seed=77)
        sol = greedy_lot_sizing(inst)
        assert inst.is_feasible(sol.production)


# -- Lot-for-lot tests ---------------------------------------------------------


class TestLotForLot:
    def test_feasible(self, inst4):
        sol = lot_for_lot(inst4)
        assert inst4.is_feasible(sol.production)

    def test_no_inventory(self, inst4):
        sol = lot_for_lot(inst4)
        # Lot-for-lot produces exactly demand, so no inventory held
        inventory = 0.0
        for t in range(inst4.T):
            inventory += sol.production[t] - inst4.demands[t]
            assert abs(inventory) < 1e-10

    def test_repr(self, inst6):
        sol = greedy_lot_sizing(inst6)
        r = repr(sol)
        assert "CapLotSizingSolution" in r
