"""
Test suite for Economic Order Quantity (EOQ) problem.

Tests cover:
- Instance creation and validation
- Classic EOQ formula
- EOQ with backorders
- EOQ with quantity discounts
- Sensitivity analysis (cost robustness near Q*)
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


_inst_mod = _load_mod("eoq_inst_test", os.path.join(_base_dir, "instance.py"))
_formula_mod = _load_mod(
    "eoq_formula_test", os.path.join(_base_dir, "exact", "eoq_formula.py")
)

EOQInstance = _inst_mod.EOQInstance
EOQSolution = _inst_mod.EOQSolution
textbook_eoq = _inst_mod.textbook_eoq
backorder_eoq = _inst_mod.backorder_eoq
discount_eoq = _inst_mod.discount_eoq

classic_eoq = _formula_mod.classic_eoq
eoq_with_backorders = _formula_mod.eoq_with_backorders
eoq_with_discounts = _formula_mod.eoq_with_discounts


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def textbook():
    return textbook_eoq()


@pytest.fixture
def backorder():
    return backorder_eoq()


@pytest.fixture
def discount():
    return discount_eoq()


# ── Instance tests ───────────────────────────────────────────────────────────


class TestEOQInstance:
    def test_creation(self, textbook):
        assert textbook.demand_rate == 1000.0
        assert textbook.ordering_cost == 50.0
        assert textbook.holding_cost == 2.0

    def test_invalid_demand(self):
        with pytest.raises(ValueError, match="demand_rate"):
            EOQInstance(demand_rate=-100, ordering_cost=50, holding_cost=2)

    def test_invalid_holding_cost(self):
        with pytest.raises(ValueError, match="holding_cost"):
            EOQInstance(demand_rate=100, ordering_cost=50, holding_cost=-1)

    def test_random_instance(self):
        inst = EOQInstance.random(seed=42)
        assert inst.demand_rate > 0
        assert inst.ordering_cost > 0
        assert inst.holding_cost > 0

    def test_total_cost(self, textbook):
        Q = 200.0
        cost = textbook.total_cost(Q)
        expected = (1000.0 / 200.0) * 50.0 + (200.0 / 2.0) * 2.0
        assert abs(cost - expected) < 1e-6


# ── Classic EOQ tests ────────────────────────────────────────────────────────


class TestClassicEOQ:
    def test_textbook_formula(self, textbook):
        sol = classic_eoq(textbook)
        expected_Q = np.sqrt(2 * 1000 * 50 / 2.0)
        assert abs(sol.order_quantity - expected_Q) < 1e-6

    def test_total_cost_at_optimum(self, textbook):
        sol = classic_eoq(textbook)
        expected_cost = np.sqrt(2 * 1000 * 50 * 2.0)
        assert abs(sol.total_cost - expected_cost) < 1e-6

    def test_num_orders(self, textbook):
        sol = classic_eoq(textbook)
        expected_orders = 1000.0 / sol.order_quantity
        assert abs(sol.num_orders - expected_orders) < 1e-6

    def test_cycle_time(self, textbook):
        sol = classic_eoq(textbook)
        expected_cycle = sol.order_quantity / 1000.0
        assert abs(sol.cycle_time - expected_cycle) < 1e-6

    def test_cost_sensitivity(self, textbook):
        """Total cost is relatively flat near Q*; deviations of 10% raise cost < 1%."""
        sol = classic_eoq(textbook)
        Q_star = sol.order_quantity
        cost_star = sol.total_cost
        cost_high = textbook.total_cost(Q_star * 1.1)
        cost_low = textbook.total_cost(Q_star * 0.9)
        assert cost_high < cost_star * 1.01
        assert cost_low < cost_star * 1.01

    def test_solution_repr(self, textbook):
        sol = classic_eoq(textbook)
        r = repr(sol)
        assert "EOQSolution" in r
        assert "Q*=" in r


# ── Backorder EOQ tests ──────────────────────────────────────────────────────


class TestEOQBackorders:
    def test_backorder_q_larger(self, backorder):
        """With backorders, Q* is larger than basic EOQ."""
        sol_bo = eoq_with_backorders(backorder)
        basic_Q = np.sqrt(2 * backorder.demand_rate * backorder.ordering_cost
                          / backorder.holding_cost)
        assert sol_bo.order_quantity > basic_Q

    def test_backorder_cost_lower(self, backorder):
        """With backorders, total cost is lower than basic EOQ."""
        sol_bo = eoq_with_backorders(backorder)
        basic_cost = np.sqrt(2 * backorder.demand_rate * backorder.ordering_cost
                             * backorder.holding_cost)
        assert sol_bo.total_cost < basic_cost

    def test_backorder_formula(self, backorder):
        D, K, h, b = 1200, 100, 5, 25
        expected_Q = np.sqrt(2 * D * K / h) * np.sqrt((h + b) / b)
        sol = eoq_with_backorders(backorder)
        assert abs(sol.order_quantity - expected_Q) < 1e-6

    def test_no_backorder_raises(self, textbook):
        with pytest.raises(ValueError, match="backorder_cost"):
            eoq_with_backorders(textbook)


# ── Discount EOQ tests ──────────────────────────────────────────────────────


class TestEOQDiscounts:
    def test_discount_selects_feasible(self, discount):
        sol = eoq_with_discounts(discount, holding_pct=0.2)
        assert sol.order_quantity > 0
        assert sol.total_cost > 0

    def test_discount_total_cost_includes_purchase(self, discount):
        sol = eoq_with_discounts(discount, holding_pct=0.2)
        # Purchase cost alone is at least D * min_price
        min_purchase = discount.demand_rate * min(discount.discount_prices)
        assert sol.total_cost >= min_purchase

    def test_no_discount_data_raises(self, textbook):
        with pytest.raises(ValueError, match="discount_breaks"):
            eoq_with_discounts(textbook)
