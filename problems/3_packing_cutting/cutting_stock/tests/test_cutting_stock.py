"""
Test suite for 1D Cutting Stock Problem.

Tests cover:
- Instance creation and validation
- Greedy largest-first heuristic
- FFD-based heuristic
"""

from __future__ import annotations

import os
import sys
import pytest
import numpy as np
import importlib.util

# ── Module loading ───────────────────────────────────────────────────────────

_base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_module(name: str, filepath: str):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_inst_mod = _load_module(
    "csp_instance_test", os.path.join(_base_dir, "instance.py"))
_gr_mod = _load_module(
    "csp_greedy_test", os.path.join(_base_dir, "heuristics", "greedy_csp.py"))

CuttingStockInstance = _inst_mod.CuttingStockInstance
CuttingStockSolution = _inst_mod.CuttingStockSolution
validate_solution = _inst_mod.validate_solution
simple_csp_3 = _inst_mod.simple_csp_3
classic_csp_4 = _inst_mod.classic_csp_4

greedy_largest_first = _gr_mod.greedy_largest_first
ffd_based = _gr_mod.ffd_based


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def inst3():
    return simple_csp_3()


@pytest.fixture
def inst4():
    return classic_csp_4()


@pytest.fixture
def random_inst():
    return CuttingStockInstance.random(5, seed=42)


# ── Instance tests ───────────────────────────────────────────────────────────


class TestCuttingStockInstance:
    def test_create_basic(self, inst3):
        assert inst3.m == 3
        assert inst3.stock_length == 100.0
        assert inst3.lengths.shape == (3,)
        assert inst3.demands.shape == (3,)

    def test_random_instance(self):
        inst = CuttingStockInstance.random(5, seed=123)
        assert inst.m == 5
        assert np.all(inst.lengths <= inst.stock_length)

    def test_lower_bound(self, inst3):
        lb = inst3.lower_bound()
        assert lb >= 1

    def test_total_items(self, inst3):
        assert inst3.total_items() == 7  # 3 + 2 + 2

    def test_invalid_length_exceeds_stock(self):
        with pytest.raises(ValueError):
            CuttingStockInstance(
                m=2, stock_length=10.0,
                lengths=np.array([5.0, 15.0]),
                demands=np.array([1, 1]))

    def test_zero_stock_length(self):
        with pytest.raises(ValueError):
            CuttingStockInstance(
                m=1, stock_length=0.0,
                lengths=np.array([1.0]),
                demands=np.array([1]))

    def test_negative_demand(self):
        with pytest.raises(ValueError):
            CuttingStockInstance(
                m=1, stock_length=10.0,
                lengths=np.array([5.0]),
                demands=np.array([-1]))


class TestValidation:
    def test_valid_solution(self, inst3):
        sol = greedy_largest_first(inst3)
        valid, errors = validate_solution(inst3, sol)
        assert valid, errors

    def test_insufficient_demand(self, inst3):
        # Pattern that doesn't satisfy all demands
        sol = CuttingStockSolution(
            patterns=[(np.array([1, 0, 0]), 1)],
            num_rolls=1,
        )
        valid, errors = validate_solution(inst3, sol)
        assert not valid

    def test_pattern_exceeds_stock(self, inst3):
        sol = CuttingStockSolution(
            patterns=[(np.array([3, 0, 0]), 1)],  # 3*45=135 > 100
            num_rolls=1,
        )
        valid, errors = validate_solution(inst3, sol)
        assert not valid


# ── Greedy heuristic tests ──────────────────────────────────────────────────


class TestGreedyLargestFirst:
    def test_feasible_simple(self, inst3):
        sol = greedy_largest_first(inst3)
        valid, errors = validate_solution(inst3, sol)
        assert valid, errors

    def test_feasible_classic(self, inst4):
        sol = greedy_largest_first(inst4)
        valid, errors = validate_solution(inst4, sol)
        assert valid, errors

    def test_at_least_lower_bound(self, inst3):
        sol = greedy_largest_first(inst3)
        assert sol.num_rolls >= inst3.lower_bound()

    def test_random_instance(self, random_inst):
        sol = greedy_largest_first(random_inst)
        valid, errors = validate_solution(random_inst, sol)
        assert valid, errors


class TestFFDBased:
    def test_feasible_simple(self, inst3):
        sol = ffd_based(inst3)
        valid, errors = validate_solution(inst3, sol)
        assert valid, errors

    def test_feasible_classic(self, inst4):
        sol = ffd_based(inst4)
        valid, errors = validate_solution(inst4, sol)
        assert valid, errors

    def test_at_least_lower_bound(self, inst3):
        sol = ffd_based(inst3)
        assert sol.num_rolls >= inst3.lower_bound()

    def test_random_instance(self, random_inst):
        sol = ffd_based(random_inst)
        valid, errors = validate_solution(random_inst, sol)
        assert valid, errors

    def test_aggregates_patterns(self, inst4):
        sol = ffd_based(inst4)
        # After aggregation, some patterns may have frequency > 1
        total = sum(freq for _, freq in sol.patterns)
        assert total == sol.num_rolls


# ── Cross-method comparison ─────────────────────────────────────────────────


class TestCrossMethodComparison:
    def test_both_valid(self, inst4):
        gl = greedy_largest_first(inst4)
        ffd = ffd_based(inst4)
        valid_gl, _ = validate_solution(inst4, gl)
        valid_ffd, _ = validate_solution(inst4, ffd)
        assert valid_gl
        assert valid_ffd

    def test_reasonable_gap(self, random_inst):
        gl = greedy_largest_first(random_inst)
        ffd = ffd_based(random_inst)
        lb = random_inst.lower_bound()
        # Both should be within 50% of lower bound
        assert gl.num_rolls <= lb * 1.5 + 2
        assert ffd.num_rolls <= lb * 1.5 + 2
