"""
Test suite for Multi-dimensional Knapsack Problem.

Tests cover:
- Instance creation and validation
- Greedy aggregate efficiency heuristic
- LP relaxation rounding heuristic
- Solution validation (all dimensions, no duplicates)
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
    "mdkp_instance_test", os.path.join(_base_dir, "instance.py")
)
_greedy_mod = _load_module(
    "mdkp_greedy_test",
    os.path.join(_base_dir, "heuristics", "greedy_mdk.py"),
)

MultidimKnapsackInstance = _inst_mod.MultidimKnapsackInstance
MultidimKnapsackSolution = _inst_mod.MultidimKnapsackSolution
validate_solution = _inst_mod.validate_solution
small_mdkp_5_2 = _inst_mod.small_mdkp_5_2
medium_mdkp_8_3 = _inst_mod.medium_mdkp_8_3

greedy_aggregate_efficiency = _greedy_mod.greedy_aggregate_efficiency
lp_relaxation_rounding = _greedy_mod.lp_relaxation_rounding


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def inst_small():
    return small_mdkp_5_2()


@pytest.fixture
def inst_medium():
    return medium_mdkp_8_3()


@pytest.fixture
def random_inst():
    return MultidimKnapsackInstance.random(15, d=3, seed=42)


# ── Instance tests ───────────────────────────────────────────────────────────


class TestMultidimKnapsackInstance:
    def test_create_basic(self, inst_small):
        assert inst_small.n == 5
        assert inst_small.d == 2
        assert inst_small.weights.shape == (5, 2)
        assert inst_small.capacities.shape == (2,)

    def test_random_instance(self):
        inst = MultidimKnapsackInstance.random(10, d=4, seed=123)
        assert inst.n == 10
        assert inst.d == 4
        assert inst.weights.shape == (10, 4)

    def test_is_feasible_empty(self, inst_small):
        assert inst_small.is_feasible([])

    def test_is_feasible_single(self, inst_small):
        assert inst_small.is_feasible([0])

    def test_total_value(self, inst_small):
        assert abs(inst_small.total_value([0, 1]) - 30.0) < 1e-10

    def test_resource_usage(self, inst_small):
        usage = inst_small.resource_usage([0, 1])
        assert abs(usage[0] - 8.0) < 1e-10  # 3 + 5
        assert abs(usage[1] - 7.0) < 1e-10  # 4 + 3

    def test_invalid_negative_weight(self):
        with pytest.raises(ValueError):
            MultidimKnapsackInstance(
                n=2, d=1,
                weights=np.array([[-1.0], [5.0]]),
                values=np.array([10.0, 20.0]),
                capacities=np.array([10.0]),
            )

    def test_invalid_shape_mismatch(self):
        with pytest.raises(ValueError):
            MultidimKnapsackInstance(
                n=2, d=2,
                weights=np.array([[1.0], [2.0]]),  # wrong shape
                values=np.array([10.0, 20.0]),
                capacities=np.array([10.0, 10.0]),
            )


# ── Validation tests ────────────────────────────────────────────────────────


class TestValidation:
    def test_valid_solution(self, inst_small):
        sol = greedy_aggregate_efficiency(inst_small)
        valid, errors = validate_solution(inst_small, sol)
        assert valid, errors

    def test_duplicate_item(self, inst_small):
        sol = MultidimKnapsackSolution(items=[0, 0, 1], value=40.0)
        valid, errors = validate_solution(inst_small, sol)
        assert not valid

    def test_wrong_value(self, inst_small):
        sol = MultidimKnapsackSolution(items=[0], value=999.0)
        valid, errors = validate_solution(inst_small, sol)
        assert not valid


# ── Greedy aggregate efficiency tests ────────────────────────────────────────


class TestGreedyAggregateEfficiency:
    def test_feasible_small(self, inst_small):
        sol = greedy_aggregate_efficiency(inst_small)
        valid, errors = validate_solution(inst_small, sol)
        assert valid, errors
        assert sol.value > 0

    def test_feasible_medium(self, inst_medium):
        sol = greedy_aggregate_efficiency(inst_medium)
        valid, errors = validate_solution(inst_medium, sol)
        assert valid, errors

    def test_respects_all_dimensions(self, inst_small):
        sol = greedy_aggregate_efficiency(inst_small)
        usage = inst_small.resource_usage(sol.items)
        for k in range(inst_small.d):
            assert usage[k] <= inst_small.capacities[k] + 1e-10

    def test_random_feasible(self, random_inst):
        sol = greedy_aggregate_efficiency(random_inst)
        valid, errors = validate_solution(random_inst, sol)
        assert valid, errors
        assert sol.value > 0


# ── LP relaxation rounding tests ─────────────────────────────────────────────


class TestLPRelaxationRounding:
    def test_feasible_small(self, inst_small):
        sol = lp_relaxation_rounding(inst_small)
        valid, errors = validate_solution(inst_small, sol)
        assert valid, errors
        assert sol.value > 0

    def test_feasible_medium(self, inst_medium):
        sol = lp_relaxation_rounding(inst_medium)
        valid, errors = validate_solution(inst_medium, sol)
        assert valid, errors

    def test_respects_all_dimensions(self, inst_medium):
        sol = lp_relaxation_rounding(inst_medium)
        usage = inst_medium.resource_usage(sol.items)
        for k in range(inst_medium.d):
            assert usage[k] <= inst_medium.capacities[k] + 1e-10

    def test_random_feasible(self, random_inst):
        sol = lp_relaxation_rounding(random_inst)
        valid, errors = validate_solution(random_inst, sol)
        assert valid, errors


# ── Cross-method comparison ─────────────────────────────────────────────────


class TestCrossMethodComparison:
    def test_both_methods_valid(self, inst_medium):
        for algo in [greedy_aggregate_efficiency, lp_relaxation_rounding]:
            sol = algo(inst_medium)
            valid, errors = validate_solution(inst_medium, sol)
            assert valid, errors

    def test_both_positive_value(self, random_inst):
        for algo in [greedy_aggregate_efficiency, lp_relaxation_rounding]:
            sol = algo(random_inst)
            assert sol.value > 0
