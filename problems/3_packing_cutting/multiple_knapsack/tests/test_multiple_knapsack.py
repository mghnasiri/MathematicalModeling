"""
Test suite for Multiple Knapsack Problem.

Tests cover:
- Instance creation and validation
- Greedy value-density and best-fit heuristics
- MILP exact solver
- Solution validation (capacity, no duplicates)
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
    "mkp_instance_test", os.path.join(_base_dir, "instance.py")
)
_greedy_mod = _load_module(
    "mkp_greedy_test",
    os.path.join(_base_dir, "heuristics", "greedy_mk.py"),
)
_ilp_mod = _load_module(
    "mkp_ilp_test",
    os.path.join(_base_dir, "exact", "ilp_mk.py"),
)

MultipleKnapsackInstance = _inst_mod.MultipleKnapsackInstance
MultipleKnapsackSolution = _inst_mod.MultipleKnapsackSolution
validate_solution = _inst_mod.validate_solution
small_mkp_6_2 = _inst_mod.small_mkp_6_2
medium_mkp_8_3 = _inst_mod.medium_mkp_8_3

greedy_value_density = _greedy_mod.greedy_value_density
greedy_best_fit = _greedy_mod.greedy_best_fit
milp_solver = _ilp_mod.milp_solver


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def inst_small():
    return small_mkp_6_2()


@pytest.fixture
def inst_medium():
    return medium_mkp_8_3()


@pytest.fixture
def random_inst():
    return MultipleKnapsackInstance.random(12, m=3, seed=42)


# ── Instance tests ───────────────────────────────────────────────────────────


class TestMultipleKnapsackInstance:
    def test_create_basic(self, inst_small):
        assert inst_small.n == 6
        assert inst_small.m == 2
        assert inst_small.weights.shape == (6,)
        assert inst_small.capacities.shape == (2,)

    def test_random_instance(self):
        inst = MultipleKnapsackInstance.random(10, m=2, seed=123)
        assert inst.n == 10
        assert inst.m == 2
        assert np.all(inst.weights > 0)

    def test_total_capacity(self, inst_small):
        assert inst_small.total_capacity() == 18.0

    def test_invalid_negative_weight(self):
        with pytest.raises(ValueError):
            MultipleKnapsackInstance(
                n=2, m=1,
                weights=np.array([-1.0, 5.0]),
                values=np.array([10.0, 20.0]),
                capacities=np.array([10.0]),
            )

    def test_invalid_zero_capacity(self):
        with pytest.raises(ValueError):
            MultipleKnapsackInstance(
                n=2, m=1,
                weights=np.array([3.0, 5.0]),
                values=np.array([10.0, 20.0]),
                capacities=np.array([0.0]),
            )


# ── Validation tests ────────────────────────────────────────────────────────


class TestValidation:
    def test_valid_solution(self, inst_small):
        sol = greedy_value_density(inst_small)
        valid, errors = validate_solution(inst_small, sol)
        assert valid, errors

    def test_duplicate_item(self, inst_small):
        sol = MultipleKnapsackSolution(
            assignments=[[0, 1], [1, 2]], value=50.0
        )
        valid, errors = validate_solution(inst_small, sol)
        assert not valid

    def test_wrong_number_of_knapsacks(self, inst_small):
        sol = MultipleKnapsackSolution(
            assignments=[[0, 1, 2]], value=45.0
        )
        valid, errors = validate_solution(inst_small, sol)
        assert not valid


# ── Greedy tests ─────────────────────────────────────────────────────────────


class TestGreedyValueDensity:
    def test_feasible_small(self, inst_small):
        sol = greedy_value_density(inst_small)
        valid, errors = validate_solution(inst_small, sol)
        assert valid, errors
        assert sol.value > 0

    def test_feasible_medium(self, inst_medium):
        sol = greedy_value_density(inst_medium)
        valid, errors = validate_solution(inst_medium, sol)
        assert valid, errors

    def test_value_positive(self, random_inst):
        sol = greedy_value_density(random_inst)
        valid, errors = validate_solution(random_inst, sol)
        assert valid, errors
        assert sol.value > 0


class TestGreedyBestFit:
    def test_feasible_small(self, inst_small):
        sol = greedy_best_fit(inst_small)
        valid, errors = validate_solution(inst_small, sol)
        assert valid, errors
        assert sol.value > 0

    def test_feasible_random(self, random_inst):
        sol = greedy_best_fit(random_inst)
        valid, errors = validate_solution(random_inst, sol)
        assert valid, errors


# ── MILP tests ───────────────────────────────────────────────────────────────


class TestMILPSolver:
    def test_optimal_small(self, inst_small):
        sol = milp_solver(inst_small, time_limit=30.0)
        assert sol is not None
        valid, errors = validate_solution(inst_small, sol)
        assert valid, errors
        # MILP should find at least as good as greedy
        greedy_sol = greedy_value_density(inst_small)
        assert sol.value >= greedy_sol.value - 1e-6

    def test_optimal_medium(self, inst_medium):
        sol = milp_solver(inst_medium, time_limit=30.0)
        assert sol is not None
        valid, errors = validate_solution(inst_medium, sol)
        assert valid, errors

    def test_milp_beats_greedy(self, inst_small):
        milp_sol = milp_solver(inst_small, time_limit=30.0)
        greedy_sol = greedy_value_density(inst_small)
        assert milp_sol is not None
        assert milp_sol.value >= greedy_sol.value - 1e-6
