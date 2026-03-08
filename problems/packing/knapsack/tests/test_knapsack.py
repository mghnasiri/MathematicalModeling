"""
Test suite for 0-1 Knapsack Problem.

Tests cover:
- Instance creation and validation
- Dynamic programming (exact)
- Branch and bound (exact)
- Greedy heuristics
- Genetic algorithm
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


_inst_mod = _load_module("kp_instance_test", os.path.join(_base_dir, "instance.py"))
_dp_mod = _load_module("kp_dp_test", os.path.join(_base_dir, "exact", "dynamic_programming.py"))
_bb_mod = _load_module("kp_bb_test", os.path.join(_base_dir, "exact", "branch_and_bound.py"))
_gr_mod = _load_module("kp_gr_test", os.path.join(_base_dir, "heuristics", "greedy.py"))
_ga_mod = _load_module("kp_ga_test", os.path.join(_base_dir, "metaheuristics", "genetic_algorithm.py"))

KnapsackInstance = _inst_mod.KnapsackInstance
KnapsackSolution = _inst_mod.KnapsackSolution
validate_solution = _inst_mod.validate_solution
small_knapsack_4 = _inst_mod.small_knapsack_4
medium_knapsack_8 = _inst_mod.medium_knapsack_8
strongly_correlated_10 = _inst_mod.strongly_correlated_10

dynamic_programming = _dp_mod.dynamic_programming
branch_and_bound = _bb_mod.branch_and_bound
greedy_value_density = _gr_mod.greedy_value_density
greedy_max_value = _gr_mod.greedy_max_value
greedy_combined = _gr_mod.greedy_combined
genetic_algorithm = _ga_mod.genetic_algorithm


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def inst4():
    return small_knapsack_4()


@pytest.fixture
def inst8():
    return medium_knapsack_8()


@pytest.fixture
def inst_corr():
    return strongly_correlated_10()


@pytest.fixture
def random_inst():
    return KnapsackInstance.random(20, seed=42)


# ── Instance tests ───────────────────────────────────────────────────────────


class TestKnapsackInstance:
    def test_create_basic(self, inst4):
        assert inst4.n == 4
        assert inst4.capacity == 7.0
        assert inst4.weights.shape == (4,)
        assert inst4.values.shape == (4,)

    def test_random_instance(self):
        inst = KnapsackInstance.random(15, seed=123)
        assert inst.n == 15
        assert inst.capacity > 0

    def test_total_weight(self, inst4):
        assert abs(inst4.total_weight([0, 2, 3]) - 7.0) < 1e-10

    def test_total_value(self, inst4):
        assert abs(inst4.total_value([0, 2, 3]) - 35.0) < 1e-10

    def test_is_feasible(self, inst4):
        assert inst4.is_feasible([0, 2, 3])
        assert not inst4.is_feasible([0, 1, 2, 3])

    def test_invalid_weights_shape(self):
        with pytest.raises(ValueError):
            KnapsackInstance(
                n=3, weights=np.array([1.0, 2.0]),
                values=np.array([1.0, 2.0, 3.0]),
                capacity=10.0)

    def test_negative_capacity(self):
        with pytest.raises(ValueError):
            KnapsackInstance(
                n=2, weights=np.array([1.0, 2.0]),
                values=np.array([1.0, 2.0]),
                capacity=-1.0)

    def test_empty_knapsack(self):
        inst = KnapsackInstance(
            n=0, weights=np.array([]),
            values=np.array([]), capacity=10.0)
        assert inst.n == 0
        assert inst.total_value([]) == 0.0


class TestValidation:
    def test_valid_solution(self, inst4):
        sol = KnapsackSolution(items=[0, 2, 3], value=35.0, weight=7.0)
        valid, errors = validate_solution(inst4, sol)
        assert valid, errors

    def test_over_capacity(self, inst4):
        sol = KnapsackSolution(items=[0, 1, 2, 3], value=50.0, weight=10.0)
        valid, errors = validate_solution(inst4, sol)
        assert not valid

    def test_duplicate_items(self, inst4):
        sol = KnapsackSolution(items=[0, 0], value=20.0, weight=4.0)
        valid, errors = validate_solution(inst4, sol)
        assert not valid

    def test_invalid_index(self, inst4):
        sol = KnapsackSolution(items=[5], value=0.0, weight=0.0)
        valid, errors = validate_solution(inst4, sol)
        assert not valid


# ── Dynamic Programming tests ────────────────────────────────────────────────


class TestDynamicProgramming:
    def test_optimal_small4(self, inst4):
        sol = dynamic_programming(inst4)
        assert abs(sol.value - 35.0) < 1e-6
        assert inst4.is_feasible(sol.items)

    def test_optimal_medium8(self, inst8):
        sol = dynamic_programming(inst8)
        assert abs(sol.value - 300.0) < 1e-6
        assert inst8.is_feasible(sol.items)

    def test_correlated_instance(self, inst_corr):
        sol = dynamic_programming(inst_corr)
        valid, errors = validate_solution(inst_corr, sol)
        assert valid, errors

    def test_zero_capacity(self):
        inst = KnapsackInstance(
            n=3, weights=np.array([1.0, 2.0, 3.0]),
            values=np.array([10.0, 20.0, 30.0]), capacity=0.0)
        sol = dynamic_programming(inst)
        assert sol.value == 0.0
        assert sol.items == []

    def test_all_items_fit(self):
        inst = KnapsackInstance(
            n=3, weights=np.array([1.0, 2.0, 3.0]),
            values=np.array([10.0, 20.0, 30.0]), capacity=100.0)
        sol = dynamic_programming(inst)
        assert abs(sol.value - 60.0) < 1e-6
        assert sorted(sol.items) == [0, 1, 2]

    def test_single_item_fits(self):
        inst = KnapsackInstance(
            n=1, weights=np.array([5.0]),
            values=np.array([10.0]), capacity=5.0)
        sol = dynamic_programming(inst)
        assert abs(sol.value - 10.0) < 1e-6

    def test_single_item_no_fit(self):
        inst = KnapsackInstance(
            n=1, weights=np.array([5.0]),
            values=np.array([10.0]), capacity=4.0)
        sol = dynamic_programming(inst)
        assert sol.value == 0.0


# ── Branch and Bound tests ───────────────────────────────────────────────────


class TestBranchAndBound:
    def test_optimal_small4(self, inst4):
        sol = branch_and_bound(inst4)
        assert abs(sol.value - 35.0) < 1e-6
        assert inst4.is_feasible(sol.items)

    def test_optimal_medium8(self, inst8):
        sol = branch_and_bound(inst8)
        assert abs(sol.value - 300.0) < 1e-6
        assert inst8.is_feasible(sol.items)

    def test_agrees_with_dp(self, random_inst):
        dp_sol = dynamic_programming(random_inst)
        bb_sol = branch_and_bound(random_inst)
        assert abs(dp_sol.value - bb_sol.value) < 1e-6

    def test_correlated_instance(self, inst_corr):
        sol = branch_and_bound(inst_corr)
        dp_sol = dynamic_programming(inst_corr)
        assert abs(sol.value - dp_sol.value) < 1e-6

    def test_zero_capacity(self):
        inst = KnapsackInstance(
            n=3, weights=np.array([1.0, 2.0, 3.0]),
            values=np.array([10.0, 20.0, 30.0]), capacity=0.0)
        sol = branch_and_bound(inst)
        assert sol.value == 0.0


# ── Greedy heuristic tests ──────────────────────────────────────────────────


class TestGreedyHeuristics:
    def test_density_feasible(self, inst4):
        sol = greedy_value_density(inst4)
        assert inst4.is_feasible(sol.items)

    def test_max_value_feasible(self, inst4):
        sol = greedy_max_value(inst4)
        assert inst4.is_feasible(sol.items)

    def test_combined_at_least_as_good(self, inst8):
        sol_d = greedy_value_density(inst8)
        sol_m = greedy_max_value(inst8)
        sol_c = greedy_combined(inst8)
        assert sol_c.value >= sol_d.value - 1e-10
        assert sol_c.value >= sol_m.value - 1e-10

    def test_greedy_lower_bound(self, random_inst):
        sol_g = greedy_combined(random_inst)
        sol_dp = dynamic_programming(random_inst)
        # Greedy should be at least 1/2 of optimal
        assert sol_g.value >= sol_dp.value * 0.5 - 1e-6

    def test_greedy_not_always_optimal(self):
        # Classic counterexample: high-ratio small item vs large valuable item
        inst = KnapsackInstance(
            n=2, weights=np.array([1.0, 10.0]),
            values=np.array([6.0, 50.0]), capacity=10.0)
        sol = greedy_value_density(inst)
        dp_sol = dynamic_programming(inst)
        # Greedy by density picks item 0 (ratio=6), missing item 1 (ratio=5, value=50)
        assert dp_sol.value == 50.0


# ── Genetic Algorithm tests ─────────────────────────────────────────────────


class TestGeneticAlgorithm:
    def test_feasible_solution(self, inst4):
        sol = genetic_algorithm(inst4, pop_size=20, generations=50, seed=42)
        valid, errors = validate_solution(inst4, sol)
        assert valid, errors

    def test_finds_optimal_small(self, inst4):
        sol = genetic_algorithm(inst4, pop_size=30, generations=100, seed=42)
        assert abs(sol.value - 35.0) < 1e-6

    def test_deterministic_with_seed(self, inst8):
        sol1 = genetic_algorithm(inst8, pop_size=20, generations=50, seed=42)
        sol2 = genetic_algorithm(inst8, pop_size=20, generations=50, seed=42)
        assert abs(sol1.value - sol2.value) < 1e-10

    def test_random_instance(self, random_inst):
        sol = genetic_algorithm(random_inst, pop_size=30, generations=100, seed=42)
        valid, errors = validate_solution(random_inst, sol)
        assert valid, errors

    def test_competitive_with_greedy(self, random_inst):
        gr_sol = greedy_combined(random_inst)
        ga_sol = genetic_algorithm(
            random_inst, pop_size=40, generations=150, seed=42)
        assert ga_sol.value >= gr_sol.value * 0.9


# ── Cross-method comparison ─────────────────────────────────────────────────


class TestCrossMethodComparison:
    def test_all_methods_valid(self, inst8):
        methods = [
            ("dp", lambda i: dynamic_programming(i)),
            ("bb", lambda i: branch_and_bound(i)),
            ("greedy", lambda i: greedy_combined(i)),
            ("ga", lambda i: genetic_algorithm(i, pop_size=20, generations=50, seed=42)),
        ]
        for name, method in methods:
            sol = method(inst8)
            valid, errors = validate_solution(inst8, sol)
            assert valid, f"{name}: {errors}"

    def test_exact_methods_agree(self, random_inst):
        dp_sol = dynamic_programming(random_inst)
        bb_sol = branch_and_bound(random_inst)
        assert abs(dp_sol.value - bb_sol.value) < 1e-6

    def test_heuristics_bounded(self, random_inst):
        dp_sol = dynamic_programming(random_inst)
        gr_sol = greedy_combined(random_inst)
        ga_sol = genetic_algorithm(
            random_inst, pop_size=30, generations=100, seed=42)
        assert gr_sol.value <= dp_sol.value + 1e-6
        assert ga_sol.value <= dp_sol.value + 1e-6
