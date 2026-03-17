"""
Test suite for 1D Bin Packing Problem.

Tests cover:
- Instance creation and validation
- First Fit, First Fit Decreasing, Best Fit Decreasing
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


_inst_mod = _load_module("bpp_instance_test", os.path.join(_base_dir, "instance.py"))
_ff_mod = _load_module("bpp_ff_test", os.path.join(_base_dir, "heuristics", "first_fit.py"))
_ga_mod = _load_module("bpp_ga_test", os.path.join(_base_dir, "metaheuristics", "genetic_algorithm.py"))

BinPackingInstance = _inst_mod.BinPackingInstance
BinPackingSolution = _inst_mod.BinPackingSolution
validate_solution = _inst_mod.validate_solution
easy_bpp_6 = _inst_mod.easy_bpp_6
tight_bpp_8 = _inst_mod.tight_bpp_8
uniform_bpp_10 = _inst_mod.uniform_bpp_10

first_fit = _ff_mod.first_fit
first_fit_decreasing = _ff_mod.first_fit_decreasing
best_fit_decreasing = _ff_mod.best_fit_decreasing
genetic_algorithm = _ga_mod.genetic_algorithm


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def inst6():
    return easy_bpp_6()


@pytest.fixture
def inst8():
    return tight_bpp_8()


@pytest.fixture
def inst10():
    return uniform_bpp_10()


@pytest.fixture
def random_inst():
    return BinPackingInstance.random(20, capacity=100.0, seed=42)


# ── Instance tests ───────────────────────────────────────────────────────────


class TestBinPackingInstance:
    def test_create_basic(self, inst6):
        assert inst6.n == 6
        assert inst6.capacity == 10.0
        assert inst6.sizes.shape == (6,)

    def test_random_instance(self):
        inst = BinPackingInstance.random(15, seed=123)
        assert inst.n == 15
        assert inst.capacity == 100.0
        assert np.all(inst.sizes <= inst.capacity)

    def test_lower_bound_l1(self, inst6):
        # sum=21, C=10 => L1 = ceil(21/10) = 3
        assert inst6.lower_bound_l1() == 3

    def test_lower_bound_l2(self, inst8):
        lb = inst8.lower_bound_l2()
        assert lb >= inst8.lower_bound_l1()

    def test_uniform_lower_bound(self, inst10):
        # 10 items of size 3, C=7 => sum=30, L1=ceil(30/7)=5
        assert inst10.lower_bound_l1() == 5

    def test_invalid_size_exceeds_capacity(self):
        with pytest.raises(ValueError):
            BinPackingInstance(
                n=2, sizes=np.array([5.0, 15.0]), capacity=10.0)

    def test_zero_capacity(self):
        with pytest.raises(ValueError):
            BinPackingInstance(
                n=1, sizes=np.array([1.0]), capacity=0.0)

    def test_negative_size(self):
        with pytest.raises(ValueError):
            BinPackingInstance(
                n=1, sizes=np.array([-1.0]), capacity=10.0)


class TestValidation:
    def test_valid_solution(self, inst6):
        sol = first_fit_decreasing(inst6)
        valid, errors = validate_solution(inst6, sol)
        assert valid, errors

    def test_missing_item(self, inst6):
        sol = BinPackingSolution(bins=[[0, 1], [2, 3]], num_bins=2)
        valid, errors = validate_solution(inst6, sol)
        assert not valid

    def test_duplicate_item(self, inst6):
        sol = BinPackingSolution(
            bins=[[0, 1, 2, 3, 4, 5, 0]], num_bins=1)
        valid, errors = validate_solution(inst6, sol)
        assert not valid


# ── First Fit heuristic tests ────────────────────────────────────────────────


class TestFirstFit:
    def test_feasible_easy(self, inst6):
        sol = first_fit(inst6)
        valid, errors = validate_solution(inst6, sol)
        assert valid, errors

    def test_feasible_tight(self, inst8):
        sol = first_fit(inst8)
        valid, errors = validate_solution(inst8, sol)
        assert valid, errors

    def test_at_least_lower_bound(self, inst6):
        sol = first_fit(inst6)
        assert sol.num_bins >= inst6.lower_bound_l1()


class TestFirstFitDecreasing:
    def test_feasible_easy(self, inst6):
        sol = first_fit_decreasing(inst6)
        valid, errors = validate_solution(inst6, sol)
        assert valid, errors

    def test_optimal_easy(self, inst6):
        sol = first_fit_decreasing(inst6)
        assert sol.num_bins == 3  # Optimal for this instance

    def test_optimal_uniform(self, inst10):
        sol = first_fit_decreasing(inst10)
        assert sol.num_bins == 5  # Optimal: 2 items per bin

    def test_tight_instance(self, inst8):
        sol = first_fit_decreasing(inst8)
        valid, errors = validate_solution(inst8, sol)
        assert valid, errors
        assert sol.num_bins == 5  # Optimal for tight8

    def test_ffd_at_most_11_9_opt(self, random_inst):
        sol = first_fit_decreasing(random_inst)
        lb = random_inst.lower_bound_l1()
        # FFD <= 11/9 * OPT + 1, so FFD <= 11/9 * LB + 1 (approximately)
        assert sol.num_bins <= int(np.ceil(11 / 9 * lb)) + 2


class TestBestFitDecreasing:
    def test_feasible(self, inst6):
        sol = best_fit_decreasing(inst6)
        valid, errors = validate_solution(inst6, sol)
        assert valid, errors

    def test_at_least_as_good_as_ff(self, random_inst):
        ff_sol = first_fit(random_inst)
        bfd_sol = best_fit_decreasing(random_inst)
        # BFD is generally better than FF (not guaranteed per-instance)
        assert bfd_sol.num_bins <= ff_sol.num_bins + 2

    def test_optimal_uniform(self, inst10):
        sol = best_fit_decreasing(inst10)
        assert sol.num_bins == 5


# ── Genetic Algorithm tests ─────────────────────────────────────────────────


class TestGeneticAlgorithm:
    def test_feasible(self, inst6):
        sol = genetic_algorithm(inst6, pop_size=20, generations=50, seed=42)
        valid, errors = validate_solution(inst6, sol)
        assert valid, errors

    def test_deterministic_with_seed(self, inst8):
        sol1 = genetic_algorithm(inst8, pop_size=20, generations=50, seed=42)
        sol2 = genetic_algorithm(inst8, pop_size=20, generations=50, seed=42)
        assert sol1.num_bins == sol2.num_bins

    def test_competitive_with_ffd(self, random_inst):
        ffd_sol = first_fit_decreasing(random_inst)
        ga_sol = genetic_algorithm(
            random_inst, pop_size=30, generations=100, seed=42)
        # GA should be within 1 bin of FFD (usually equal or better)
        assert ga_sol.num_bins <= ffd_sol.num_bins + 1

    def test_finds_optimal_uniform(self, inst10):
        sol = genetic_algorithm(inst10, pop_size=20, generations=100, seed=42)
        assert sol.num_bins == 5


# ── Cross-method comparison ─────────────────────────────────────────────────


class TestCrossMethodComparison:
    def test_all_methods_valid(self, inst8):
        methods = [
            ("ff", first_fit),
            ("ffd", first_fit_decreasing),
            ("bfd", best_fit_decreasing),
            ("ga", lambda i: genetic_algorithm(i, pop_size=20, generations=50, seed=42)),
        ]
        for name, method in methods:
            sol = method(inst8)
            valid, errors = validate_solution(inst8, sol)
            assert valid, f"{name}: {errors}"

    def test_ffd_at_least_as_good_as_ff(self, random_inst):
        ff_sol = first_fit(random_inst)
        ffd_sol = first_fit_decreasing(random_inst)
        assert ffd_sol.num_bins <= ff_sol.num_bins

    def test_all_above_lower_bound(self, random_inst):
        lb = random_inst.lower_bound_l1()
        for method in [first_fit, first_fit_decreasing, best_fit_decreasing]:
            sol = method(random_inst)
            assert sol.num_bins >= lb
