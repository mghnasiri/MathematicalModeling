"""Tests for the Maximum Coverage problem."""
from __future__ import annotations

import sys
import os
import importlib.util
import math

import numpy as np
import pytest


def _load_mod(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_base = os.path.join(os.path.dirname(__file__), "..")
_instance_mod = _load_mod("mc_instance", os.path.join(_base, "instance.py"))
_greedy_mod = _load_mod(
    "mc_greedy", os.path.join(_base, "heuristics", "greedy_coverage.py")
)

MaxCoverageInstance = _instance_mod.MaxCoverageInstance
MaxCoverageSolution = _instance_mod.MaxCoverageSolution
greedy_coverage = _greedy_mod.greedy_coverage
exhaustive_coverage = _greedy_mod.exhaustive_coverage


class TestMaxCoverageInstance:
    """Tests for instance construction."""

    def test_random_instance_dimensions(self):
        inst = MaxCoverageInstance.random(n=10, m=5, k=2, seed=1)
        assert inst.n == 10
        assert inst.m == 5
        assert inst.k == 2
        assert len(inst.subsets) == 5

    def test_subsets_contain_valid_elements(self):
        inst = MaxCoverageInstance.random(n=10, m=5, k=2, seed=2)
        for s in inst.subsets:
            assert len(s) > 0
            for e in s:
                assert 0 <= e < inst.n

    def test_coverage_computation(self):
        inst = MaxCoverageInstance(
            n=5, m=3, k=2,
            subsets=[{0, 1}, {1, 2, 3}, {3, 4}]
        )
        cov = inst.coverage([0, 2])
        assert cov == {0, 1, 3, 4}


class TestGreedyCoverage:
    """Tests for the greedy coverage heuristic."""

    def test_respects_budget(self):
        inst = MaxCoverageInstance.random(n=15, m=8, k=3, seed=10)
        sol = greedy_coverage(inst)
        assert len(sol.selected) <= inst.k

    def test_handcrafted_optimal(self):
        """Greedy should find optimal on easy instance."""
        inst = MaxCoverageInstance(
            n=6, m=3, k=2,
            subsets=[{0, 1, 2, 3}, {2, 3, 4, 5}, {0, 5}]
        )
        sol = greedy_coverage(inst)
        assert sol.objective == 6  # Select S0 and S1

    def test_coverage_matches_objective(self):
        inst = MaxCoverageInstance.random(n=15, m=8, k=3, seed=11)
        sol = greedy_coverage(inst)
        assert sol.objective == len(sol.covered)
        # Verify covered matches union
        expected = inst.coverage(sol.selected)
        assert sol.covered == expected

    def test_greedy_approximation_ratio(self):
        """Greedy should achieve at least (1 - 1/e) of optimal."""
        inst = MaxCoverageInstance.random(n=20, m=10, k=3, seed=12)
        sol_greedy = greedy_coverage(inst)
        sol_opt = exhaustive_coverage(inst)
        ratio = 1 - 1 / math.e  # ~0.632
        assert sol_greedy.objective >= sol_opt.objective * ratio - 1e-9

    def test_single_subset_budget(self):
        inst = MaxCoverageInstance.random(n=10, m=5, k=1, seed=13)
        sol = greedy_coverage(inst)
        assert len(sol.selected) == 1

    def test_budget_exceeds_subsets(self):
        inst = MaxCoverageInstance(
            n=5, m=2, k=5,
            subsets=[{0, 1, 2}, {3, 4}]
        )
        sol = greedy_coverage(inst)
        assert sol.objective == 5
        assert len(sol.selected) <= 2


class TestExhaustiveCoverage:
    """Tests for exhaustive enumeration."""

    def test_optimal_small(self):
        inst = MaxCoverageInstance(
            n=4, m=3, k=1,
            subsets=[{0}, {0, 1, 2}, {3}]
        )
        sol = exhaustive_coverage(inst)
        assert sol.objective == 3
        assert 1 in sol.selected

    def test_optimal_vs_greedy(self):
        inst = MaxCoverageInstance.random(n=12, m=6, k=2, seed=20)
        sol_greedy = greedy_coverage(inst)
        sol_opt = exhaustive_coverage(inst)
        assert sol_opt.objective >= sol_greedy.objective
