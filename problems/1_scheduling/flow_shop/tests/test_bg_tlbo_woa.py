"""
Test Suite — Bonney-Gundry, TLBO, and WOA.

Tests cover:
1. Bonney-Gundry Slope Index heuristic
2. Teaching-Learning-Based Optimization (TLBO) metaheuristic
3. Whale Optimization Algorithm (WOA) metaheuristic
"""

import sys
import os
import numpy as np
import pytest

_flow_shop_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _flow_shop_dir)

from instance import FlowShopInstance, compute_makespan
from heuristics.neh import neh
from heuristics.bonney_gundry import bonney_gundry
from metaheuristics.tlbo import tlbo
from metaheuristics.whale_optimization import whale_optimization


# ──────────────────────────────────────────────
# Bonney-Gundry Tests
# ──────────────────────────────────────────────

class TestBonneyGundry:
    """Verify Bonney-Gundry heuristic correctness."""

    def test_returns_valid_permutation(self):
        instance = FlowShopInstance.random(n=10, m=3, seed=42)
        sol = bonney_gundry(instance)
        assert sorted(sol.permutation) == list(range(10))

    def test_makespan_is_correct(self):
        instance = FlowShopInstance.random(n=15, m=4, seed=7)
        sol = bonney_gundry(instance)
        assert sol.makespan == compute_makespan(instance, sol.permutation)

    def test_deterministic(self):
        instance = FlowShopInstance.random(n=15, m=4, seed=42)
        sol_a = bonney_gundry(instance)
        sol_b = bonney_gundry(instance)
        assert sol_a.makespan == sol_b.makespan
        assert sol_a.permutation == sol_b.permutation

    def test_single_job(self):
        instance = FlowShopInstance.random(n=1, m=3, seed=42)
        sol = bonney_gundry(instance)
        assert sol.permutation == [0]

    def test_two_jobs(self):
        instance = FlowShopInstance.random(n=2, m=5, seed=42)
        sol = bonney_gundry(instance)
        assert sorted(sol.permutation) == [0, 1]

    def test_two_machines(self):
        instance = FlowShopInstance.random(n=10, m=2, seed=42)
        sol = bonney_gundry(instance)
        assert sorted(sol.permutation) == list(range(10))

    def test_handcrafted_instance(self):
        """Jobs with increasing processing times should go first."""
        instance = FlowShopInstance(
            n=3, m=3,
            processing_times=np.array([
                [1, 5, 3],   # Machine 1
                [5, 5, 3],   # Machine 2
                [9, 5, 3],   # Machine 3
            ]),
        )
        sol = bonney_gundry(instance)
        assert sorted(sol.permutation) == [0, 1, 2]
        assert sol.makespan == compute_makespan(instance, sol.permutation)

    def test_many_machines(self):
        instance = FlowShopInstance.random(n=10, m=15, seed=42)
        sol = bonney_gundry(instance)
        assert sorted(sol.permutation) == list(range(10))


# ──────────────────────────────────────────────
# TLBO Tests
# ──────────────────────────────────────────────

class TestTLBO:
    """Verify TLBO correctness and improvement."""

    def test_returns_valid_permutation(self):
        instance = FlowShopInstance.random(n=10, m=3, seed=42)
        sol = tlbo(instance, population_size=10, max_iterations=20, seed=42)
        assert sorted(sol.permutation) == list(range(10))

    def test_makespan_is_correct(self):
        instance = FlowShopInstance.random(n=15, m=4, seed=7)
        sol = tlbo(instance, population_size=10, max_iterations=30, seed=7)
        assert sol.makespan == compute_makespan(instance, sol.permutation)

    def test_no_worse_than_neh(self):
        instance = FlowShopInstance.random(n=20, m=5, seed=42)
        sol_neh = neh(instance)
        sol_tlbo = tlbo(
            instance, population_size=20, max_iterations=100, seed=42,
        )
        assert sol_tlbo.makespan <= sol_neh.makespan

    def test_improves_with_more_iterations(self):
        instance = FlowShopInstance.random(n=20, m=5, seed=42)
        sol_short = tlbo(
            instance, population_size=10, max_iterations=10, seed=42,
        )
        sol_long = tlbo(
            instance, population_size=10, max_iterations=200, seed=42,
        )
        assert sol_long.makespan <= sol_short.makespan

    def test_deterministic_with_seed(self):
        instance = FlowShopInstance.random(n=15, m=4, seed=42)
        sol_a = tlbo(
            instance, population_size=10, max_iterations=30, seed=123,
        )
        sol_b = tlbo(
            instance, population_size=10, max_iterations=30, seed=123,
        )
        assert sol_a.makespan == sol_b.makespan
        assert sol_a.permutation == sol_b.permutation

    def test_time_limit(self):
        instance = FlowShopInstance.random(n=20, m=5, seed=42)
        sol = tlbo(instance, time_limit=1.0, seed=42)
        assert sorted(sol.permutation) == list(range(20))
        assert sol.makespan == compute_makespan(instance, sol.permutation)

    def test_single_job(self):
        instance = FlowShopInstance.random(n=1, m=3, seed=42)
        sol = tlbo(
            instance, population_size=6, max_iterations=5, seed=42,
        )
        assert sol.permutation == [0]

    def test_small_population(self):
        instance = FlowShopInstance.random(n=10, m=3, seed=42)
        sol = tlbo(
            instance, population_size=4, max_iterations=20, seed=42,
        )
        assert sorted(sol.permutation) == list(range(10))


# ──────────────────────────────────────────────
# WOA Tests
# ──────────────────────────────────────────────

class TestWOA:
    """Verify Whale Optimization Algorithm correctness."""

    def test_returns_valid_permutation(self):
        instance = FlowShopInstance.random(n=10, m=3, seed=42)
        sol = whale_optimization(
            instance, population_size=10, max_iterations=20, seed=42,
        )
        assert sorted(sol.permutation) == list(range(10))

    def test_makespan_is_correct(self):
        instance = FlowShopInstance.random(n=15, m=4, seed=7)
        sol = whale_optimization(
            instance, population_size=10, max_iterations=30, seed=7,
        )
        assert sol.makespan == compute_makespan(instance, sol.permutation)

    def test_no_worse_than_neh(self):
        instance = FlowShopInstance.random(n=20, m=5, seed=42)
        sol_neh = neh(instance)
        sol_woa = whale_optimization(
            instance, population_size=20, max_iterations=100, seed=42,
        )
        assert sol_woa.makespan <= sol_neh.makespan

    def test_improves_with_more_iterations(self):
        instance = FlowShopInstance.random(n=20, m=5, seed=42)
        sol_short = whale_optimization(
            instance, population_size=10, max_iterations=10, seed=42,
        )
        sol_long = whale_optimization(
            instance, population_size=10, max_iterations=200, seed=42,
        )
        assert sol_long.makespan <= sol_short.makespan

    def test_deterministic_with_seed(self):
        instance = FlowShopInstance.random(n=15, m=4, seed=42)
        sol_a = whale_optimization(
            instance, population_size=10, max_iterations=30, seed=123,
        )
        sol_b = whale_optimization(
            instance, population_size=10, max_iterations=30, seed=123,
        )
        assert sol_a.makespan == sol_b.makespan
        assert sol_a.permutation == sol_b.permutation

    def test_time_limit(self):
        instance = FlowShopInstance.random(n=20, m=5, seed=42)
        sol = whale_optimization(instance, time_limit=1.0, seed=42)
        assert sorted(sol.permutation) == list(range(20))
        assert sol.makespan == compute_makespan(instance, sol.permutation)

    def test_single_job(self):
        instance = FlowShopInstance.random(n=1, m=3, seed=42)
        sol = whale_optimization(
            instance, population_size=6, max_iterations=5, seed=42,
        )
        assert sol.permutation == [0]

    def test_small_population(self):
        instance = FlowShopInstance.random(n=10, m=3, seed=42)
        sol = whale_optimization(
            instance, population_size=4, max_iterations=20, seed=42,
        )
        assert sorted(sol.permutation) == list(range(10))
