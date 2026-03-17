"""
Test Suite — VNS, Memetic Algorithm, and EDA.

Tests cover:
1. Variable Neighborhood Search (VNS)
2. Memetic Algorithm (MA)
3. Estimation of Distribution Algorithm (EDA)
"""

import sys
import os
import numpy as np
import pytest

_flow_shop_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _flow_shop_dir)

from instance import FlowShopInstance, compute_makespan
from heuristics.neh import neh
from metaheuristics.vns import vns
from metaheuristics.memetic_algorithm import memetic_algorithm
from metaheuristics.eda import eda


# ──────────────────────────────────────────────
# VNS Tests
# ──────────────────────────────────────────────

class TestVNS:
    """Verify Variable Neighborhood Search correctness."""

    def test_returns_valid_permutation(self):
        instance = FlowShopInstance.random(n=10, m=3, seed=42)
        sol = vns(instance, max_iterations=5, seed=42)
        assert sorted(sol.permutation) == list(range(10))

    def test_makespan_is_correct(self):
        instance = FlowShopInstance.random(n=10, m=3, seed=7)
        sol = vns(instance, max_iterations=5, seed=7)
        assert sol.makespan == compute_makespan(instance, sol.permutation)

    def test_no_worse_than_neh(self):
        instance = FlowShopInstance.random(n=15, m=4, seed=42)
        sol_neh = neh(instance)
        sol_vns = vns(instance, max_iterations=10, seed=42)
        assert sol_vns.makespan <= sol_neh.makespan

    def test_deterministic_with_seed(self):
        instance = FlowShopInstance.random(n=10, m=3, seed=42)
        sol_a = vns(instance, max_iterations=5, seed=123)
        sol_b = vns(instance, max_iterations=5, seed=123)
        assert sol_a.makespan == sol_b.makespan
        assert sol_a.permutation == sol_b.permutation

    def test_single_job(self):
        instance = FlowShopInstance.random(n=1, m=3, seed=42)
        sol = vns(instance, max_iterations=3, seed=42)
        assert sol.permutation == [0]

    def test_two_jobs(self):
        instance = FlowShopInstance.random(n=2, m=5, seed=42)
        sol = vns(instance, max_iterations=3, seed=42)
        assert sorted(sol.permutation) == [0, 1]

    def test_time_limit(self):
        instance = FlowShopInstance.random(n=15, m=4, seed=42)
        sol = vns(instance, time_limit=2.0, seed=42)
        assert sorted(sol.permutation) == list(range(15))
        assert sol.makespan == compute_makespan(instance, sol.permutation)

    def test_custom_k_max(self):
        instance = FlowShopInstance.random(n=10, m=3, seed=42)
        sol = vns(instance, k_max=3, max_iterations=5, seed=42)
        assert sorted(sol.permutation) == list(range(10))


# ──────────────────────────────────────────────
# Memetic Algorithm Tests
# ──────────────────────────────────────────────

class TestMemeticAlgorithm:
    """Verify Memetic Algorithm correctness."""

    def test_returns_valid_permutation(self):
        instance = FlowShopInstance.random(n=10, m=3, seed=42)
        sol = memetic_algorithm(
            instance, population_size=10, generations=5, seed=42,
        )
        assert sorted(sol.permutation) == list(range(10))

    def test_makespan_is_correct(self):
        instance = FlowShopInstance.random(n=10, m=3, seed=7)
        sol = memetic_algorithm(
            instance, population_size=10, generations=5, seed=7,
        )
        assert sol.makespan == compute_makespan(instance, sol.permutation)

    def test_no_worse_than_neh(self):
        instance = FlowShopInstance.random(n=15, m=4, seed=42)
        sol_neh = neh(instance)
        sol_ma = memetic_algorithm(
            instance, population_size=15, generations=20, seed=42,
        )
        assert sol_ma.makespan <= sol_neh.makespan

    def test_beats_plain_random(self):
        """MA should be much better than random permutations."""
        instance = FlowShopInstance.random(n=20, m=5, seed=42)
        rng = np.random.default_rng(42)
        random_best = float("inf")
        for _ in range(100):
            perm = list(range(20))
            rng.shuffle(perm)
            ms = compute_makespan(instance, perm)
            random_best = min(random_best, ms)

        sol_ma = memetic_algorithm(
            instance, population_size=15, generations=15, seed=42,
        )
        assert sol_ma.makespan < random_best

    def test_deterministic_with_seed(self):
        instance = FlowShopInstance.random(n=10, m=3, seed=42)
        sol_a = memetic_algorithm(
            instance, population_size=10, generations=5, seed=123,
        )
        sol_b = memetic_algorithm(
            instance, population_size=10, generations=5, seed=123,
        )
        assert sol_a.makespan == sol_b.makespan
        assert sol_a.permutation == sol_b.permutation

    def test_single_job(self):
        instance = FlowShopInstance.random(n=1, m=3, seed=42)
        sol = memetic_algorithm(
            instance, population_size=6, generations=3, seed=42,
        )
        assert sol.permutation == [0]

    def test_time_limit(self):
        instance = FlowShopInstance.random(n=15, m=4, seed=42)
        sol = memetic_algorithm(instance, time_limit=2.0, seed=42)
        assert sorted(sol.permutation) == list(range(15))
        assert sol.makespan == compute_makespan(instance, sol.permutation)

    def test_high_mutation_rate(self):
        instance = FlowShopInstance.random(n=10, m=3, seed=42)
        sol = memetic_algorithm(
            instance, population_size=10, generations=5,
            mutation_rate=0.8, seed=42,
        )
        assert sorted(sol.permutation) == list(range(10))


# ──────────────────────────────────────────────
# EDA Tests
# ──────────────────────────────────────────────

class TestEDA:
    """Verify Estimation of Distribution Algorithm correctness."""

    def test_returns_valid_permutation(self):
        instance = FlowShopInstance.random(n=10, m=3, seed=42)
        sol = eda(
            instance, population_size=15, generations=20, seed=42,
        )
        assert sorted(sol.permutation) == list(range(10))

    def test_makespan_is_correct(self):
        instance = FlowShopInstance.random(n=10, m=3, seed=7)
        sol = eda(
            instance, population_size=15, generations=20, seed=7,
        )
        assert sol.makespan == compute_makespan(instance, sol.permutation)

    def test_no_worse_than_neh(self):
        instance = FlowShopInstance.random(n=15, m=4, seed=42)
        sol_neh = neh(instance)
        sol_eda = eda(
            instance, population_size=20, generations=100, seed=42,
        )
        assert sol_eda.makespan <= sol_neh.makespan

    def test_improves_with_more_generations(self):
        instance = FlowShopInstance.random(n=15, m=4, seed=42)
        sol_short = eda(
            instance, population_size=15, generations=10, seed=42,
        )
        sol_long = eda(
            instance, population_size=15, generations=200, seed=42,
        )
        assert sol_long.makespan <= sol_short.makespan

    def test_deterministic_with_seed(self):
        instance = FlowShopInstance.random(n=10, m=3, seed=42)
        sol_a = eda(
            instance, population_size=15, generations=20, seed=123,
        )
        sol_b = eda(
            instance, population_size=15, generations=20, seed=123,
        )
        assert sol_a.makespan == sol_b.makespan
        assert sol_a.permutation == sol_b.permutation

    def test_time_limit(self):
        instance = FlowShopInstance.random(n=15, m=4, seed=42)
        sol = eda(instance, time_limit=2.0, seed=42)
        assert sorted(sol.permutation) == list(range(15))
        assert sol.makespan == compute_makespan(instance, sol.permutation)

    def test_single_job(self):
        instance = FlowShopInstance.random(n=1, m=3, seed=42)
        sol = eda(
            instance, population_size=8, generations=5, seed=42,
        )
        assert sol.permutation == [0]

    def test_custom_selection_ratio(self):
        instance = FlowShopInstance.random(n=10, m=3, seed=42)
        sol = eda(
            instance, population_size=15, selection_ratio=0.6,
            generations=20, seed=42,
        )
        assert sorted(sol.permutation) == list(range(10))

    def test_custom_learning_rate(self):
        instance = FlowShopInstance.random(n=10, m=3, seed=42)
        sol = eda(
            instance, population_size=15, learning_rate=0.8,
            generations=20, seed=42,
        )
        assert sorted(sol.permutation) == list(range(10))
