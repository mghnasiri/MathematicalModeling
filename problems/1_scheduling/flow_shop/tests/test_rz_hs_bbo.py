"""
Test Suite — Rajendran-Ziegler, Harmony Search, and BBO.

Tests cover:
1. Rajendran-Ziegler (RZ) constructive heuristic
2. Harmony Search (HS) metaheuristic
3. Biogeography-Based Optimization (BBO) metaheuristic
"""

import sys
import os
import numpy as np
import pytest

_flow_shop_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _flow_shop_dir)

from instance import FlowShopInstance, compute_makespan
from heuristics.neh import neh
from heuristics.rajendran_ziegler import rajendran_ziegler
from metaheuristics.harmony_search import harmony_search
from metaheuristics.bbo import bbo


# ──────────────────────────────────────────────
# Rajendran-Ziegler Tests
# ──────────────────────────────────────────────

class TestRajendranZiegler:
    """Verify RZ heuristic correctness and improvement pass."""

    def test_returns_valid_permutation(self):
        instance = FlowShopInstance.random(n=10, m=3, seed=42)
        sol = rajendran_ziegler(instance)
        assert sorted(sol.permutation) == list(range(10))

    def test_makespan_is_correct(self):
        instance = FlowShopInstance.random(n=15, m=4, seed=7)
        sol = rajendran_ziegler(instance)
        assert sol.makespan == compute_makespan(instance, sol.permutation)

    def test_deterministic(self):
        instance = FlowShopInstance.random(n=15, m=4, seed=42)
        sol_a = rajendran_ziegler(instance)
        sol_b = rajendran_ziegler(instance)
        assert sol_a.makespan == sol_b.makespan
        assert sol_a.permutation == sol_b.permutation

    def test_single_job(self):
        instance = FlowShopInstance.random(n=1, m=3, seed=42)
        sol = rajendran_ziegler(instance)
        assert sol.permutation == [0]

    def test_two_jobs(self):
        instance = FlowShopInstance.random(n=2, m=5, seed=42)
        sol = rajendran_ziegler(instance)
        assert sorted(sol.permutation) == [0, 1]
        assert sol.makespan == compute_makespan(instance, sol.permutation)

    def test_improvement_pass_helps(self):
        """RZ should produce competitive results thanks to improvement."""
        competitive = 0
        for s in range(10):
            instance = FlowShopInstance.random(n=20, m=5, seed=s)
            sol_neh = neh(instance)
            sol_rz = rajendran_ziegler(instance)
            # Allow up to 5% worse than NEH (RZ uses different initial order)
            if sol_rz.makespan <= sol_neh.makespan * 1.05:
                competitive += 1
        assert competitive >= 7

    def test_small_handcrafted(self):
        instance = FlowShopInstance(
            n=4, m=2,
            processing_times=np.array([[3, 7, 2, 5], [6, 1, 8, 4]]),
        )
        sol = rajendran_ziegler(instance)
        assert sorted(sol.permutation) == [0, 1, 2, 3]
        assert sol.makespan == compute_makespan(instance, sol.permutation)


# ──────────────────────────────────────────────
# Harmony Search Tests
# ──────────────────────────────────────────────

class TestHarmonySearch:
    """Verify Harmony Search correctness."""

    def test_returns_valid_permutation(self):
        instance = FlowShopInstance.random(n=10, m=3, seed=42)
        sol = harmony_search(
            instance, harmony_memory_size=10, max_iterations=20, seed=42,
        )
        assert sorted(sol.permutation) == list(range(10))

    def test_makespan_is_correct(self):
        instance = FlowShopInstance.random(n=15, m=4, seed=7)
        sol = harmony_search(
            instance, harmony_memory_size=10, max_iterations=30, seed=7,
        )
        assert sol.makespan == compute_makespan(instance, sol.permutation)

    def test_no_worse_than_neh(self):
        instance = FlowShopInstance.random(n=20, m=5, seed=42)
        sol_neh = neh(instance)
        sol_hs = harmony_search(
            instance, harmony_memory_size=20, max_iterations=200, seed=42,
        )
        assert sol_hs.makespan <= sol_neh.makespan

    def test_improves_with_more_iterations(self):
        instance = FlowShopInstance.random(n=20, m=5, seed=42)
        sol_short = harmony_search(
            instance, harmony_memory_size=10, max_iterations=10, seed=42,
        )
        sol_long = harmony_search(
            instance, harmony_memory_size=10, max_iterations=300, seed=42,
        )
        assert sol_long.makespan <= sol_short.makespan

    def test_deterministic_with_seed(self):
        instance = FlowShopInstance.random(n=15, m=4, seed=42)
        sol_a = harmony_search(
            instance, harmony_memory_size=10, max_iterations=30, seed=123,
        )
        sol_b = harmony_search(
            instance, harmony_memory_size=10, max_iterations=30, seed=123,
        )
        assert sol_a.makespan == sol_b.makespan
        assert sol_a.permutation == sol_b.permutation

    def test_time_limit(self):
        instance = FlowShopInstance.random(n=20, m=5, seed=42)
        sol = harmony_search(instance, time_limit=1.0, seed=42)
        assert sorted(sol.permutation) == list(range(20))
        assert sol.makespan == compute_makespan(instance, sol.permutation)

    def test_single_job(self):
        instance = FlowShopInstance.random(n=1, m=3, seed=42)
        sol = harmony_search(
            instance, harmony_memory_size=6, max_iterations=5, seed=42,
        )
        assert sol.permutation == [0]

    def test_custom_parameters(self):
        instance = FlowShopInstance.random(n=10, m=3, seed=42)
        sol = harmony_search(
            instance, harmony_memory_size=10, hmcr=0.95, par=0.5,
            max_iterations=30, seed=42,
        )
        assert sorted(sol.permutation) == list(range(10))


# ──────────────────────────────────────────────
# BBO Tests
# ──────────────────────────────────────────────

class TestBBO:
    """Verify Biogeography-Based Optimization correctness."""

    def test_returns_valid_permutation(self):
        instance = FlowShopInstance.random(n=10, m=3, seed=42)
        sol = bbo(
            instance, population_size=10, max_iterations=20, seed=42,
        )
        assert sorted(sol.permutation) == list(range(10))

    def test_makespan_is_correct(self):
        instance = FlowShopInstance.random(n=15, m=4, seed=7)
        sol = bbo(
            instance, population_size=10, max_iterations=30, seed=7,
        )
        assert sol.makespan == compute_makespan(instance, sol.permutation)

    def test_no_worse_than_neh(self):
        instance = FlowShopInstance.random(n=20, m=5, seed=42)
        sol_neh = neh(instance)
        sol_bbo = bbo(
            instance, population_size=20, max_iterations=100, seed=42,
        )
        assert sol_bbo.makespan <= sol_neh.makespan

    def test_improves_with_more_iterations(self):
        instance = FlowShopInstance.random(n=20, m=5, seed=42)
        sol_short = bbo(
            instance, population_size=10, max_iterations=10, seed=42,
        )
        sol_long = bbo(
            instance, population_size=10, max_iterations=200, seed=42,
        )
        assert sol_long.makespan <= sol_short.makespan

    def test_deterministic_with_seed(self):
        instance = FlowShopInstance.random(n=15, m=4, seed=42)
        sol_a = bbo(
            instance, population_size=10, max_iterations=30, seed=123,
        )
        sol_b = bbo(
            instance, population_size=10, max_iterations=30, seed=123,
        )
        assert sol_a.makespan == sol_b.makespan
        assert sol_a.permutation == sol_b.permutation

    def test_time_limit(self):
        instance = FlowShopInstance.random(n=20, m=5, seed=42)
        sol = bbo(instance, time_limit=1.0, seed=42)
        assert sorted(sol.permutation) == list(range(20))
        assert sol.makespan == compute_makespan(instance, sol.permutation)

    def test_single_job(self):
        instance = FlowShopInstance.random(n=1, m=3, seed=42)
        sol = bbo(
            instance, population_size=6, max_iterations=5, seed=42,
        )
        assert sol.permutation == [0]

    def test_custom_mutation_rate(self):
        instance = FlowShopInstance.random(n=10, m=3, seed=42)
        sol = bbo(
            instance, population_size=10, mutation_rate=0.1,
            max_iterations=30, seed=42,
        )
        assert sorted(sol.permutation) == list(range(10))
