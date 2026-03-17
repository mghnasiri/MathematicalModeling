"""
Test Suite — NEHKK, Artificial Bee Colony, and Beam Search.

Tests cover:
1. NEHKK (NEH with Kalczynski-Kamburowski tie-breaking) heuristic
2. Beam Search constructive heuristic
3. Artificial Bee Colony (ABC) metaheuristic
"""

import sys
import os
import numpy as np
import pytest

_flow_shop_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _flow_shop_dir)

from instance import FlowShopInstance, compute_makespan
from heuristics.neh import neh
from heuristics.nehkk import nehkk
from heuristics.beam_search import beam_search
from metaheuristics.bee_colony import artificial_bee_colony


# ──────────────────────────────────────────────
# NEHKK Tests
# ──────────────────────────────────────────────

class TestNEHKK:
    """Verify NEHKK correctness and tie-breaking improvement."""

    def test_returns_valid_permutation(self):
        instance = FlowShopInstance.random(n=10, m=3, seed=42)
        sol = nehkk(instance)
        assert sorted(sol.permutation) == list(range(10))

    def test_makespan_is_correct(self):
        instance = FlowShopInstance.random(n=15, m=4, seed=7)
        sol = nehkk(instance)
        assert sol.makespan == compute_makespan(instance, sol.permutation)

    def test_no_worse_than_neh_on_average(self):
        """NEHKK should match or beat NEH on most instances."""
        better_or_equal = 0
        n_tests = 15
        for s in range(n_tests):
            instance = FlowShopInstance.random(n=20, m=5, seed=s)
            sol_neh = neh(instance)
            sol_kk = nehkk(instance)
            if sol_kk.makespan <= sol_neh.makespan:
                better_or_equal += 1
        # Should match or beat NEH on majority of instances
        assert better_or_equal >= n_tests // 2

    def test_deterministic(self):
        instance = FlowShopInstance.random(n=15, m=4, seed=42)
        sol_a = nehkk(instance)
        sol_b = nehkk(instance)
        assert sol_a.makespan == sol_b.makespan
        assert sol_a.permutation == sol_b.permutation

    def test_single_job(self):
        instance = FlowShopInstance.random(n=1, m=3, seed=42)
        sol = nehkk(instance)
        assert sol.permutation == [0]

    def test_two_jobs(self):
        instance = FlowShopInstance.random(n=2, m=5, seed=42)
        sol = nehkk(instance)
        assert sorted(sol.permutation) == [0, 1]
        assert sol.makespan == compute_makespan(instance, sol.permutation)

    def test_two_machines(self):
        instance = FlowShopInstance.random(n=10, m=2, seed=42)
        sol = nehkk(instance)
        assert sorted(sol.permutation) == list(range(10))

    def test_small_handcrafted(self):
        instance = FlowShopInstance(
            n=4, m=2,
            processing_times=np.array([[3, 7, 2, 5], [6, 1, 8, 4]]),
        )
        sol = nehkk(instance)
        assert sorted(sol.permutation) == [0, 1, 2, 3]
        assert sol.makespan == compute_makespan(instance, sol.permutation)


# ──────────────────────────────────────────────
# Beam Search Tests
# ──────────────────────────────────────────────

class TestBeamSearch:
    """Verify Beam Search correctness and improvement over NEH."""

    def test_returns_valid_permutation(self):
        instance = FlowShopInstance.random(n=10, m=3, seed=42)
        sol = beam_search(instance, beam_width=3)
        assert sorted(sol.permutation) == list(range(10))

    def test_makespan_is_correct(self):
        instance = FlowShopInstance.random(n=15, m=4, seed=7)
        sol = beam_search(instance, beam_width=5)
        assert sol.makespan == compute_makespan(instance, sol.permutation)

    def test_beam_1_matches_neh(self):
        """Beam width 1 should produce identical result to NEH."""
        instance = FlowShopInstance.random(n=15, m=4, seed=42)
        sol_neh = neh(instance)
        sol_bs1 = beam_search(instance, beam_width=1)
        assert sol_bs1.makespan == sol_neh.makespan

    def test_wider_beam_no_worse(self):
        """Wider beam should never be worse than narrower beam."""
        instance = FlowShopInstance.random(n=20, m=5, seed=42)
        sol_3 = beam_search(instance, beam_width=3)
        sol_10 = beam_search(instance, beam_width=10)
        assert sol_10.makespan <= sol_3.makespan

    def test_improves_over_neh(self):
        """Beam search with width > 1 should improve over NEH on some instances."""
        improved = 0
        for s in range(10):
            instance = FlowShopInstance.random(n=20, m=5, seed=s)
            sol_neh = neh(instance)
            sol_bs = beam_search(instance, beam_width=10)
            if sol_bs.makespan < sol_neh.makespan:
                improved += 1
        assert improved >= 1

    def test_deterministic(self):
        instance = FlowShopInstance.random(n=15, m=4, seed=42)
        sol_a = beam_search(instance, beam_width=5)
        sol_b = beam_search(instance, beam_width=5)
        assert sol_a.makespan == sol_b.makespan
        assert sol_a.permutation == sol_b.permutation

    def test_single_job(self):
        instance = FlowShopInstance.random(n=1, m=3, seed=42)
        sol = beam_search(instance, beam_width=5)
        assert sol.permutation == [0]

    def test_large_beam_small_instance(self):
        """Beam wider than permutation count should still work."""
        instance = FlowShopInstance.random(n=4, m=3, seed=42)
        sol = beam_search(instance, beam_width=50)
        assert sorted(sol.permutation) == [0, 1, 2, 3]


# ──────────────────────────────────────────────
# ABC Tests
# ──────────────────────────────────────────────

class TestABC:
    """Verify Artificial Bee Colony correctness and improvement."""

    def test_returns_valid_permutation(self):
        instance = FlowShopInstance.random(n=10, m=3, seed=42)
        sol = artificial_bee_colony(
            instance, colony_size=10, max_iterations=20, seed=42,
        )
        assert sorted(sol.permutation) == list(range(10))

    def test_makespan_is_correct(self):
        instance = FlowShopInstance.random(n=15, m=4, seed=7)
        sol = artificial_bee_colony(
            instance, colony_size=10, max_iterations=30, seed=7,
        )
        assert sol.makespan == compute_makespan(instance, sol.permutation)

    def test_no_worse_than_neh(self):
        instance = FlowShopInstance.random(n=20, m=5, seed=42)
        sol_neh = neh(instance)
        sol_abc = artificial_bee_colony(
            instance, colony_size=20, max_iterations=100, seed=42,
        )
        assert sol_abc.makespan <= sol_neh.makespan

    def test_improves_with_more_iterations(self):
        instance = FlowShopInstance.random(n=20, m=5, seed=42)
        sol_short = artificial_bee_colony(
            instance, colony_size=10, max_iterations=10, seed=42,
        )
        sol_long = artificial_bee_colony(
            instance, colony_size=10, max_iterations=200, seed=42,
        )
        assert sol_long.makespan <= sol_short.makespan

    def test_deterministic_with_seed(self):
        instance = FlowShopInstance.random(n=15, m=4, seed=42)
        sol_a = artificial_bee_colony(
            instance, colony_size=10, max_iterations=30, seed=123,
        )
        sol_b = artificial_bee_colony(
            instance, colony_size=10, max_iterations=30, seed=123,
        )
        assert sol_a.makespan == sol_b.makespan
        assert sol_a.permutation == sol_b.permutation

    def test_time_limit(self):
        instance = FlowShopInstance.random(n=20, m=5, seed=42)
        sol = artificial_bee_colony(
            instance, time_limit=1.0, seed=42,
        )
        assert sorted(sol.permutation) == list(range(20))
        assert sol.makespan == compute_makespan(instance, sol.permutation)

    def test_single_job(self):
        instance = FlowShopInstance.random(n=1, m=3, seed=42)
        sol = artificial_bee_colony(
            instance, colony_size=6, max_iterations=5, seed=42,
        )
        assert sol.permutation == [0]

    def test_custom_limit(self):
        instance = FlowShopInstance.random(n=10, m=3, seed=42)
        sol = artificial_bee_colony(
            instance, colony_size=10, max_iterations=30,
            limit=10, seed=42,
        )
        assert sorted(sol.permutation) == list(range(10))
