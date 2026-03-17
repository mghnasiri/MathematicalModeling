"""
Test Suite — PSO, Differential Evolution, Scatter Search, and RA Heuristic.

Tests cover:
1. Particle Swarm Optimization (PSO) metaheuristic
2. Differential Evolution (DE) metaheuristic
3. Scatter Search (SS) metaheuristic
4. RA multi-ordering constructive heuristic
"""

import sys
import os
import numpy as np
import pytest

# Add parent directories to path
_flow_shop_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _flow_shop_dir)

from instance import FlowShopInstance, compute_makespan
from heuristics.neh import neh
from heuristics.ra_heuristic import ra_heuristic
from metaheuristics.particle_swarm import particle_swarm_optimization
from metaheuristics.differential_evolution import differential_evolution
from metaheuristics.scatter_search import scatter_search


# ──────────────────────────────────────────────
# PSO Tests
# ──────────────────────────────────────────────

class TestPSO:
    """Verify Particle Swarm Optimization correctness and improvement."""

    def test_returns_valid_permutation(self):
        instance = FlowShopInstance.random(n=10, m=3, seed=42)
        sol = particle_swarm_optimization(
            instance, swarm_size=10, max_iterations=20, seed=42,
        )
        assert sorted(sol.permutation) == list(range(10))

    def test_makespan_is_correct(self):
        instance = FlowShopInstance.random(n=15, m=4, seed=7)
        sol = particle_swarm_optimization(
            instance, swarm_size=10, max_iterations=30, seed=7,
        )
        assert sol.makespan == compute_makespan(instance, sol.permutation)

    def test_no_worse_than_neh(self):
        instance = FlowShopInstance.random(n=20, m=5, seed=42)
        sol_neh = neh(instance)
        sol_pso = particle_swarm_optimization(
            instance, swarm_size=20, max_iterations=100, seed=42,
        )
        assert sol_pso.makespan <= sol_neh.makespan

    def test_improves_with_more_iterations(self):
        instance = FlowShopInstance.random(n=20, m=5, seed=42)
        sol_short = particle_swarm_optimization(
            instance, swarm_size=15, max_iterations=10, seed=42,
        )
        sol_long = particle_swarm_optimization(
            instance, swarm_size=15, max_iterations=200, seed=42,
        )
        assert sol_long.makespan <= sol_short.makespan

    def test_deterministic_with_seed(self):
        instance = FlowShopInstance.random(n=15, m=4, seed=42)
        sol_a = particle_swarm_optimization(
            instance, swarm_size=10, max_iterations=30, seed=123,
        )
        sol_b = particle_swarm_optimization(
            instance, swarm_size=10, max_iterations=30, seed=123,
        )
        assert sol_a.makespan == sol_b.makespan
        assert sol_a.permutation == sol_b.permutation

    def test_time_limit(self):
        instance = FlowShopInstance.random(n=20, m=5, seed=42)
        sol = particle_swarm_optimization(
            instance, time_limit=1.0, seed=42,
        )
        assert sorted(sol.permutation) == list(range(20))
        assert sol.makespan == compute_makespan(instance, sol.permutation)

    def test_without_local_search(self):
        instance = FlowShopInstance.random(n=15, m=4, seed=42)
        sol = particle_swarm_optimization(
            instance, swarm_size=10, max_iterations=30,
            local_search=False, seed=42,
        )
        assert sorted(sol.permutation) == list(range(15))

    def test_single_job(self):
        instance = FlowShopInstance.random(n=1, m=3, seed=42)
        sol = particle_swarm_optimization(
            instance, swarm_size=5, max_iterations=5, seed=42,
        )
        assert sol.permutation == [0]


# ──────────────────────────────────────────────
# Differential Evolution Tests
# ──────────────────────────────────────────────

class TestDE:
    """Verify Differential Evolution correctness and improvement."""

    def test_returns_valid_permutation(self):
        instance = FlowShopInstance.random(n=10, m=3, seed=42)
        sol = differential_evolution(
            instance, population_size=10, max_iterations=20, seed=42,
        )
        assert sorted(sol.permutation) == list(range(10))

    def test_makespan_is_correct(self):
        instance = FlowShopInstance.random(n=15, m=4, seed=7)
        sol = differential_evolution(
            instance, population_size=10, max_iterations=30, seed=7,
        )
        assert sol.makespan == compute_makespan(instance, sol.permutation)

    def test_no_worse_than_neh(self):
        instance = FlowShopInstance.random(n=20, m=5, seed=42)
        sol_neh = neh(instance)
        sol_de = differential_evolution(
            instance, population_size=20, max_iterations=100, seed=42,
        )
        assert sol_de.makespan <= sol_neh.makespan

    def test_improves_with_more_iterations(self):
        instance = FlowShopInstance.random(n=20, m=5, seed=42)
        sol_short = differential_evolution(
            instance, population_size=15, max_iterations=10, seed=42,
        )
        sol_long = differential_evolution(
            instance, population_size=15, max_iterations=200, seed=42,
        )
        assert sol_long.makespan <= sol_short.makespan

    def test_deterministic_with_seed(self):
        instance = FlowShopInstance.random(n=15, m=4, seed=42)
        sol_a = differential_evolution(
            instance, population_size=10, max_iterations=30, seed=123,
        )
        sol_b = differential_evolution(
            instance, population_size=10, max_iterations=30, seed=123,
        )
        assert sol_a.makespan == sol_b.makespan
        assert sol_a.permutation == sol_b.permutation

    def test_time_limit(self):
        instance = FlowShopInstance.random(n=20, m=5, seed=42)
        sol = differential_evolution(
            instance, time_limit=1.0, seed=42,
        )
        assert sorted(sol.permutation) == list(range(20))
        assert sol.makespan == compute_makespan(instance, sol.permutation)

    def test_without_local_search(self):
        instance = FlowShopInstance.random(n=15, m=4, seed=42)
        sol = differential_evolution(
            instance, population_size=10, max_iterations=30,
            local_search=False, seed=42,
        )
        assert sorted(sol.permutation) == list(range(15))

    def test_mutation_factor(self):
        """Different F values should all produce valid results."""
        instance = FlowShopInstance.random(n=10, m=3, seed=42)
        for f_val in [0.3, 0.7, 1.0]:
            sol = differential_evolution(
                instance, population_size=10, max_iterations=10,
                F=f_val, seed=42,
            )
            assert sorted(sol.permutation) == list(range(10))


# ──────────────────────────────────────────────
# Scatter Search Tests
# ──────────────────────────────────────────────

class TestScatterSearch:
    """Verify Scatter Search correctness and improvement."""

    def test_returns_valid_permutation(self):
        instance = FlowShopInstance.random(n=10, m=3, seed=42)
        sol = scatter_search(
            instance, refset_size=6, pop_size=15, max_iterations=5, seed=42,
        )
        assert sorted(sol.permutation) == list(range(10))

    def test_makespan_is_correct(self):
        instance = FlowShopInstance.random(n=15, m=4, seed=7)
        sol = scatter_search(
            instance, refset_size=6, pop_size=15, max_iterations=10, seed=7,
        )
        assert sol.makespan == compute_makespan(instance, sol.permutation)

    def test_no_worse_than_neh(self):
        instance = FlowShopInstance.random(n=20, m=5, seed=42)
        sol_neh = neh(instance)
        sol_ss = scatter_search(
            instance, refset_size=8, pop_size=20, max_iterations=20, seed=42,
        )
        assert sol_ss.makespan <= sol_neh.makespan

    def test_deterministic_with_seed(self):
        instance = FlowShopInstance.random(n=15, m=4, seed=42)
        sol_a = scatter_search(
            instance, refset_size=6, pop_size=15, max_iterations=5, seed=123,
        )
        sol_b = scatter_search(
            instance, refset_size=6, pop_size=15, max_iterations=5, seed=123,
        )
        assert sol_a.makespan == sol_b.makespan
        assert sol_a.permutation == sol_b.permutation

    def test_time_limit(self):
        instance = FlowShopInstance.random(n=20, m=5, seed=42)
        sol = scatter_search(
            instance, time_limit=2.0, seed=42,
        )
        assert sorted(sol.permutation) == list(range(20))
        assert sol.makespan == compute_makespan(instance, sol.permutation)

    def test_small_instance(self):
        instance = FlowShopInstance.random(n=3, m=2, seed=42)
        sol = scatter_search(
            instance, refset_size=4, pop_size=8, max_iterations=3, seed=42,
        )
        assert sorted(sol.permutation) == list(range(3))


# ──────────────────────────────────────────────
# RA Heuristic Tests
# ──────────────────────────────────────────────

class TestRAHeuristic:
    """Verify RA multi-ordering constructive heuristic."""

    def test_returns_valid_permutation(self):
        instance = FlowShopInstance.random(n=10, m=3, seed=42)
        sol = ra_heuristic(instance)
        assert sorted(sol.permutation) == list(range(10))

    def test_makespan_is_correct(self):
        instance = FlowShopInstance.random(n=15, m=4, seed=7)
        sol = ra_heuristic(instance)
        assert sol.makespan == compute_makespan(instance, sol.permutation)

    def test_no_worse_than_neh_usually(self):
        """RA should match or beat NEH on many instances."""
        better_or_equal = 0
        n_tests = 10
        for s in range(n_tests):
            instance = FlowShopInstance.random(n=20, m=5, seed=s)
            sol_neh = neh(instance)
            sol_ra = ra_heuristic(instance)
            if sol_ra.makespan <= sol_neh.makespan:
                better_or_equal += 1
        # RA should match or beat NEH on at least some instances
        assert better_or_equal >= 2

    def test_single_job(self):
        instance = FlowShopInstance.random(n=1, m=3, seed=42)
        sol = ra_heuristic(instance)
        assert sol.permutation == [0]

    def test_two_machines(self):
        instance = FlowShopInstance.random(n=8, m=2, seed=42)
        sol = ra_heuristic(instance)
        assert sorted(sol.permutation) == list(range(8))
        assert sol.makespan == compute_makespan(instance, sol.permutation)

    def test_deterministic(self):
        instance = FlowShopInstance.random(n=15, m=4, seed=42)
        sol_a = ra_heuristic(instance)
        sol_b = ra_heuristic(instance)
        assert sol_a.makespan == sol_b.makespan
        assert sol_a.permutation == sol_b.permutation

    def test_ten_jobs(self):
        instance = FlowShopInstance.random(n=10, m=5, seed=99)
        sol = ra_heuristic(instance)
        assert len(sol.permutation) == 10
        assert sol.makespan > 0
