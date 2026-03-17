"""
Test Suite — Tabu Search, Ant Colony Optimization, and SDST Variant

Tests cover:
1. Tabu Search metaheuristic for standard PFSP
2. Ant Colony Optimization metaheuristic for standard PFSP
3. Sequence-Dependent Setup Times (SDST) variant (instance, heuristics, metaheuristics)
"""

import sys
import os
import importlib.util
import numpy as np
import pytest

# Add parent directories to path
_flow_shop_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _flow_shop_dir)

from instance import (
    FlowShopInstance,
    compute_makespan,
)
from heuristics.neh import neh
from metaheuristics.tabu_search import tabu_search
from metaheuristics.ant_colony import ant_colony_optimization


def _load_module(name: str, filepath: str):
    """Load a Python module from an explicit file path."""
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Pre-load SDST variant modules using explicit file paths
_sdst_dir = os.path.join(_flow_shop_dir, "variants", "setup_times")

sdst_instance_mod = _load_module(
    "sdst_instance_test", os.path.join(_sdst_dir, "instance.py")
)
sdst_heuristics_mod = _load_module(
    "sdst_heuristics_test", os.path.join(_sdst_dir, "heuristics.py")
)
sdst_meta_mod = _load_module(
    "sdst_metaheuristics_test", os.path.join(_sdst_dir, "metaheuristics.py")
)


# ──────────────────────────────────────────────
# Tabu Search Tests
# ──────────────────────────────────────────────

class TestTabuSearch:
    """Verify Tabu Search correctness and improvement."""

    def test_returns_valid_permutation(self):
        instance = FlowShopInstance.random(n=10, m=3, seed=42)
        sol = tabu_search(instance, max_iterations=50, seed=42)
        assert sorted(sol.permutation) == list(range(10))

    def test_makespan_is_correct(self):
        """Reported makespan should match recomputed value."""
        instance = FlowShopInstance.random(n=15, m=4, seed=7)
        sol = tabu_search(instance, max_iterations=100, seed=7)
        assert sol.makespan == compute_makespan(instance, sol.permutation)

    def test_no_worse_than_neh(self):
        """TS starts from NEH, should never be worse."""
        instance = FlowShopInstance.random(n=15, m=4, seed=42)
        sol_neh = neh(instance)
        sol_ts = tabu_search(instance, max_iterations=200, seed=42)
        assert sol_ts.makespan <= sol_neh.makespan

    def test_deterministic_with_seed(self):
        """Same seed should give identical results."""
        instance = FlowShopInstance.random(n=10, m=3, seed=42)
        sol_a = tabu_search(instance, max_iterations=100, seed=123)
        sol_b = tabu_search(instance, max_iterations=100, seed=123)
        assert sol_a.makespan == sol_b.makespan
        assert sol_a.permutation == sol_b.permutation

    def test_swap_neighborhood(self):
        """Swap neighborhood should also produce valid solutions."""
        instance = FlowShopInstance.random(n=10, m=3, seed=42)
        sol = tabu_search(
            instance, max_iterations=50, neighborhood="swap", seed=42
        )
        assert sorted(sol.permutation) == list(range(10))
        assert sol.makespan == compute_makespan(instance, sol.permutation)

    def test_time_limited_run(self):
        """Time-limited run should terminate and return valid solution."""
        instance = FlowShopInstance.random(n=20, m=5, seed=42)
        sol = tabu_search(instance, time_limit=0.5, seed=42)
        assert sorted(sol.permutation) == list(range(20))
        assert sol.makespan == compute_makespan(instance, sol.permutation)

    def test_single_job(self):
        """Edge case: single job."""
        instance = FlowShopInstance.random(n=1, m=3, seed=42)
        sol = tabu_search(instance, max_iterations=10, seed=42)
        assert sol.permutation == [0]
        assert sol.makespan == compute_makespan(instance, [0])

    def test_two_jobs(self):
        """Edge case: two jobs."""
        instance = FlowShopInstance.random(n=2, m=3, seed=42)
        sol = tabu_search(instance, max_iterations=20, seed=42)
        assert sorted(sol.permutation) == [0, 1]
        assert sol.makespan == compute_makespan(instance, sol.permutation)


# ──────────────────────────────────────────────
# Ant Colony Optimization Tests
# ──────────────────────────────────────────────

class TestAntColonyOptimization:
    """Verify Ant Colony Optimization correctness and improvement."""

    def test_returns_valid_permutation(self):
        instance = FlowShopInstance.random(n=10, m=3, seed=42)
        sol = ant_colony_optimization(
            instance, n_ants=5, max_iterations=20, seed=42
        )
        assert sorted(sol.permutation) == list(range(10))

    def test_makespan_is_correct(self):
        """Reported makespan should match recomputed value."""
        instance = FlowShopInstance.random(n=15, m=4, seed=7)
        sol = ant_colony_optimization(
            instance, n_ants=5, max_iterations=30, seed=7
        )
        assert sol.makespan == compute_makespan(instance, sol.permutation)

    def test_no_worse_than_neh(self):
        """ACO initializes pheromones from NEH, should match or beat NEH."""
        instance = FlowShopInstance.random(n=15, m=4, seed=42)
        sol_neh = neh(instance)
        sol_aco = ant_colony_optimization(
            instance, n_ants=10, max_iterations=50, seed=42
        )
        assert sol_aco.makespan <= sol_neh.makespan

    def test_deterministic_with_seed(self):
        """Same seed should give identical results."""
        instance = FlowShopInstance.random(n=10, m=3, seed=42)
        sol_a = ant_colony_optimization(
            instance, n_ants=5, max_iterations=20, seed=123
        )
        sol_b = ant_colony_optimization(
            instance, n_ants=5, max_iterations=20, seed=123
        )
        assert sol_a.makespan == sol_b.makespan

    def test_with_local_search(self):
        """ACO with local search should produce valid solutions."""
        instance = FlowShopInstance.random(n=8, m=3, seed=42)
        sol = ant_colony_optimization(
            instance, n_ants=3, max_iterations=10,
            use_local_search=True, seed=42
        )
        assert sorted(sol.permutation) == list(range(8))
        assert sol.makespan == compute_makespan(instance, sol.permutation)

    def test_time_limited_run(self):
        """Time-limited run should terminate and return valid solution."""
        instance = FlowShopInstance.random(n=20, m=5, seed=42)
        sol = ant_colony_optimization(instance, time_limit=0.5, seed=42)
        assert sorted(sol.permutation) == list(range(20))
        assert sol.makespan == compute_makespan(instance, sol.permutation)

    def test_single_job(self):
        """Edge case: single job."""
        instance = FlowShopInstance.random(n=1, m=3, seed=42)
        sol = ant_colony_optimization(
            instance, n_ants=3, max_iterations=5, seed=42
        )
        assert sol.permutation == [0]

    def test_different_parameters(self):
        """Different alpha/beta/rho should still produce valid solutions."""
        instance = FlowShopInstance.random(n=10, m=3, seed=42)
        sol = ant_colony_optimization(
            instance, n_ants=5, alpha=2.0, beta=3.0, rho=0.5,
            max_iterations=20, seed=42
        )
        assert sorted(sol.permutation) == list(range(10))
        assert sol.makespan == compute_makespan(instance, sol.permutation)


# ──────────────────────────────────────────────
# SDST Flow Shop Instance Tests
# ──────────────────────────────────────────────

class TestSDSTInstance:
    """Verify SDST Flow Shop data structures and makespan computation."""

    @pytest.fixture
    def small_sdst_instance(self):
        rng = np.random.default_rng(42)
        n, m = 4, 3
        p = rng.integers(1, 50, size=(m, n))
        s = rng.integers(1, 25, size=(m, n + 1, n))
        for i in range(m):
            for j in range(n):
                s[i, j, j] = 0
        return sdst_instance_mod.SDSTFlowShopInstance(
            n=n, m=m, processing_times=p, setup_times=s
        )

    def test_instance_creation(self, small_sdst_instance):
        """Instance should be created with correct dimensions."""
        assert small_sdst_instance.n == 4
        assert small_sdst_instance.m == 3
        assert small_sdst_instance.processing_times.shape == (3, 4)
        assert small_sdst_instance.setup_times.shape == (3, 5, 4)

    def test_random_instance(self):
        """Random factory should create valid instances."""
        instance = sdst_instance_mod.SDSTFlowShopInstance.random(
            n=10, m=4, seed=42
        )
        assert instance.n == 10
        assert instance.m == 4
        assert instance.processing_times.shape == (4, 10)
        assert instance.setup_times.shape == (4, 11, 10)
        # Diagonal should be zero
        for i in range(4):
            for j in range(10):
                assert instance.setup_times[i, j, j] == 0

    def test_makespan_single_job(self):
        """Single job: makespan = sum of processing + initial setups."""
        p = np.array([[5], [3], [7]])
        s = np.zeros((3, 2, 1), dtype=int)
        s[:, 1, 0] = [2, 1, 3]  # initial setup (index n=1)
        instance = sdst_instance_mod.SDSTFlowShopInstance(
            n=1, m=3, processing_times=p, setup_times=s
        )
        ms = sdst_instance_mod.compute_makespan_sdst(instance, [0])
        # Machine 0: s[0,1,0] + p[0,0] = 2 + 5 = 7
        # Machine 1: max(7, 1) + 3 = 7 + 3 = 10
        # Machine 2: max(10, 3) + 7 = 10 + 7 = 17
        assert ms == 17

    def test_makespan_positive(self, small_sdst_instance):
        """Makespan should always be positive."""
        ms = sdst_instance_mod.compute_makespan_sdst(
            small_sdst_instance, [0, 1, 2, 3]
        )
        assert ms > 0

    def test_makespan_different_permutations(self, small_sdst_instance):
        """Different permutations should generally give different makespans."""
        perms = [[0, 1, 2, 3], [3, 2, 1, 0], [1, 0, 3, 2], [2, 3, 0, 1]]
        makespans = set()
        for perm in perms:
            ms = sdst_instance_mod.compute_makespan_sdst(
                small_sdst_instance, perm
            )
            makespans.add(ms)
        # At least 2 different makespans expected (very unlikely all are equal)
        assert len(makespans) >= 2

    def test_setup_times_affect_makespan(self):
        """Instance with larger setup times should have larger makespan."""
        p = np.array([[10, 10], [10, 10]])
        s_small = np.ones((2, 3, 2), dtype=int)
        s_large = np.full((2, 3, 2), 50, dtype=int)
        for s in [s_small, s_large]:
            for i in range(2):
                for j in range(2):
                    s[i, j, j] = 0

        inst_small = sdst_instance_mod.SDSTFlowShopInstance(
            n=2, m=2, processing_times=p.copy(), setup_times=s_small
        )
        inst_large = sdst_instance_mod.SDSTFlowShopInstance(
            n=2, m=2, processing_times=p.copy(), setup_times=s_large
        )

        ms_small = sdst_instance_mod.compute_makespan_sdst(inst_small, [0, 1])
        ms_large = sdst_instance_mod.compute_makespan_sdst(inst_large, [0, 1])
        assert ms_large > ms_small


# ──────────────────────────────────────────────
# SDST Heuristics Tests
# ──────────────────────────────────────────────

class TestSDSTHeuristics:
    """Verify SDST heuristics correctness."""

    def test_neh_sdst_valid_permutation(self):
        instance = sdst_instance_mod.SDSTFlowShopInstance.random(
            n=10, m=4, seed=42
        )
        sol = sdst_heuristics_mod.neh_sdst(instance)
        assert sorted(sol.permutation) == list(range(10))

    def test_neh_sdst_makespan_correct(self):
        instance = sdst_instance_mod.SDSTFlowShopInstance.random(
            n=10, m=4, seed=42
        )
        sol = sdst_heuristics_mod.neh_sdst(instance)
        assert sol.makespan == sdst_instance_mod.compute_makespan_sdst(
            instance, sol.permutation
        )

    def test_neh_sdst_beats_random(self):
        """NEH-SDST should beat most random permutations."""
        rng = np.random.default_rng(42)
        instance = sdst_instance_mod.SDSTFlowShopInstance.random(
            n=15, m=4, seed=42
        )
        sol = sdst_heuristics_mod.neh_sdst(instance)
        random_better = 0
        for _ in range(50):
            random_perm = list(rng.permutation(15))
            random_ms = sdst_instance_mod.compute_makespan_sdst(
                instance, random_perm
            )
            if random_ms < sol.makespan:
                random_better += 1
        assert random_better <= 5

    def test_grasp_sdst_valid_permutation(self):
        instance = sdst_instance_mod.SDSTFlowShopInstance.random(
            n=10, m=4, seed=42
        )
        sol = sdst_heuristics_mod.grasp_sdst(
            instance, max_constructions=3, seed=42
        )
        assert sorted(sol.permutation) == list(range(10))

    def test_grasp_sdst_makespan_correct(self):
        instance = sdst_instance_mod.SDSTFlowShopInstance.random(
            n=10, m=4, seed=42
        )
        sol = sdst_heuristics_mod.grasp_sdst(
            instance, max_constructions=3, seed=42
        )
        assert sol.makespan == sdst_instance_mod.compute_makespan_sdst(
            instance, sol.permutation
        )

    def test_grasp_deterministic_with_seed(self):
        """Same seed should give same results."""
        instance = sdst_instance_mod.SDSTFlowShopInstance.random(
            n=8, m=3, seed=42
        )
        sol_a = sdst_heuristics_mod.grasp_sdst(
            instance, max_constructions=3, seed=123
        )
        sol_b = sdst_heuristics_mod.grasp_sdst(
            instance, max_constructions=3, seed=123
        )
        assert sol_a.makespan == sol_b.makespan


# ──────────────────────────────────────────────
# SDST Metaheuristics Tests
# ──────────────────────────────────────────────

class TestSDSTMetaheuristics:
    """Verify SDST metaheuristics correctness."""

    def test_ig_sdst_valid_permutation(self):
        instance = sdst_instance_mod.SDSTFlowShopInstance.random(
            n=10, m=4, seed=42
        )
        sol = sdst_meta_mod.iterated_greedy_sdst(
            instance, max_iterations=50, seed=42
        )
        assert sorted(sol.permutation) == list(range(10))

    def test_ig_sdst_makespan_correct(self):
        instance = sdst_instance_mod.SDSTFlowShopInstance.random(
            n=10, m=4, seed=42
        )
        sol = sdst_meta_mod.iterated_greedy_sdst(
            instance, max_iterations=50, seed=42
        )
        assert sol.makespan == sdst_instance_mod.compute_makespan_sdst(
            instance, sol.permutation
        )

    def test_ig_sdst_no_worse_than_neh(self):
        """IG-SDST should not be worse than NEH-SDST."""
        instance = sdst_instance_mod.SDSTFlowShopInstance.random(
            n=15, m=4, seed=42
        )
        sol_neh = sdst_heuristics_mod.neh_sdst(instance)
        sol_ig = sdst_meta_mod.iterated_greedy_sdst(
            instance, max_iterations=100, seed=42
        )
        assert sol_ig.makespan <= sol_neh.makespan

    def test_ig_sdst_deterministic_with_seed(self):
        """Same seed should give identical results."""
        instance = sdst_instance_mod.SDSTFlowShopInstance.random(
            n=10, m=3, seed=42
        )
        sol_a = sdst_meta_mod.iterated_greedy_sdst(
            instance, max_iterations=50, seed=123
        )
        sol_b = sdst_meta_mod.iterated_greedy_sdst(
            instance, max_iterations=50, seed=123
        )
        assert sol_a.makespan == sol_b.makespan
        assert sol_a.permutation == sol_b.permutation

    def test_ig_sdst_time_limited(self):
        """Time-limited run should terminate and return valid solution."""
        instance = sdst_instance_mod.SDSTFlowShopInstance.random(
            n=15, m=4, seed=42
        )
        sol = sdst_meta_mod.iterated_greedy_sdst(
            instance, time_limit=0.5, seed=42
        )
        assert sorted(sol.permutation) == list(range(15))
        assert sol.makespan == sdst_instance_mod.compute_makespan_sdst(
            instance, sol.permutation
        )


# ──────────────────────────────────────────────
# Cross-Algorithm Comparison Tests
# ──────────────────────────────────────────────

class TestCrossAlgorithmComparison:
    """Compare new algorithms on the same instance."""

    def test_all_new_metaheuristics_valid(self):
        """All new metaheuristics should return valid solutions."""
        instance = FlowShopInstance.random(n=8, m=3, seed=42)

        algorithms = {
            "TS": tabu_search(instance, max_iterations=50, seed=42),
            "ACO": ant_colony_optimization(
                instance, n_ants=5, max_iterations=20, seed=42
            ),
        }

        for name, sol in algorithms.items():
            assert sorted(sol.permutation) == list(range(8)), (
                f"{name} returned invalid permutation"
            )
            assert sol.makespan == compute_makespan(instance, sol.permutation), (
                f"{name} makespan mismatch"
            )

    def test_sdst_all_algorithms_valid(self):
        """All SDST algorithms should return valid solutions."""
        instance = sdst_instance_mod.SDSTFlowShopInstance.random(
            n=8, m=3, seed=42
        )

        algorithms = {
            "NEH-SDST": sdst_heuristics_mod.neh_sdst(instance),
            "GRASP-SDST": sdst_heuristics_mod.grasp_sdst(
                instance, max_constructions=3, seed=42
            ),
            "IG-SDST": sdst_meta_mod.iterated_greedy_sdst(
                instance, max_iterations=30, seed=42
            ),
        }

        for name, sol in algorithms.items():
            assert sorted(sol.permutation) == list(range(8)), (
                f"{name} returned invalid permutation"
            )
            assert sol.makespan == sdst_instance_mod.compute_makespan_sdst(
                instance, sol.permutation
            ), f"{name} makespan mismatch"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
