"""
Test Suite — New PFSP Algorithms and Flow Shop Variants

Tests cover:
1. Dannenbring's Rapid Access heuristic
2. Simulated Annealing metaheuristic
3. Genetic Algorithm metaheuristic
4. No-Wait Flow Shop variant (instance, heuristics, metaheuristics)
5. Blocking Flow Shop variant (instance, heuristics, metaheuristics)
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
from heuristics.dannenbring import dannenbring_ra
from heuristics.neh import neh
from heuristics.palmers_slope import palmers_slope
from metaheuristics.simulated_annealing import simulated_annealing
from metaheuristics.genetic_algorithm import genetic_algorithm


def _load_module(name: str, filepath: str):
    """Load a Python module from an explicit file path."""
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # Register so dataclass decorator can find it
    spec.loader.exec_module(mod)
    return mod


# Pre-load variant modules using explicit file paths to avoid import collisions
_nw_dir = os.path.join(_flow_shop_dir, "variants", "no_wait")
_blk_dir = os.path.join(_flow_shop_dir, "variants", "blocking")

nw_instance_mod = _load_module("nw_instance", os.path.join(_nw_dir, "instance.py"))
nw_heuristics_mod = _load_module("nw_heuristics", os.path.join(_nw_dir, "heuristics.py"))
nw_meta_mod = _load_module("nw_metaheuristics", os.path.join(_nw_dir, "metaheuristics.py"))

blk_instance_mod = _load_module("blk_instance", os.path.join(_blk_dir, "instance.py"))
blk_heuristics_mod = _load_module("blk_heuristics", os.path.join(_blk_dir, "heuristics.py"))
blk_meta_mod = _load_module("blk_metaheuristics", os.path.join(_blk_dir, "metaheuristics.py"))


# ──────────────────────────────────────────────
# Dannenbring's Rapid Access Tests
# ──────────────────────────────────────────────

class TestDannenbringRA:
    """Verify Dannenbring's Rapid Access heuristic correctness."""

    def test_returns_valid_permutation(self):
        instance = FlowShopInstance.random(n=10, m=4, seed=42)
        sol = dannenbring_ra(instance)
        assert sorted(sol.permutation) == list(range(10))

    def test_makespan_is_correct(self):
        """Reported makespan should match recomputed value."""
        instance = FlowShopInstance.random(n=15, m=5, seed=7)
        sol = dannenbring_ra(instance)
        assert sol.makespan == compute_makespan(instance, sol.permutation)

    def test_beats_arbitrary_order(self):
        """Should beat an arbitrary fixed order."""
        instance = FlowShopInstance.random(n=20, m=5, seed=42)
        sol = dannenbring_ra(instance)
        identity_ms = compute_makespan(instance, list(range(20)))
        assert sol.makespan <= identity_ms

    def test_small_instance_2_machines(self):
        """On 2-machine instances, RA uses Johnson's Rule directly."""
        from exact.johnsons_rule import johnsons_rule
        instance = FlowShopInstance.random(n=6, m=2, seed=42)
        sol_ra = dannenbring_ra(instance)
        sol_johnson = johnsons_rule(instance)
        # RA constructs a single virtual 2-machine problem;
        # on m=2 it should produce a valid solution
        assert sorted(sol_ra.permutation) == list(range(6))
        # May not match Johnson exactly (different virtual weights)
        # but should be reasonable
        assert sol_ra.makespan > 0

    def test_quality_reasonable(self):
        """Dannenbring should produce reasonable solutions."""
        instance = FlowShopInstance.random(n=20, m=5, seed=42)
        sol_ra = dannenbring_ra(instance)
        sol_neh = neh(instance)
        # RA should not be dramatically worse than NEH (within 20%)
        assert sol_ra.makespan <= sol_neh.makespan * 1.20


# ──────────────────────────────────────────────
# Simulated Annealing Tests
# ──────────────────────────────────────────────

class TestSimulatedAnnealing:
    """Verify Simulated Annealing correctness and improvement."""

    def test_returns_valid_permutation(self):
        instance = FlowShopInstance.random(n=10, m=3, seed=42)
        sol = simulated_annealing(instance, max_iterations=500, seed=42)
        assert sorted(sol.permutation) == list(range(10))

    def test_makespan_is_correct(self):
        instance = FlowShopInstance.random(n=15, m=4, seed=7)
        sol = simulated_annealing(instance, max_iterations=1000, seed=7)
        assert sol.makespan == compute_makespan(instance, sol.permutation)

    def test_no_worse_than_neh(self):
        """SA starts from NEH, should never be worse."""
        instance = FlowShopInstance.random(n=20, m=5, seed=42)
        sol_neh = neh(instance)
        sol_sa = simulated_annealing(instance, max_iterations=5000, seed=42)
        assert sol_sa.makespan <= sol_neh.makespan

    def test_improves_with_more_iterations(self):
        """More iterations should give same or better results."""
        instance = FlowShopInstance.random(n=20, m=5, seed=42)
        sol_short = simulated_annealing(instance, max_iterations=100, seed=42)
        sol_long = simulated_annealing(instance, max_iterations=10000, seed=42)
        assert sol_long.makespan <= sol_short.makespan

    def test_deterministic_with_seed(self):
        """Same seed should give identical results."""
        instance = FlowShopInstance.random(n=15, m=4, seed=42)
        sol_a = simulated_annealing(instance, max_iterations=1000, seed=123)
        sol_b = simulated_annealing(instance, max_iterations=1000, seed=123)
        assert sol_a.makespan == sol_b.makespan
        assert sol_a.permutation == sol_b.permutation

    def test_time_limited_run(self):
        """Time-limited run should terminate and return valid solution."""
        instance = FlowShopInstance.random(n=30, m=5, seed=42)
        sol = simulated_annealing(instance, time_limit=0.5, seed=42)
        assert sorted(sol.permutation) == list(range(30))
        assert sol.makespan == compute_makespan(instance, sol.permutation)


# ──────────────────────────────────────────────
# Genetic Algorithm Tests
# ──────────────────────────────────────────────

class TestGeneticAlgorithm:
    """Verify Genetic Algorithm correctness and improvement."""

    def test_returns_valid_permutation(self):
        instance = FlowShopInstance.random(n=10, m=3, seed=42)
        sol = genetic_algorithm(instance, max_generations=100, seed=42)
        assert sorted(sol.permutation) == list(range(10))

    def test_makespan_is_correct(self):
        instance = FlowShopInstance.random(n=15, m=4, seed=7)
        sol = genetic_algorithm(instance, max_generations=200, seed=7)
        assert sol.makespan == compute_makespan(instance, sol.permutation)

    def test_no_worse_than_neh(self):
        """GA population includes NEH, should never be worse."""
        instance = FlowShopInstance.random(n=20, m=5, seed=42)
        sol_neh = neh(instance)
        sol_ga = genetic_algorithm(instance, max_generations=300, seed=42)
        assert sol_ga.makespan <= sol_neh.makespan

    def test_deterministic_with_seed(self):
        """Same seed should give identical results."""
        instance = FlowShopInstance.random(n=10, m=3, seed=42)
        sol_a = genetic_algorithm(instance, max_generations=100, seed=123)
        sol_b = genetic_algorithm(instance, max_generations=100, seed=123)
        assert sol_a.makespan == sol_b.makespan

    def test_memetic_variant(self):
        """Memetic GA (with local search) should produce valid solutions."""
        instance = FlowShopInstance.random(n=8, m=3, seed=42)
        sol = genetic_algorithm(
            instance, max_generations=20, use_local_search=True, seed=42
        )
        assert sorted(sol.permutation) == list(range(8))
        assert sol.makespan == compute_makespan(instance, sol.permutation)

    def test_time_limited_run(self):
        """Time-limited run should terminate and return valid solution."""
        instance = FlowShopInstance.random(n=30, m=5, seed=42)
        sol = genetic_algorithm(instance, time_limit=0.5, seed=42)
        assert sorted(sol.permutation) == list(range(30))


# ──────────────────────────────────────────────
# No-Wait Flow Shop Tests
# ──────────────────────────────────────────────

class TestNoWaitFlowShop:
    """Verify No-Wait Flow Shop data structures and algorithms."""

    @pytest.fixture
    def small_nw_instance(self):
        return nw_instance_mod.NoWaitFlowShopInstance(
            n=4, m=3,
            processing_times=np.array([
                [3, 5, 2, 7],
                [4, 2, 6, 1],
                [2, 3, 4, 5],
            ])
        )

    def test_delay_matrix_shape(self, small_nw_instance):
        D = nw_instance_mod.compute_delay_matrix(small_nw_instance)
        assert D.shape == (4, 4)
        for i in range(4):
            assert D[i, i] == 0

    def test_delay_is_positive(self, small_nw_instance):
        D = nw_instance_mod.compute_delay_matrix(small_nw_instance)
        for i in range(4):
            for j in range(4):
                if i != j:
                    assert D[i, j] > 0

    def test_delay_asymmetric(self, small_nw_instance):
        """Delay matrix should generally be asymmetric."""
        D = nw_instance_mod.compute_delay_matrix(small_nw_instance)
        asymmetric = any(
            D[i, j] != D[j, i]
            for i in range(4) for j in range(i + 1, 4)
        )
        assert asymmetric

    def test_makespan_nw_single_job(self):
        instance = nw_instance_mod.NoWaitFlowShopInstance(
            n=1, m=3,
            processing_times=np.array([[3], [5], [2]])
        )
        assert nw_instance_mod.compute_makespan_nw(instance, [0]) == 10

    def test_makespan_nw_consistent(self, small_nw_instance):
        """Makespan should be consistent with delay computation."""
        D = nw_instance_mod.compute_delay_matrix(small_nw_instance)
        perm = [0, 1, 2, 3]
        ms = nw_instance_mod.compute_makespan_nw(small_nw_instance, perm, D)
        ms_no_cache = nw_instance_mod.compute_makespan_nw(small_nw_instance, perm)
        assert ms == ms_no_cache

    def test_nn_returns_valid_permutation(self):
        instance = nw_instance_mod.NoWaitFlowShopInstance.random(n=10, m=4, seed=42)
        sol = nw_heuristics_mod.nearest_neighbor_nw(instance)
        assert sorted(sol.permutation) == list(range(10))

    def test_neh_nw_returns_valid_permutation(self):
        instance = nw_instance_mod.NoWaitFlowShopInstance.random(n=10, m=4, seed=42)
        sol = nw_heuristics_mod.neh_no_wait(instance)
        assert sorted(sol.permutation) == list(range(10))

    def test_neh_nw_makespan_correct(self):
        instance = nw_instance_mod.NoWaitFlowShopInstance.random(n=10, m=4, seed=42)
        sol = nw_heuristics_mod.neh_no_wait(instance)
        assert sol.makespan == nw_instance_mod.compute_makespan_nw(
            instance, sol.permutation
        )

    def test_gr_returns_valid_permutation(self):
        instance = nw_instance_mod.NoWaitFlowShopInstance.random(n=10, m=4, seed=42)
        sol = nw_heuristics_mod.gangadharan_rajendran(instance)
        assert sorted(sol.permutation) == list(range(10))

    def test_neh_nw_beats_random(self):
        """NEH-NW should beat most random permutations."""
        rng = np.random.default_rng(42)
        instance = nw_instance_mod.NoWaitFlowShopInstance.random(n=15, m=4, seed=42)
        sol = nw_heuristics_mod.neh_no_wait(instance)
        random_better = 0
        for _ in range(50):
            random_perm = list(rng.permutation(15))
            random_ms = nw_instance_mod.compute_makespan_nw(instance, random_perm)
            if random_ms < sol.makespan:
                random_better += 1
        assert random_better <= 5

    def test_ig_nw_no_worse_than_neh(self):
        """IG-NW should not be worse than NEH-NW."""
        instance = nw_instance_mod.NoWaitFlowShopInstance.random(n=15, m=4, seed=42)
        sol_neh = nw_heuristics_mod.neh_no_wait(instance)
        sol_ig = nw_meta_mod.iterated_greedy_nw(instance, max_iterations=100, seed=42)
        assert sol_ig.makespan <= sol_neh.makespan


# ──────────────────────────────────────────────
# Blocking Flow Shop Tests
# ──────────────────────────────────────────────

class TestBlockingFlowShop:
    """Verify Blocking Flow Shop data structures and algorithms."""

    @pytest.fixture
    def small_blk_instance(self):
        return blk_instance_mod.BlockingFlowShopInstance(
            n=4, m=3,
            processing_times=np.array([
                [3, 5, 2, 7],
                [4, 2, 6, 1],
                [2, 3, 4, 5],
            ])
        )

    def test_blocking_makespan_single_job(self):
        """Single job: blocking makespan equals standard makespan."""
        instance = blk_instance_mod.BlockingFlowShopInstance(
            n=1, m=3,
            processing_times=np.array([[3], [5], [2]])
        )
        assert blk_instance_mod.compute_makespan_blocking(instance, [0]) == 10

    def test_blocking_makespan_geq_standard(self, small_blk_instance):
        """Blocking makespan >= standard PFSP makespan for any permutation."""
        standard = FlowShopInstance(
            n=small_blk_instance.n,
            m=small_blk_instance.m,
            processing_times=small_blk_instance.processing_times,
        )
        from itertools import permutations
        for perm in permutations(range(4)):
            perm_list = list(perm)
            ms_std = compute_makespan(standard, perm_list)
            ms_blk = blk_instance_mod.compute_makespan_blocking(
                small_blk_instance, perm_list
            )
            assert ms_blk >= ms_std, (
                f"Blocking makespan {ms_blk} < standard {ms_std} "
                f"for permutation {perm_list}"
            )

    def test_neh_blocking_valid_permutation(self):
        instance = blk_instance_mod.BlockingFlowShopInstance.random(n=10, m=4, seed=42)
        sol = blk_heuristics_mod.neh_blocking(instance)
        assert sorted(sol.permutation) == list(range(10))

    def test_neh_blocking_makespan_correct(self):
        instance = blk_instance_mod.BlockingFlowShopInstance.random(n=10, m=4, seed=42)
        sol = blk_heuristics_mod.neh_blocking(instance)
        assert sol.makespan == blk_instance_mod.compute_makespan_blocking(
            instance, sol.permutation
        )

    def test_profile_fitting_valid(self):
        instance = blk_instance_mod.BlockingFlowShopInstance.random(n=10, m=4, seed=42)
        sol = blk_heuristics_mod.profile_fitting_blocking(instance)
        assert sorted(sol.permutation) == list(range(10))

    def test_neh_blocking_beats_random(self):
        """NEH-B should beat most random permutations."""
        rng = np.random.default_rng(42)
        instance = blk_instance_mod.BlockingFlowShopInstance.random(n=15, m=4, seed=42)
        sol = blk_heuristics_mod.neh_blocking(instance)
        random_better = 0
        for _ in range(50):
            random_perm = list(rng.permutation(15))
            random_ms = blk_instance_mod.compute_makespan_blocking(
                instance, random_perm
            )
            if random_ms < sol.makespan:
                random_better += 1
        assert random_better <= 5

    def test_ig_blocking_no_worse_than_neh(self):
        """IG-B should not be worse than NEH-B."""
        instance = blk_instance_mod.BlockingFlowShopInstance.random(n=15, m=4, seed=42)
        sol_neh = blk_heuristics_mod.neh_blocking(instance)
        sol_ig = blk_meta_mod.iterated_greedy_blocking(
            instance, max_iterations=100, seed=42
        )
        assert sol_ig.makespan <= sol_neh.makespan

    def test_blocking_geq_standard_2_machines(self):
        """With 2 machines, blocking makespan >= standard makespan."""
        instance_b = blk_instance_mod.BlockingFlowShopInstance(
            n=3, m=2,
            processing_times=np.array([[3, 5, 2], [4, 2, 6]])
        )
        instance_s = FlowShopInstance(
            n=3, m=2,
            processing_times=np.array([[3, 5, 2], [4, 2, 6]])
        )
        for perm in [[0, 1, 2], [2, 0, 1], [1, 2, 0]]:
            ms_b = blk_instance_mod.compute_makespan_blocking(instance_b, perm)
            ms_s = compute_makespan(instance_s, perm)
            assert ms_b >= ms_s


# ──────────────────────────────────────────────
# Cross-Variant Comparison Tests
# ──────────────────────────────────────────────

class TestCrossVariantComparison:
    """Compare behavior across flow shop variants on the same instance."""

    def test_all_new_pfsp_algorithms_on_small_instance(self):
        """All new PFSP algorithms should return valid solutions."""
        instance = FlowShopInstance.random(n=5, m=3, seed=42)

        algorithms = {
            "Dannenbring": dannenbring_ra(instance),
            "SA": simulated_annealing(instance, max_iterations=500, seed=42),
            "GA": genetic_algorithm(instance, max_generations=100, seed=42),
        }

        for name, sol in algorithms.items():
            assert sorted(sol.permutation) == list(range(5)), (
                f"{name} returned invalid permutation"
            )
            assert sol.makespan == compute_makespan(instance, sol.permutation), (
                f"{name} makespan mismatch"
            )

    def test_variant_instances_same_data(self):
        """All variant instances should accept the same processing times."""
        p = np.array([[3, 5], [4, 2], [2, 3]])
        nw = nw_instance_mod.NoWaitFlowShopInstance(n=2, m=3, processing_times=p.copy())
        blk = blk_instance_mod.BlockingFlowShopInstance(n=2, m=3, processing_times=p.copy())
        std = FlowShopInstance(n=2, m=3, processing_times=p.copy())

        assert nw.n == blk.n == std.n == 2
        assert nw.m == blk.m == std.m == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
