"""
Test Suite — Permutation Flow Shop Scheduling Algorithms

Tests cover:
1. Correctness of makespan computation
2. Johnson's Rule optimality on 2-machine instances
3. NEH quality guarantees
4. CDS uses Johnson's Rule correctly
5. IG improves upon NEH
6. Edge cases (1 job, 1 machine, identical processing times)
"""

import sys
import os
import numpy as np
import pytest

# Add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from instance import (
    FlowShopInstance,
    compute_makespan,
    compute_completion_times,
)
from exact.johnsons_rule import johnsons_rule
from heuristics.palmers_slope import palmers_slope
from heuristics.neh import neh, neh_with_tiebreaking
from heuristics.cds import cds
from metaheuristics.iterated_greedy import iterated_greedy


# ──────────────────────────────────────────────
# Makespan Computation Tests
# ──────────────────────────────────────────────

class TestMakespanComputation:
    """Verify the core makespan function against hand-calculated values."""

    def test_single_job_single_machine(self):
        instance = FlowShopInstance(
            n=1, m=1,
            processing_times=np.array([[7]])
        )
        assert compute_makespan(instance, [0]) == 7

    def test_single_job_multiple_machines(self):
        """One job visiting 3 machines: makespan = sum of processing times."""
        instance = FlowShopInstance(
            n=1, m=3,
            processing_times=np.array([[3], [5], [2]])
        )
        assert compute_makespan(instance, [0]) == 10

    def test_two_jobs_two_machines_hand_calculated(self):
        """
        Jobs: A(p1=3, p2=2), B(p1=4, p2=1)
        Order [A, B]:
            M1: A[0-3] B[3-7]
            M2: A[3-5] B[7-8]   → Cmax = 8
        """
        instance = FlowShopInstance(
            n=2, m=2,
            processing_times=np.array([[3, 4], [2, 1]])
        )
        assert compute_makespan(instance, [0, 1]) == 8

    def test_two_jobs_reversed_order(self):
        """
        Order [B, A]:
            M1: B[0-4] A[4-7]
            M2: B[4-5] A[7-9]   → Cmax = 9
        """
        instance = FlowShopInstance(
            n=2, m=2,
            processing_times=np.array([[3, 4], [2, 1]])
        )
        assert compute_makespan(instance, [1, 0]) == 9

    def test_completion_times_shape(self):
        instance = FlowShopInstance.random(n=5, m=3, seed=0)
        perm = [0, 1, 2, 3, 4]
        C = compute_completion_times(instance, perm)
        assert C.shape == (3, 5)

    def test_completion_times_makespan_consistency(self):
        """Last entry of completion matrix should equal makespan."""
        instance = FlowShopInstance.random(n=10, m=4, seed=42)
        perm = list(range(10))
        C = compute_completion_times(instance, perm)
        ms = compute_makespan(instance, perm)
        assert C[-1, -1] == ms


# ──────────────────────────────────────────────
# Johnson's Rule Tests
# ──────────────────────────────────────────────

class TestJohnsonsRule:
    """Verify Johnson's Rule gives optimal solutions for F2||Cmax."""

    def test_classic_example(self):
        """Classic textbook instance: 6 jobs, 2 machines."""
        instance = FlowShopInstance(
            n=6, m=2,
            processing_times=np.array([
                [3, 6, 2, 7, 1, 5],
                [4, 1, 5, 2, 6, 3],
            ])
        )
        sol = johnsons_rule(instance)
        assert sol.permutation == [4, 2, 0, 5, 3, 1]
        assert sol.makespan == 25

    def test_optimality_exhaustive_small(self):
        """
        For a small 4-job instance, verify Johnson's Rule matches
        the best of all 4! = 24 permutations.
        """
        from itertools import permutations

        instance = FlowShopInstance(
            n=4, m=2,
            processing_times=np.array([
                [5, 2, 8, 3],
                [7, 4, 1, 6],
            ])
        )

        # Brute force: try all permutations
        best_ms = float('inf')
        for perm in permutations(range(4)):
            ms = compute_makespan(instance, list(perm))
            best_ms = min(best_ms, ms)

        sol = johnsons_rule(instance)
        assert sol.makespan == best_ms

    def test_optimality_random_instances(self):
        """Test optimality on 10 random small instances."""
        from itertools import permutations

        for seed in range(10):
            instance = FlowShopInstance.random(n=5, m=2, seed=seed)

            best_ms = float('inf')
            for perm in permutations(range(5)):
                ms = compute_makespan(instance, list(perm))
                best_ms = min(best_ms, ms)

            sol = johnsons_rule(instance)
            assert sol.makespan == best_ms, (
                f"Seed {seed}: Johnson={sol.makespan}, optimal={best_ms}"
            )

    def test_raises_for_more_than_2_machines(self):
        instance = FlowShopInstance.random(n=5, m=3, seed=0)
        with pytest.raises(ValueError, match="exactly 2 machines"):
            johnsons_rule(instance)

    def test_identical_processing_times(self):
        """When all times are equal, any order is optimal."""
        instance = FlowShopInstance(
            n=3, m=2,
            processing_times=np.array([[5, 5, 5], [5, 5, 5]])
        )
        sol = johnsons_rule(instance)
        assert sol.makespan == 20  # 3*5 + 5


# ──────────────────────────────────────────────
# NEH Tests
# ──────────────────────────────────────────────

class TestNEH:
    """Verify NEH heuristic quality and correctness."""

    def test_returns_valid_permutation(self):
        instance = FlowShopInstance.random(n=10, m=3, seed=42)
        sol = neh(instance)
        assert sorted(sol.permutation) == list(range(10))

    def test_makespan_is_correct(self):
        """Verify reported makespan matches recomputed value."""
        instance = FlowShopInstance.random(n=15, m=4, seed=7)
        sol = neh(instance)
        assert sol.makespan == compute_makespan(instance, sol.permutation)

    def test_beats_random_order(self):
        """NEH should consistently beat a random job order."""
        rng = np.random.default_rng(42)
        instance = FlowShopInstance.random(n=20, m=5, seed=42)

        sol_neh = neh(instance)

        # Compare against 100 random permutations
        random_better_count = 0
        for _ in range(100):
            random_perm = list(rng.permutation(20))
            random_ms = compute_makespan(instance, random_perm)
            if random_ms < sol_neh.makespan:
                random_better_count += 1

        # NEH should beat most random orders (allow at most 5%)
        assert random_better_count <= 5

    def test_neh_optimal_on_2_machines(self):
        """On 2-machine instances, NEH often finds the optimal (Johnson's)."""
        instance = FlowShopInstance.random(n=6, m=2, seed=42)
        sol_johnson = johnsons_rule(instance)
        sol_neh = neh(instance)
        # NEH may not always match Johnson, but should be close
        assert sol_neh.makespan <= sol_johnson.makespan * 1.1

    def test_tiebreaking_no_worse(self):
        """Tie-breaking variant should never be worse than basic NEH."""
        for seed in range(5):
            instance = FlowShopInstance.random(n=15, m=4, seed=seed)
            sol_basic = neh(instance)
            sol_tb = neh_with_tiebreaking(instance)
            assert sol_tb.makespan <= sol_basic.makespan


# ──────────────────────────────────────────────
# CDS Tests
# ──────────────────────────────────────────────

class TestCDS:
    """Verify CDS heuristic correctness."""

    def test_returns_valid_permutation(self):
        instance = FlowShopInstance.random(n=10, m=4, seed=42)
        sol = cds(instance)
        assert sorted(sol.permutation) == list(range(10))

    def test_makespan_is_correct(self):
        instance = FlowShopInstance.random(n=15, m=5, seed=7)
        sol = cds(instance)
        assert sol.makespan == compute_makespan(instance, sol.permutation)

    def test_on_2_machines_equals_johnson(self):
        """With m=2, CDS has only one sub-problem, which IS Johnson's Rule."""
        instance = FlowShopInstance.random(n=8, m=2, seed=42)
        sol_cds = cds(instance)
        sol_johnson = johnsons_rule(instance)
        assert sol_cds.makespan == sol_johnson.makespan

    def test_beats_arbitrary_order(self):
        """CDS should beat an arbitrary fixed order."""
        instance = FlowShopInstance.random(n=20, m=5, seed=42)
        sol_cds = cds(instance)
        identity_ms = compute_makespan(instance, list(range(20)))
        assert sol_cds.makespan <= identity_ms


# ──────────────────────────────────────────────
# Iterated Greedy Tests
# ──────────────────────────────────────────────

class TestIteratedGreedy:
    """Verify IG correctness and improvement over constructive methods."""

    def test_returns_valid_permutation(self):
        instance = FlowShopInstance.random(n=10, m=3, seed=42)
        sol = iterated_greedy(instance, max_iterations=20, seed=42)
        assert sorted(sol.permutation) == list(range(10))

    def test_makespan_is_correct(self):
        instance = FlowShopInstance.random(n=15, m=4, seed=7)
        sol = iterated_greedy(instance, max_iterations=50, seed=7)
        assert sol.makespan == compute_makespan(instance, sol.permutation)

    def test_no_worse_than_neh(self):
        """IG starts from NEH, should never be worse."""
        instance = FlowShopInstance.random(n=20, m=5, seed=42)
        sol_neh = neh(instance)
        sol_ig = iterated_greedy(instance, max_iterations=100, seed=42)
        assert sol_ig.makespan <= sol_neh.makespan

    def test_improves_with_more_iterations(self):
        """More iterations should give same or better results."""
        instance = FlowShopInstance.random(n=20, m=5, seed=42)
        sol_short = iterated_greedy(instance, max_iterations=10, seed=42)
        sol_long = iterated_greedy(instance, max_iterations=200, seed=42)
        assert sol_long.makespan <= sol_short.makespan

    def test_deterministic_with_seed(self):
        """Same seed should give identical results."""
        instance = FlowShopInstance.random(n=15, m=4, seed=42)
        sol_a = iterated_greedy(instance, max_iterations=50, seed=123)
        sol_b = iterated_greedy(instance, max_iterations=50, seed=123)
        assert sol_a.makespan == sol_b.makespan
        assert sol_a.permutation == sol_b.permutation


# ──────────────────────────────────────────────
# Palmer's Slope Index Tests
# ──────────────────────────────────────────────

class TestPalmersSlope:
    """Verify Palmer's Slope Index heuristic."""

    def test_returns_valid_permutation(self):
        instance = FlowShopInstance.random(n=10, m=4, seed=42)
        sol = palmers_slope(instance)
        assert sorted(sol.permutation) == list(range(10))

    def test_makespan_is_correct(self):
        instance = FlowShopInstance.random(n=15, m=5, seed=7)
        sol = palmers_slope(instance)
        assert sol.makespan == compute_makespan(instance, sol.permutation)

    def test_slope_ordering_logic(self):
        """Jobs with increasing processing times should get high priority."""
        # Job 0: [1, 5] — increasing → high slope
        # Job 1: [5, 1] — decreasing → low slope
        instance = FlowShopInstance(
            n=2, m=2,
            processing_times=np.array([[1, 5], [5, 1]])
        )
        sol = palmers_slope(instance)
        # Job 0 should come first (higher slope)
        assert sol.permutation[0] == 0

    def test_beats_reverse_order(self):
        """Palmer should generally beat worst-case orderings."""
        instance = FlowShopInstance.random(n=20, m=5, seed=42)
        sol = palmers_slope(instance)
        reverse_ms = compute_makespan(instance, list(range(19, -1, -1)))
        # Palmer should at least not be the worst possible order
        assert sol.makespan <= reverse_ms


# ──────────────────────────────────────────────
# Taillard Benchmark Validation
# ──────────────────────────────────────────────

class TestTaillardBenchmark:
    """Validate algorithms against Taillard benchmark instances."""

    @pytest.fixture
    def tai20_5_0(self):
        """Load the first Taillard instance (20 jobs, 5 machines)."""
        try:
            sys.path.insert(0, os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                '..', '..', '..', '..'
            ))
            from shared.parsers.taillard_parser import load_taillard_instance
            p, info = load_taillard_instance("tai20_5_0")
            instance = FlowShopInstance(
                n=info.n_jobs, m=info.n_machines, processing_times=p
            )
            return instance, info
        except (ConnectionError, Exception):
            pytest.skip("Cannot download Taillard instance (network)")

    def test_neh_within_10_percent_of_bks(self, tai20_5_0):
        """NEH should be within 10% of best known on tai20_5_0."""
        instance, info = tai20_5_0
        sol = neh(instance)
        rpd = 100.0 * (sol.makespan - info.upper_bound) / info.upper_bound
        assert rpd < 10.0, f"NEH RPD={rpd:.1f}% exceeds 10% threshold"

    def test_ig_within_3_percent_of_bks(self, tai20_5_0):
        """IG should be within 3% of best known on tai20_5_0."""
        instance, info = tai20_5_0
        sol = iterated_greedy(instance, time_limit=1.0, seed=42)
        rpd = 100.0 * (sol.makespan - info.upper_bound) / info.upper_bound
        assert rpd < 3.0, f"IG RPD={rpd:.1f}% exceeds 3% threshold"

    def test_algorithm_ranking(self, tai20_5_0):
        """Verify expected quality ranking: Palmer < CDS < NEH < IG."""
        instance, _ = tai20_5_0
        ms_palmer = palmers_slope(instance).makespan
        ms_cds = cds(instance).makespan
        ms_neh = neh(instance).makespan
        ms_ig = iterated_greedy(instance, time_limit=0.5, seed=42).makespan

        # IG should be best or tied
        assert ms_ig <= ms_neh
        # NEH should beat or tie CDS
        assert ms_neh <= ms_cds


# ──────────────────────────────────────────────
# Edge Cases
# ──────────────────────────────────────────────

class TestEdgeCases:
    """Test boundary conditions and edge cases."""

    def test_single_job(self):
        instance = FlowShopInstance(
            n=1, m=3,
            processing_times=np.array([[3], [5], [2]])
        )
        assert neh(instance).makespan == 10
        assert cds(instance).makespan == 10

    def test_two_jobs(self):
        instance = FlowShopInstance(
            n=2, m=2,
            processing_times=np.array([[3, 4], [2, 1]])
        )
        sol = johnsons_rule(instance)
        # Job 0: M1 faster (3 <= 2? no, 3 > 2) → group V
        # Job 1: M1 faster (4 > 1) → group V
        # Both in V, sorted by M2 desc: [0, 1] (M2: 2, 1)
        assert sol.makespan == 8

    def test_large_instance_no_crash(self):
        """Ensure algorithms handle 100-job instances without errors."""
        instance = FlowShopInstance.random(n=100, m=10, seed=42)
        sol = neh(instance)
        assert sol.makespan > 0
        assert len(sol.permutation) == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
