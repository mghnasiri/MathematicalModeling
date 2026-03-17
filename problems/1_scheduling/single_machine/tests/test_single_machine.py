"""
Test Suite for Single Machine Scheduling Algorithms

Covers:
    - Instance creation and objective evaluation functions
    - Dispatching rules: SPT, WSPT, EDD, LPT
    - Moore's Algorithm (1 || ΣUj)
    - ATC rule (1 || ΣwjTj)
    - Dynamic Programming (1 || ΣTj)
    - Branch and Bound (1 || ΣwjTj)
    - Simulated Annealing (1 || ΣwjTj, 1 || ΣTj)
"""

from __future__ import annotations
import sys
import os
import itertools
import importlib.util
import pytest
import numpy as np

# Use importlib.util for explicit path-based imports to avoid collision
# with flow_shop/instance.py when running from the repo root.
_this_dir = os.path.dirname(os.path.abspath(__file__))
_sm_dir = os.path.dirname(_this_dir)


def _load_module(name: str, filepath: str):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_instance_mod = _load_module("sm_instance", os.path.join(_sm_dir, "instance.py"))
_rules_mod = _load_module("sm_dispatching", os.path.join(_sm_dir, "heuristics", "dispatching_rules.py"))
_moore_mod = _load_module("sm_moore", os.path.join(_sm_dir, "heuristics", "moores_algorithm.py"))
_atc_mod = _load_module("sm_atc", os.path.join(_sm_dir, "heuristics", "apparent_tardiness_cost.py"))
_dp_mod = _load_module("sm_dp", os.path.join(_sm_dir, "exact", "dynamic_programming.py"))
_bb_mod = _load_module("sm_bb", os.path.join(_sm_dir, "exact", "branch_and_bound.py"))
_sa_mod = _load_module("sm_sa", os.path.join(_sm_dir, "metaheuristics", "simulated_annealing.py"))

SingleMachineInstance = _instance_mod.SingleMachineInstance
SingleMachineSolution = _instance_mod.SingleMachineSolution
compute_completion_times = _instance_mod.compute_completion_times
compute_total_completion_time = _instance_mod.compute_total_completion_time
compute_weighted_completion_time = _instance_mod.compute_weighted_completion_time
compute_makespan = _instance_mod.compute_makespan
compute_maximum_lateness = _instance_mod.compute_maximum_lateness
compute_total_tardiness = _instance_mod.compute_total_tardiness
compute_weighted_tardiness = _instance_mod.compute_weighted_tardiness
compute_number_tardy = _instance_mod.compute_number_tardy

spt = _rules_mod.spt
wspt = _rules_mod.wspt
edd = _rules_mod.edd
lpt = _rules_mod.lpt

moores_algorithm = _moore_mod.moores_algorithm
atc = _atc_mod.atc
dp_total_tardiness = _dp_mod.dp_total_tardiness
branch_and_bound_weighted_tardiness = _bb_mod.branch_and_bound_weighted_tardiness
simulated_annealing_wt = _sa_mod.simulated_annealing_wt
simulated_annealing_tt = _sa_mod.simulated_annealing_tt


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_instance():
    """5-job instance with all attributes."""
    return SingleMachineInstance.from_arrays(
        processing_times=[3, 5, 2, 7, 4],
        weights=[2, 1, 3, 1, 2],
        due_dates=[10, 12, 8, 20, 15],
    )


@pytest.fixture
def edd_optimal_instance():
    """Instance where EDD gives an obvious Lmax-optimal schedule."""
    return SingleMachineInstance.from_arrays(
        processing_times=[2, 3, 1],
        due_dates=[3, 7, 10],
    )


@pytest.fixture
def single_job():
    """Trivial single-job instance."""
    return SingleMachineInstance.from_arrays(
        processing_times=[5],
        weights=[3],
        due_dates=[10],
    )


@pytest.fixture
def tight_deadlines():
    """Instance with tight deadlines — many jobs tardy."""
    return SingleMachineInstance.from_arrays(
        processing_times=[5, 3, 7, 2, 4],
        weights=[1, 2, 1, 3, 2],
        due_dates=[4, 5, 6, 3, 7],
    )


# ---------------------------------------------------------------------------
# Test Instance and Objectives
# ---------------------------------------------------------------------------

class TestInstance:
    def test_creation(self, small_instance):
        assert small_instance.n == 5
        assert small_instance.processing_times.shape == (5,)
        assert small_instance.weights.shape == (5,)
        assert small_instance.due_dates.shape == (5,)

    def test_random_generation(self):
        inst = SingleMachineInstance.random(n=10, seed=42)
        assert inst.n == 10
        assert inst.processing_times.shape == (10,)
        assert inst.weights.shape == (10,)
        assert inst.due_dates.shape == (10,)
        assert all(inst.processing_times >= 1)

    def test_random_reproducibility(self):
        inst1 = SingleMachineInstance.random(n=10, seed=42)
        inst2 = SingleMachineInstance.random(n=10, seed=42)
        np.testing.assert_array_equal(inst1.processing_times, inst2.processing_times)
        np.testing.assert_array_equal(inst1.due_dates, inst2.due_dates)

    def test_from_arrays(self):
        inst = SingleMachineInstance.from_arrays([3, 5], weights=[1, 2])
        assert inst.n == 2
        assert inst.due_dates is None

    def test_single_job(self, single_job):
        assert single_job.n == 1
        C = compute_completion_times(single_job, [0])
        assert C[0] == 5

    def test_completion_times(self, small_instance):
        seq = [0, 1, 2, 3, 4]
        C = compute_completion_times(small_instance, seq)
        # C[0] = 3, C[1] = 3+5=8, C[2] = 8+2=10, C[3] = 10+7=17, C[4] = 17+4=21
        np.testing.assert_array_equal(C, [3, 8, 10, 17, 21])

    def test_total_completion_time(self, small_instance):
        seq = [0, 1, 2, 3, 4]
        assert compute_total_completion_time(small_instance, seq) == 3 + 8 + 10 + 17 + 21

    def test_weighted_completion_time(self, small_instance):
        seq = [0, 1, 2, 3, 4]
        # w*C: 2*3 + 1*8 + 3*10 + 1*17 + 2*21 = 6+8+30+17+42 = 103
        assert compute_weighted_completion_time(small_instance, seq) == 103

    def test_maximum_lateness(self, small_instance):
        seq = [0, 1, 2, 3, 4]
        # L: 3-10=-7, 8-12=-4, 10-8=2, 17-20=-3, 21-15=6
        assert compute_maximum_lateness(small_instance, seq) == 6

    def test_total_tardiness(self, small_instance):
        seq = [0, 1, 2, 3, 4]
        # T: max(0,-7)=0, max(0,-4)=0, max(0,2)=2, max(0,-3)=0, max(0,6)=6
        assert compute_total_tardiness(small_instance, seq) == 8

    def test_number_tardy(self, small_instance):
        seq = [0, 1, 2, 3, 4]
        # Jobs 2 and 4 are tardy
        assert compute_number_tardy(small_instance, seq) == 2


# ---------------------------------------------------------------------------
# Test SPT
# ---------------------------------------------------------------------------

class TestSPT:
    def test_optimality_small(self, small_instance):
        """SPT is optimal for ΣCj — verify by brute force."""
        sol = spt(small_instance)

        # Brute force
        best_obj = float('inf')
        for perm in itertools.permutations(range(small_instance.n)):
            obj = compute_total_completion_time(small_instance, list(perm))
            best_obj = min(best_obj, obj)

        assert sol.objective_value == best_obj

    def test_spt_ordering(self, small_instance):
        """SPT should sort by non-decreasing processing time."""
        sol = spt(small_instance)
        p = small_instance.processing_times
        for i in range(len(sol.sequence) - 1):
            assert p[sol.sequence[i]] <= p[sol.sequence[i + 1]]

    def test_single_job(self, single_job):
        sol = spt(single_job)
        assert sol.sequence == [0]
        assert sol.objective_value == 5


# ---------------------------------------------------------------------------
# Test WSPT
# ---------------------------------------------------------------------------

class TestWSPT:
    def test_optimality_small(self, small_instance):
        """WSPT is optimal for ΣwjCj — verify by brute force."""
        sol = wspt(small_instance)

        best_obj = float('inf')
        for perm in itertools.permutations(range(small_instance.n)):
            obj = compute_weighted_completion_time(small_instance, list(perm))
            best_obj = min(best_obj, obj)

        assert sol.objective_value == best_obj

    def test_wspt_ratio_ordering(self, small_instance):
        """WSPT sorts by non-decreasing p/w ratio."""
        sol = wspt(small_instance)
        p = small_instance.processing_times
        w = small_instance.weights
        for i in range(len(sol.sequence) - 1):
            j1, j2 = sol.sequence[i], sol.sequence[i + 1]
            assert p[j1] / w[j1] <= p[j2] / w[j2] + 1e-10

    def test_wspt_no_weights(self):
        """WSPT without weights should behave like SPT."""
        inst = SingleMachineInstance.from_arrays(processing_times=[5, 2, 8, 1])
        sol = wspt(inst)
        sol_spt = spt(inst)
        assert sol.sequence == sol_spt.sequence


# ---------------------------------------------------------------------------
# Test EDD
# ---------------------------------------------------------------------------

class TestEDD:
    def test_optimality_small(self, small_instance):
        """EDD is optimal for Lmax — verify by brute force."""
        sol = edd(small_instance)

        best_obj = float('inf')
        for perm in itertools.permutations(range(small_instance.n)):
            obj = compute_maximum_lateness(small_instance, list(perm))
            best_obj = min(best_obj, obj)

        assert sol.objective_value == best_obj

    def test_edd_ordering(self, small_instance):
        """EDD should sort by non-decreasing due date."""
        sol = edd(small_instance)
        d = small_instance.due_dates
        for i in range(len(sol.sequence) - 1):
            assert d[sol.sequence[i]] <= d[sol.sequence[i + 1]]

    def test_no_tardy(self, edd_optimal_instance):
        """All jobs on time when total processing fits within all due dates."""
        sol = edd(edd_optimal_instance)
        assert sol.objective_value <= 0  # Lmax <= 0 means all on time


# ---------------------------------------------------------------------------
# Test LPT
# ---------------------------------------------------------------------------

class TestLPT:
    def test_ordering(self, small_instance):
        """LPT should sort by non-increasing processing time."""
        sol = lpt(small_instance)
        p = small_instance.processing_times
        for i in range(len(sol.sequence) - 1):
            assert p[sol.sequence[i]] >= p[sol.sequence[i + 1]]


# ---------------------------------------------------------------------------
# Test Moore's Algorithm
# ---------------------------------------------------------------------------

class TestMoore:
    def test_optimality_small(self, small_instance):
        """Moore's is optimal for ΣUj — verify by brute force."""
        sol = moores_algorithm(small_instance)

        best_obj = float('inf')
        for perm in itertools.permutations(range(small_instance.n)):
            obj = compute_number_tardy(small_instance, list(perm))
            best_obj = min(best_obj, obj)

        assert sol.objective_value == best_obj

    def test_all_on_time(self, edd_optimal_instance):
        """When all jobs can be on time, ΣUj = 0."""
        sol = moores_algorithm(edd_optimal_instance)
        assert sol.objective_value == 0

    def test_tight_deadlines(self, tight_deadlines):
        """With tight deadlines, verify optimality by brute force."""
        sol = moores_algorithm(tight_deadlines)

        best_obj = float('inf')
        for perm in itertools.permutations(range(tight_deadlines.n)):
            obj = compute_number_tardy(tight_deadlines, list(perm))
            best_obj = min(best_obj, obj)

        assert sol.objective_value == best_obj

    def test_single_job_on_time(self, single_job):
        sol = moores_algorithm(single_job)
        assert sol.objective_value == 0

    def test_single_job_tardy(self):
        inst = SingleMachineInstance.from_arrays(
            processing_times=[10],
            due_dates=[5],
        )
        sol = moores_algorithm(inst)
        assert sol.objective_value == 1

    def test_valid_permutation(self, small_instance):
        sol = moores_algorithm(small_instance)
        assert sorted(sol.sequence) == list(range(small_instance.n))


# ---------------------------------------------------------------------------
# Test ATC
# ---------------------------------------------------------------------------

class TestATC:
    def test_valid_solution(self, small_instance):
        sol = atc(small_instance)
        assert sorted(sol.sequence) == list(range(small_instance.n))
        assert sol.objective_value >= 0

    def test_different_K_values(self, small_instance):
        """Different K values should produce valid but possibly different solutions."""
        results = []
        for k in [0.5, 1.0, 2.0, 5.0]:
            sol = atc(small_instance, K=k)
            assert sorted(sol.sequence) == list(range(small_instance.n))
            results.append(sol.objective_value)
        # All should be non-negative
        assert all(r >= 0 for r in results)

    def test_high_K_approaches_wspt(self):
        """With very high K, ATC should behave like WSPT."""
        inst = SingleMachineInstance.from_arrays(
            processing_times=[3, 5, 2, 7, 4],
            weights=[2, 1, 3, 1, 2],
            due_dates=[100, 100, 100, 100, 100],  # very loose
        )
        sol_atc = atc(inst, K=100.0)
        sol_wspt = wspt(inst)
        # With very loose deadlines and high K, ATC should approximate WSPT
        assert sol_atc.sequence == sol_wspt.sequence

    def test_quality_vs_edd(self, tight_deadlines):
        """ATC should produce reasonable solutions."""
        sol = atc(tight_deadlines)
        # Just verify it's a valid permutation with non-negative objective
        assert sorted(sol.sequence) == list(range(tight_deadlines.n))
        assert sol.objective_value >= 0


# ---------------------------------------------------------------------------
# Test DP for Total Tardiness
# ---------------------------------------------------------------------------

class TestDPTotalTardiness:
    def test_optimality_small(self, small_instance):
        """Verify DP against brute force for 5-job instance."""
        sol = dp_total_tardiness(small_instance)

        best_obj = float('inf')
        for perm in itertools.permutations(range(small_instance.n)):
            obj = compute_total_tardiness(small_instance, list(perm))
            best_obj = min(best_obj, obj)

        assert sol.objective_value == best_obj

    def test_no_tardiness(self, edd_optimal_instance):
        """When EDD gives zero tardiness, DP should find zero."""
        sol = dp_total_tardiness(edd_optimal_instance)
        assert sol.objective_value == 0

    def test_tight_deadlines(self, tight_deadlines):
        """Verify DP optimality on tight-deadline instance."""
        sol = dp_total_tardiness(tight_deadlines)

        best_obj = float('inf')
        for perm in itertools.permutations(range(tight_deadlines.n)):
            obj = compute_total_tardiness(tight_deadlines, list(perm))
            best_obj = min(best_obj, obj)

        assert sol.objective_value == best_obj

    def test_single_job(self, single_job):
        sol = dp_total_tardiness(single_job)
        assert sol.sequence == [0]
        assert sol.objective_value == 0  # C=5, d=10, not tardy

    def test_valid_permutation(self, small_instance):
        sol = dp_total_tardiness(small_instance)
        assert sorted(sol.sequence) == list(range(small_instance.n))

    def test_max_jobs_limit(self):
        """Should raise ValueError for instances exceeding max_jobs."""
        inst = SingleMachineInstance.random(n=25, seed=42)
        with pytest.raises(ValueError, match="exceeding max_jobs"):
            dp_total_tardiness(inst, max_jobs=20)

    def test_objective_matches_evaluation(self, small_instance):
        """Reported objective should match independent evaluation."""
        sol = dp_total_tardiness(small_instance)
        verify = compute_total_tardiness(small_instance, sol.sequence)
        assert sol.objective_value == verify


# ---------------------------------------------------------------------------
# Test B&B for Weighted Tardiness
# ---------------------------------------------------------------------------

class TestBBWeightedTardiness:
    def test_optimality_small(self, small_instance):
        """Verify B&B against brute force for 5-job instance."""
        sol = branch_and_bound_weighted_tardiness(small_instance)

        best_obj = float('inf')
        for perm in itertools.permutations(range(small_instance.n)):
            obj = compute_weighted_tardiness(small_instance, list(perm))
            best_obj = min(best_obj, obj)

        assert sol.objective_value == best_obj

    def test_tight_deadlines(self, tight_deadlines):
        """B&B optimality on tight-deadline instance."""
        sol = branch_and_bound_weighted_tardiness(tight_deadlines)

        best_obj = float('inf')
        for perm in itertools.permutations(range(tight_deadlines.n)):
            obj = compute_weighted_tardiness(tight_deadlines, list(perm))
            best_obj = min(best_obj, obj)

        assert sol.objective_value == best_obj

    def test_no_tardiness(self, edd_optimal_instance):
        """When no job is tardy, weighted tardiness = 0."""
        inst = SingleMachineInstance.from_arrays(
            processing_times=[2, 3, 1],
            weights=[1, 1, 1],
            due_dates=[3, 7, 10],
        )
        sol = branch_and_bound_weighted_tardiness(inst)
        assert sol.objective_value == 0

    def test_single_job(self, single_job):
        sol = branch_and_bound_weighted_tardiness(single_job)
        assert sol.sequence == [0]
        assert sol.objective_value == 0

    def test_valid_permutation(self, small_instance):
        sol = branch_and_bound_weighted_tardiness(small_instance)
        assert sorted(sol.sequence) == list(range(small_instance.n))

    def test_objective_matches_evaluation(self, small_instance):
        sol = branch_and_bound_weighted_tardiness(small_instance)
        verify = compute_weighted_tardiness(small_instance, sol.sequence)
        assert sol.objective_value == verify


# ---------------------------------------------------------------------------
# Test SA for Weighted Tardiness
# ---------------------------------------------------------------------------

class TestSAWeightedTardiness:
    def test_valid_solution(self, small_instance):
        sol = simulated_annealing_wt(small_instance, max_iterations=500, seed=42)
        assert sorted(sol.sequence) == list(range(small_instance.n))
        assert sol.objective_value >= 0

    def test_improves_over_initial(self):
        """On a non-trivial instance, SA should match or improve ATC."""
        inst = SingleMachineInstance.random(n=15, seed=42)
        sol_atc = atc(inst)
        sol_sa = simulated_annealing_wt(inst, max_iterations=5000, seed=42)
        assert sol_sa.objective_value <= sol_atc.objective_value

    def test_reproducibility(self, small_instance):
        sol1 = simulated_annealing_wt(small_instance, max_iterations=500, seed=123)
        sol2 = simulated_annealing_wt(small_instance, max_iterations=500, seed=123)
        assert sol1.sequence == sol2.sequence
        assert sol1.objective_value == sol2.objective_value

    def test_time_limit(self, small_instance):
        import time
        start = time.time()
        sol = simulated_annealing_wt(
            small_instance, max_iterations=10**8, time_limit=1.0, seed=42
        )
        elapsed = time.time() - start
        assert elapsed < 3.0  # generous margin
        assert sorted(sol.sequence) == list(range(small_instance.n))

    def test_objective_matches_evaluation(self):
        inst = SingleMachineInstance.random(n=10, seed=42)
        sol = simulated_annealing_wt(inst, max_iterations=1000, seed=42)
        verify = compute_weighted_tardiness(inst, sol.sequence)
        assert sol.objective_value == verify


# ---------------------------------------------------------------------------
# Test SA for Total Tardiness
# ---------------------------------------------------------------------------

class TestSATotalTardiness:
    def test_valid_solution(self, small_instance):
        sol = simulated_annealing_tt(small_instance, max_iterations=500, seed=42)
        assert sorted(sol.sequence) == list(range(small_instance.n))
        assert sol.objective_value >= 0

    def test_quality_vs_dp(self):
        """SA should find solutions close to optimal on small instances."""
        inst = SingleMachineInstance.random(n=10, seed=42)
        sol_dp = dp_total_tardiness(inst)
        sol_sa = simulated_annealing_tt(inst, max_iterations=5000, seed=42)
        # SA should be within 50% of optimal (generous bound)
        if sol_dp.objective_value > 0:
            assert sol_sa.objective_value <= sol_dp.objective_value * 1.5
        else:
            assert sol_sa.objective_value >= 0

    def test_objective_matches_evaluation(self):
        inst = SingleMachineInstance.random(n=10, seed=42)
        sol = simulated_annealing_tt(inst, max_iterations=1000, seed=42)
        verify = compute_total_tardiness(inst, sol.sequence)
        assert sol.objective_value == verify


# ---------------------------------------------------------------------------
# Cross-algorithm consistency
# ---------------------------------------------------------------------------

class TestCrossAlgorithm:
    def test_dp_vs_bb_small(self):
        """DP total tardiness and brute-force should agree."""
        inst = SingleMachineInstance.from_arrays(
            processing_times=[4, 2, 6, 3],
            weights=[1, 1, 1, 1],
            due_dates=[8, 5, 15, 10],
        )
        sol_dp = dp_total_tardiness(inst)

        # Brute force total tardiness
        best_obj = float('inf')
        for perm in itertools.permutations(range(inst.n)):
            obj = compute_total_tardiness(inst, list(perm))
            best_obj = min(best_obj, obj)

        assert sol_dp.objective_value == best_obj

    def test_all_rules_valid_permutations(self):
        """All dispatching rules produce valid permutations."""
        inst = SingleMachineInstance.random(n=10, seed=42)
        jobs = list(range(inst.n))

        assert sorted(spt(inst).sequence) == jobs
        assert sorted(wspt(inst).sequence) == jobs
        assert sorted(edd(inst).sequence) == jobs
        assert sorted(lpt(inst).sequence) == jobs
        assert sorted(moores_algorithm(inst).sequence) == jobs
        assert sorted(atc(inst).sequence) == jobs

    def test_spt_optimal_for_sum_cj(self):
        """No rule can beat SPT for ΣCj."""
        inst = SingleMachineInstance.random(n=8, seed=42)
        spt_obj = compute_total_completion_time(inst, spt(inst).sequence)

        for rule_fn in [wspt, edd, lpt]:
            sol = rule_fn(inst)
            rule_obj = compute_total_completion_time(inst, sol.sequence)
            assert spt_obj <= rule_obj
