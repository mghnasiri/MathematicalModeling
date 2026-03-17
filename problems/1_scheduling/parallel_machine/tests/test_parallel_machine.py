"""
Test Suite — Parallel Machine Scheduling

Tests cover:
1. Instance creation and objective computation (identical, uniform, unrelated)
2. LPT heuristic and SPT heuristic
3. MULTIFIT heuristic
4. List scheduling heuristic
5. MIP formulation (exact)
6. Genetic Algorithm metaheuristic
7. Cross-algorithm comparisons and edge cases
"""

import sys
import os
import importlib.util
import numpy as np
import pytest

_parallel_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_module(name: str, filepath: str):
    """Load a Python module from an explicit file path."""
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Use explicit file-path imports to avoid collision with flow_shop/instance.py
_inst_mod = _load_module("pm_instance", os.path.join(_parallel_dir, "instance.py"))

ParallelMachineInstance = _inst_mod.ParallelMachineInstance
ParallelMachineSolution = _inst_mod.ParallelMachineSolution
compute_makespan = _inst_mod.compute_makespan
compute_machine_loads = _inst_mod.compute_machine_loads
compute_total_completion_time = _inst_mod.compute_total_completion_time

# Insert at position 0 so parallel_machine modules resolve correctly
sys.path.insert(0, _parallel_dir)

_lpt_mod = _load_module("pm_lpt", os.path.join(_parallel_dir, "heuristics", "lpt.py"))
lpt = _lpt_mod.lpt
spt_parallel = _lpt_mod.spt_parallel

_mf_mod = _load_module("pm_multifit", os.path.join(_parallel_dir, "heuristics", "multifit.py"))
multifit = _mf_mod.multifit

_ls_mod = _load_module("pm_list_sched", os.path.join(_parallel_dir, "heuristics", "list_scheduling.py"))
list_scheduling = _ls_mod.list_scheduling

_mip_mod = _load_module("pm_mip", os.path.join(_parallel_dir, "exact", "mip_makespan.py"))
mip_makespan = _mip_mod.mip_makespan

_ga_mod = _load_module("pm_ga", os.path.join(_parallel_dir, "metaheuristics", "genetic_algorithm.py"))
genetic_algorithm = _ga_mod.genetic_algorithm


# ──────────────────────────────────────────────
# Instance Tests
# ──────────────────────────────────────────────

class TestInstance:
    """Verify instance creation and objective computation."""

    def test_identical_creation(self):
        inst = ParallelMachineInstance.random_identical(n=10, m=3, seed=42)
        assert inst.n == 10
        assert inst.m == 3
        assert inst.processing_times.shape == (10,)
        assert inst.machine_type == "identical"

    def test_uniform_creation(self):
        inst = ParallelMachineInstance.random_uniform(n=10, m=3, seed=42)
        assert inst.n == 10
        assert inst.m == 3
        assert inst.speeds.shape == (3,)
        assert inst.machine_type == "uniform"

    def test_unrelated_creation(self):
        inst = ParallelMachineInstance.random_unrelated(n=10, m=3, seed=42)
        assert inst.n == 10
        assert inst.m == 3
        assert inst.processing_times.shape == (3, 10)
        assert inst.machine_type == "unrelated"

    def test_get_processing_time_identical(self):
        inst = ParallelMachineInstance(
            n=3, m=2,
            processing_times=np.array([10.0, 20.0, 30.0]),
            machine_type="identical",
        )
        assert inst.get_processing_time(0, 0) == 10.0
        assert inst.get_processing_time(0, 1) == 10.0  # same on all machines

    def test_get_processing_time_uniform(self):
        inst = ParallelMachineInstance(
            n=2, m=2,
            processing_times=np.array([10.0, 20.0]),
            speeds=np.array([1.0, 2.0]),
            machine_type="uniform",
        )
        assert inst.get_processing_time(0, 0) == 10.0
        assert inst.get_processing_time(0, 1) == 5.0  # 10/2

    def test_get_processing_time_unrelated(self):
        inst = ParallelMachineInstance(
            n=2, m=2,
            processing_times=np.array([[10.0, 20.0], [30.0, 5.0]]),
            machine_type="unrelated",
        )
        assert inst.get_processing_time(0, 0) == 10.0
        assert inst.get_processing_time(0, 1) == 30.0

    def test_compute_makespan(self):
        inst = ParallelMachineInstance(
            n=4, m=2,
            processing_times=np.array([10.0, 20.0, 30.0, 40.0]),
            machine_type="identical",
        )
        # Machine 0: jobs 0,2 (10+30=40), Machine 1: jobs 1,3 (20+40=60)
        assignment = [[0, 2], [1, 3]]
        assert compute_makespan(inst, assignment) == 60.0

    def test_compute_machine_loads(self):
        inst = ParallelMachineInstance(
            n=3, m=2,
            processing_times=np.array([10.0, 20.0, 30.0]),
            machine_type="identical",
        )
        assignment = [[0, 1], [2]]
        loads = compute_machine_loads(inst, assignment)
        assert loads[0] == 30.0
        assert loads[1] == 30.0

    def test_total_completion_time(self):
        inst = ParallelMachineInstance(
            n=3, m=1,
            processing_times=np.array([10.0, 20.0, 30.0]),
            machine_type="identical",
        )
        # Single machine, order [0,1,2]: C0=10, C1=30, C2=60
        # With unit weights: total = 10 + 30 + 60 = 100
        assignment = [[0, 1, 2]]
        total = compute_total_completion_time(inst, assignment)
        assert total == 100.0


# ──────────────────────────────────────────────
# LPT Tests
# ──────────────────────────────────────────────

class TestLPT:
    """Verify LPT heuristic correctness."""

    def test_valid_assignment(self):
        inst = ParallelMachineInstance.random_identical(n=10, m=3, seed=42)
        sol = lpt(inst)
        all_jobs = sorted(j for jobs in sol.assignment for j in jobs)
        assert all_jobs == list(range(10))

    def test_makespan_correct(self):
        inst = ParallelMachineInstance.random_identical(n=15, m=3, seed=7)
        sol = lpt(inst)
        assert sol.makespan == compute_makespan(inst, sol.assignment)

    def test_approximation_bound(self):
        """LPT should be within 4/3 of optimal lower bound."""
        inst = ParallelMachineInstance.random_identical(n=20, m=3, seed=42)
        sol = lpt(inst)
        lb = max(float(inst.processing_times.max()),
                 float(inst.processing_times.sum()) / inst.m)
        assert sol.makespan <= lb * (4.0 / 3.0) + 1e-9

    def test_beats_arbitrary_assignment(self):
        """LPT should beat a simple round-robin assignment."""
        inst = ParallelMachineInstance.random_identical(n=20, m=3, seed=42)
        sol = lpt(inst)
        # Round-robin
        rr = [[] for _ in range(inst.m)]
        for j in range(inst.n):
            rr[j % inst.m].append(j)
        rr_ms = compute_makespan(inst, rr)
        assert sol.makespan <= rr_ms

    def test_two_machines_small(self):
        """Small 2-machine instance with known optimum."""
        inst = ParallelMachineInstance(
            n=4, m=2,
            processing_times=np.array([7.0, 5.0, 3.0, 1.0]),
            machine_type="identical",
        )
        sol = lpt(inst)
        # Optimal: {7,1}=8, {5,3}=8 → Cmax=8
        assert sol.makespan == 8.0

    def test_unrelated_machines(self):
        """LPT should work for unrelated machines."""
        inst = ParallelMachineInstance.random_unrelated(n=10, m=3, seed=42)
        sol = lpt(inst)
        all_jobs = sorted(j for jobs in sol.assignment for j in jobs)
        assert all_jobs == list(range(10))
        assert sol.makespan == compute_makespan(inst, sol.assignment)

    def test_single_machine(self):
        """Single machine: all jobs on one machine."""
        inst = ParallelMachineInstance(
            n=3, m=1,
            processing_times=np.array([10.0, 20.0, 30.0]),
            machine_type="identical",
        )
        sol = lpt(inst)
        assert sol.makespan == 60.0

    def test_single_job(self):
        """Single job: assigned to one machine."""
        inst = ParallelMachineInstance(
            n=1, m=3,
            processing_times=np.array([42.0]),
            machine_type="identical",
        )
        sol = lpt(inst)
        assert sol.makespan == 42.0


# ──────────────────────────────────────────────
# SPT Tests
# ──────────────────────────────────────────────

class TestSPT:
    """Verify SPT heuristic for total completion time."""

    def test_valid_assignment(self):
        inst = ParallelMachineInstance.random_identical(n=10, m=3, seed=42)
        sol = spt_parallel(inst)
        all_jobs = sorted(j for jobs in sol.assignment for j in jobs)
        assert all_jobs == list(range(10))

    def test_spt_ordering(self):
        """Jobs should be sorted by increasing processing time across machines."""
        inst = ParallelMachineInstance(
            n=6, m=2,
            processing_times=np.array([50.0, 10.0, 30.0, 20.0, 40.0, 5.0]),
            machine_type="identical",
        )
        sol = spt_parallel(inst)
        # SPT order: [5,1,3,2,4,0] (5,10,20,30,40,50)
        # Round-robin on 2 machines
        all_jobs = []
        for jobs in sol.assignment:
            all_jobs.extend(jobs)
        # Check that smallest jobs come first
        sorted_by_pt = sorted(range(6), key=lambda j: inst.processing_times[j])
        # Each machine should have jobs in SPT order
        for jobs in sol.assignment:
            pts = [inst.processing_times[j] for j in jobs]
            assert pts == sorted(pts)


# ──────────────────────────────────────────────
# MULTIFIT Tests
# ──────────────────────────────────────────────

class TestMultifit:
    """Verify MULTIFIT heuristic correctness."""

    def test_valid_assignment(self):
        inst = ParallelMachineInstance.random_identical(n=15, m=3, seed=42)
        sol = multifit(inst)
        all_jobs = sorted(j for jobs in sol.assignment for j in jobs)
        assert all_jobs == list(range(15))

    def test_makespan_correct(self):
        inst = ParallelMachineInstance.random_identical(n=15, m=3, seed=7)
        sol = multifit(inst)
        assert abs(sol.makespan - compute_makespan(inst, sol.assignment)) < 1e-9

    def test_no_worse_than_2x_lb(self):
        """MULTIFIT should be within 1.22 of lower bound (generous test: 1.5)."""
        inst = ParallelMachineInstance.random_identical(n=20, m=3, seed=42)
        sol = multifit(inst)
        lb = max(float(inst.processing_times.max()),
                 float(inst.processing_times.sum()) / inst.m)
        assert sol.makespan <= lb * 1.5

    def test_competitive_with_lpt(self):
        """MULTIFIT should be no worse than LPT on most instances."""
        inst = ParallelMachineInstance.random_identical(n=30, m=4, seed=42)
        sol_mf = multifit(inst)
        sol_lpt = lpt(inst)
        # Allow small tolerance — MULTIFIT should be competitive
        assert sol_mf.makespan <= sol_lpt.makespan * 1.05 + 1

    def test_small_instance(self):
        """Small instance: should find good solution."""
        inst = ParallelMachineInstance(
            n=4, m=2,
            processing_times=np.array([7.0, 5.0, 3.0, 1.0]),
            machine_type="identical",
        )
        sol = multifit(inst)
        # Optimal is 8 ({7,1}, {5,3})
        assert sol.makespan <= 8.0


# ──────────────────────────────────────────────
# List Scheduling Tests
# ──────────────────────────────────────────────

class TestListScheduling:
    """Verify list scheduling heuristic."""

    def test_valid_assignment(self):
        inst = ParallelMachineInstance.random_identical(n=10, m=3, seed=42)
        sol = list_scheduling(inst)
        all_jobs = sorted(j for jobs in sol.assignment for j in jobs)
        assert all_jobs == list(range(10))

    def test_makespan_correct(self):
        inst = ParallelMachineInstance.random_identical(n=15, m=3, seed=7)
        sol = list_scheduling(inst)
        assert abs(sol.makespan - compute_makespan(inst, sol.assignment)) < 1e-9

    def test_approximation_bound(self):
        """List scheduling should be within 2 - 1/m of optimal."""
        inst = ParallelMachineInstance.random_identical(n=20, m=3, seed=42)
        sol = list_scheduling(inst)
        lb = max(float(inst.processing_times.max()),
                 float(inst.processing_times.sum()) / inst.m)
        bound = 2.0 - 1.0 / inst.m
        assert sol.makespan <= lb * bound + 1e-9

    def test_custom_order(self):
        """Custom job order should be respected."""
        inst = ParallelMachineInstance(
            n=4, m=2,
            processing_times=np.array([10.0, 20.0, 30.0, 40.0]),
            machine_type="identical",
        )
        # LPT order
        sol_lpt = list_scheduling(inst, job_order=[3, 2, 1, 0])
        # Natural order
        sol_nat = list_scheduling(inst, job_order=[0, 1, 2, 3])
        # LPT order should give better or equal makespan
        assert sol_lpt.makespan <= sol_nat.makespan


# ──────────────────────────────────────────────
# MIP Tests
# ──────────────────────────────────────────────

class TestMIP:
    """Verify MIP formulation correctness."""

    def test_valid_assignment(self):
        inst = ParallelMachineInstance.random_identical(n=8, m=2, seed=42)
        sol = mip_makespan(inst, time_limit=10.0)
        all_jobs = sorted(j for jobs in sol.assignment for j in jobs)
        assert all_jobs == list(range(8))

    def test_makespan_correct(self):
        inst = ParallelMachineInstance.random_identical(n=8, m=2, seed=7)
        sol = mip_makespan(inst, time_limit=10.0)
        assert abs(sol.makespan - compute_makespan(inst, sol.assignment)) < 1e-9

    def test_optimal_small_instance(self):
        """MIP should find optimal for small instances."""
        inst = ParallelMachineInstance(
            n=4, m=2,
            processing_times=np.array([7.0, 5.0, 3.0, 1.0]),
            machine_type="identical",
        )
        sol = mip_makespan(inst, time_limit=10.0)
        assert sol.makespan == 8.0

    def test_optimal_matches_lower_bound(self):
        """For easy instances, MIP should match the lower bound."""
        inst = ParallelMachineInstance(
            n=4, m=2,
            processing_times=np.array([10.0, 10.0, 10.0, 10.0]),
            machine_type="identical",
        )
        sol = mip_makespan(inst, time_limit=10.0)
        # LB = max(10, 40/2) = 20
        assert sol.makespan == 20.0

    def test_no_worse_than_lpt(self):
        """MIP should be at least as good as LPT."""
        inst = ParallelMachineInstance.random_identical(n=10, m=3, seed=42)
        sol_mip = mip_makespan(inst, time_limit=10.0)
        sol_lpt = lpt(inst)
        assert sol_mip.makespan <= sol_lpt.makespan + 1e-9


# ──────────────────────────────────────────────
# Genetic Algorithm Tests
# ──────────────────────────────────────────────

class TestGeneticAlgorithm:
    """Verify Genetic Algorithm correctness."""

    def test_valid_assignment(self):
        inst = ParallelMachineInstance.random_identical(n=10, m=3, seed=42)
        sol = genetic_algorithm(inst, max_generations=100, seed=42)
        all_jobs = sorted(j for jobs in sol.assignment for j in jobs)
        assert all_jobs == list(range(10))

    def test_makespan_correct(self):
        inst = ParallelMachineInstance.random_identical(n=15, m=3, seed=7)
        sol = genetic_algorithm(inst, max_generations=200, seed=7)
        assert abs(sol.makespan - compute_makespan(inst, sol.assignment)) < 1e-9

    def test_no_worse_than_lpt(self):
        """GA population includes LPT, should never be worse."""
        inst = ParallelMachineInstance.random_identical(n=15, m=3, seed=42)
        sol_lpt = lpt(inst)
        sol_ga = genetic_algorithm(inst, max_generations=200, seed=42)
        assert sol_ga.makespan <= sol_lpt.makespan + 1e-9

    def test_deterministic_with_seed(self):
        """Same seed should give identical results."""
        inst = ParallelMachineInstance.random_identical(n=10, m=3, seed=42)
        sol_a = genetic_algorithm(inst, max_generations=100, seed=123)
        sol_b = genetic_algorithm(inst, max_generations=100, seed=123)
        assert sol_a.makespan == sol_b.makespan

    def test_memetic_variant(self):
        """Memetic GA with local search should produce valid solutions."""
        inst = ParallelMachineInstance.random_identical(n=10, m=3, seed=42)
        sol = genetic_algorithm(
            inst, max_generations=50, use_local_search=True, seed=42
        )
        all_jobs = sorted(j for jobs in sol.assignment for j in jobs)
        assert all_jobs == list(range(10))

    def test_time_limited_run(self):
        inst = ParallelMachineInstance.random_identical(n=20, m=3, seed=42)
        sol = genetic_algorithm(inst, time_limit=0.5, seed=42)
        all_jobs = sorted(j for jobs in sol.assignment for j in jobs)
        assert all_jobs == list(range(20))


# ──────────────────────────────────────────────
# Edge Cases
# ──────────────────────────────────────────────

class TestEdgeCases:
    """Edge cases and cross-algorithm tests."""

    def test_single_job_all_algorithms(self):
        inst = ParallelMachineInstance(
            n=1, m=3,
            processing_times=np.array([42.0]),
            machine_type="identical",
        )
        for name, sol in [
            ("LPT", lpt(inst)),
            ("List", list_scheduling(inst)),
            ("MULTIFIT", multifit(inst)),
            ("MIP", mip_makespan(inst, time_limit=5.0)),
            ("GA", genetic_algorithm(inst, max_generations=10, seed=42)),
        ]:
            assert sol.makespan == 42.0, f"{name} failed on single job"

    def test_equal_jobs_two_machines(self):
        """n equal jobs on 2 machines: optimal = ceil(n/2) * p."""
        p = 10.0
        n = 6
        inst = ParallelMachineInstance(
            n=n, m=2,
            processing_times=np.full(n, p),
            machine_type="identical",
        )
        optimal = (n // 2) * p
        sol = lpt(inst)
        assert sol.makespan == optimal

    def test_more_machines_than_jobs(self):
        """When m > n, each job gets its own machine."""
        inst = ParallelMachineInstance(
            n=3, m=5,
            processing_times=np.array([10.0, 20.0, 30.0]),
            machine_type="identical",
        )
        sol = lpt(inst)
        assert sol.makespan == 30.0  # One job per machine

    def test_all_algorithms_consistent(self):
        """All algorithms should return valid assignments on a medium instance."""
        inst = ParallelMachineInstance.random_identical(n=15, m=3, seed=42)

        results = {
            "LPT": lpt(inst),
            "SPT": spt_parallel(inst),
            "MULTIFIT": multifit(inst),
            "List": list_scheduling(inst),
            "MIP": mip_makespan(inst, time_limit=10.0),
            "GA": genetic_algorithm(inst, max_generations=100, seed=42),
        }

        for name, sol in results.items():
            all_jobs = sorted(j for jobs in sol.assignment for j in jobs)
            assert all_jobs == list(range(15)), f"{name}: invalid assignment"
            recomputed = compute_makespan(inst, sol.assignment)
            assert abs(sol.makespan - recomputed) < 1e-9, (
                f"{name}: makespan mismatch"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
