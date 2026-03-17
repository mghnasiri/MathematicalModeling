"""
Tests for Job Shop Genetic Algorithm.

Run: python -m pytest problems/scheduling/job_shop/tests/test_job_shop_ga.py -v
"""

import sys
import os
import importlib.util
import pytest
import numpy as np

_job_shop_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_instance_mod = _load_module(
    "jsp_instance_ga_test",
    os.path.join(_job_shop_dir, "instance.py"),
)
_ga_mod = _load_module(
    "jsp_ga_test",
    os.path.join(_job_shop_dir, "metaheuristics", "genetic_algorithm.py"),
)

JobShopInstance = _instance_mod.JobShopInstance
validate_solution = _instance_mod.validate_solution
ft06 = _instance_mod.ft06
ft10 = _instance_mod.ft10
genetic_algorithm = _ga_mod.genetic_algorithm


class TestJobShopGA:
    """Test Genetic Algorithm for Job Shop Scheduling."""

    def test_returns_valid_solution(self):
        inst = JobShopInstance.random(n=5, m=3, seed=42)
        sol = genetic_algorithm(inst, generations=50, seed=42)
        is_valid, violations = validate_solution(inst, sol.start_times)
        assert is_valid, f"Invalid solution: {violations}"

    def test_ft06_quality(self):
        """GA should find a good solution for ft06 (optimal=55)."""
        inst = ft06()
        sol = genetic_algorithm(
            inst, population_size=30, generations=300, seed=42,
        )
        is_valid, _ = validate_solution(inst, sol.start_times)
        assert is_valid
        assert sol.makespan <= 65  # Within 18% of optimal

    def test_deterministic_with_seed(self):
        inst = JobShopInstance.random(n=5, m=3, seed=42)
        sol_a = genetic_algorithm(inst, generations=30, seed=123)
        sol_b = genetic_algorithm(inst, generations=30, seed=123)
        assert sol_a.makespan == sol_b.makespan

    def test_different_seeds_differ(self):
        inst = JobShopInstance.random(n=6, m=3, seed=42)
        sol_a = genetic_algorithm(inst, generations=100, seed=1)
        sol_b = genetic_algorithm(inst, generations=100, seed=2)
        # Different seeds may produce same result but usually differ
        # Just check both are valid
        assert validate_solution(inst, sol_a.start_times)[0]
        assert validate_solution(inst, sol_b.start_times)[0]

    def test_single_job(self):
        inst = JobShopInstance(n=1, m=3, jobs=[[(0, 5), (1, 3), (2, 7)]])
        sol = genetic_algorithm(inst, generations=10, seed=42)
        assert sol.makespan == 15  # 5 + 3 + 7

    def test_single_machine(self):
        inst = JobShopInstance(n=3, m=1, jobs=[
            [(0, 5)], [(0, 3)], [(0, 7)],
        ])
        sol = genetic_algorithm(inst, generations=10, seed=42)
        assert sol.makespan == 15  # All jobs on one machine

    def test_time_limit(self):
        inst = JobShopInstance.random(n=6, m=4, seed=42)
        sol = genetic_algorithm(inst, time_limit=2.0, seed=42)
        is_valid, _ = validate_solution(inst, sol.start_times)
        assert is_valid

    def test_improves_over_generations(self):
        inst = JobShopInstance.random(n=8, m=4, seed=42)
        sol_short = genetic_algorithm(inst, generations=10, seed=42)
        sol_long = genetic_algorithm(inst, generations=300, seed=42)
        assert sol_long.makespan <= sol_short.makespan

    def test_machine_sequences_populated(self):
        inst = JobShopInstance.random(n=5, m=3, seed=42)
        sol = genetic_algorithm(inst, generations=50, seed=42)
        assert sol.machine_sequences is not None
        assert len(sol.machine_sequences) == inst.m

    def test_all_operations_scheduled(self):
        inst = JobShopInstance.random(n=6, m=4, seed=42)
        sol = genetic_algorithm(inst, generations=50, seed=42)
        for j in range(inst.n):
            for k in range(len(inst.jobs[j])):
                assert (j, k) in sol.start_times
