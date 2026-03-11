"""
Tests for Parallel Machine Simulated Annealing.

Run: python -m pytest problems/scheduling/parallel_machine/tests/test_pm_sa.py -v
"""

import sys
import os
import importlib.util
import pytest
import numpy as np

_parallel_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_module(name: str, filepath: str):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_inst_mod = _load_module("pm_inst_sa_test", os.path.join(_parallel_dir, "instance.py"))
_lpt_mod = _load_module("pm_lpt_sa_test", os.path.join(_parallel_dir, "heuristics", "lpt.py"))
_sa_mod = _load_module("pm_sa_test", os.path.join(_parallel_dir, "metaheuristics", "simulated_annealing.py"))

ParallelMachineInstance = _inst_mod.ParallelMachineInstance
compute_makespan = _inst_mod.compute_makespan
lpt = _lpt_mod.lpt
simulated_annealing = _sa_mod.simulated_annealing


class TestParallelMachineSA:
    """Test Simulated Annealing for parallel machine scheduling."""

    def test_returns_valid_assignment(self):
        inst = ParallelMachineInstance.random_identical(n=10, m=3, seed=42)
        sol = simulated_annealing(inst, max_iterations=500, seed=42)
        all_jobs = sorted(j for machine_jobs in sol.assignment for j in machine_jobs)
        assert all_jobs == list(range(10))
        assert len(sol.assignment) == 3

    def test_makespan_correct(self):
        inst = ParallelMachineInstance.random_identical(n=10, m=3, seed=42)
        sol = simulated_annealing(inst, max_iterations=500, seed=42)
        actual_ms = compute_makespan(inst, sol.assignment)
        assert abs(sol.makespan - actual_ms) < 1e-6

    def test_no_worse_than_lpt(self):
        inst = ParallelMachineInstance.random_identical(n=20, m=4, seed=42)
        lpt_sol = lpt(inst)
        sa_sol = simulated_annealing(inst, max_iterations=2000, seed=42)
        assert sa_sol.makespan <= lpt_sol.makespan + 1e-6

    def test_deterministic_with_seed(self):
        inst = ParallelMachineInstance.random_identical(n=10, m=3, seed=42)
        sol_a = simulated_annealing(inst, max_iterations=200, seed=123)
        sol_b = simulated_annealing(inst, max_iterations=200, seed=123)
        assert abs(sol_a.makespan - sol_b.makespan) < 1e-6

    def test_uniform_machines(self):
        inst = ParallelMachineInstance.random_uniform(n=10, m=3, seed=42)
        sol = simulated_annealing(inst, max_iterations=500, seed=42)
        all_jobs = sorted(j for mj in sol.assignment for j in mj)
        assert all_jobs == list(range(10))

    def test_unrelated_machines(self):
        inst = ParallelMachineInstance.random_unrelated(n=10, m=3, seed=42)
        sol = simulated_annealing(inst, max_iterations=500, seed=42)
        all_jobs = sorted(j for mj in sol.assignment for j in mj)
        assert all_jobs == list(range(10))

    def test_single_machine(self):
        inst = ParallelMachineInstance.random_identical(n=5, m=1, seed=42)
        sol = simulated_annealing(inst, max_iterations=100, seed=42)
        assert sorted(sol.assignment[0]) == list(range(5))

    def test_single_job(self):
        inst = ParallelMachineInstance.random_identical(n=1, m=3, seed=42)
        sol = simulated_annealing(inst, max_iterations=100, seed=42)
        all_jobs = [j for mj in sol.assignment for j in mj]
        assert all_jobs == [0]

    def test_time_limit(self):
        inst = ParallelMachineInstance.random_identical(n=15, m=4, seed=42)
        sol = simulated_annealing(inst, time_limit=2.0, seed=42)
        all_jobs = sorted(j for mj in sol.assignment for j in mj)
        assert all_jobs == list(range(15))

    def test_machine_loads_populated(self):
        inst = ParallelMachineInstance.random_identical(n=10, m=3, seed=42)
        sol = simulated_annealing(inst, max_iterations=200, seed=42)
        assert sol.machine_loads is not None
        assert len(sol.machine_loads) == 3

    def test_improves_over_iterations(self):
        inst = ParallelMachineInstance.random_identical(n=20, m=4, seed=42)
        sol_short = simulated_annealing(inst, max_iterations=50, seed=42)
        sol_long = simulated_annealing(inst, max_iterations=3000, seed=42)
        assert sol_long.makespan <= sol_short.makespan + 1e-6
