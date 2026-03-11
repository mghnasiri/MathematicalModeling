"""Tests for Simulated Annealing on RCPSP."""

from __future__ import annotations

import sys
import os
import importlib.util

import numpy as np
import pytest

_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_mod(name, filepath):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_inst = _load_mod("rcpsp_instance_test_sa", os.path.join(_parent_dir, "instance.py"))
RCPSPInstance = _inst.RCPSPInstance
RCPSPSolution = _inst.RCPSPSolution
validate_solution = _inst.validate_solution

_sa = _load_mod(
    "rcpsp_sa_test",
    os.path.join(_parent_dir, "metaheuristics", "simulated_annealing.py"),
)
simulated_annealing = _sa.simulated_annealing

_sgs = _load_mod(
    "rcpsp_sgs_test_sa",
    os.path.join(_parent_dir, "heuristics", "serial_sgs.py"),
)
serial_sgs = _sgs.serial_sgs


def _small_instance():
    """Small 4-activity RCPSP instance."""
    return RCPSPInstance.from_arrays(
        durations=[0, 3, 2, 4, 1, 0],
        resource_demands=[
            [0], [2], [3], [1], [2], [0],
        ],
        resource_capacities=[3],
        successors={
            0: [1, 2],
            1: [3],
            2: [4],
            3: [5],
            4: [5],
        },
    )


class TestRCPSPSAValidity:
    """Test that SA produces valid solutions."""

    def test_small_valid(self):
        inst = _small_instance()
        sol = simulated_annealing(inst, seed=42)
        valid, errors = validate_solution(inst, sol.start_times)
        assert valid, f"Invalid solution: {errors}"

    def test_random_instance_valid(self):
        inst = RCPSPInstance.random(n=10, num_resources=2, seed=42)
        sol = simulated_annealing(inst, seed=42)
        valid, errors = validate_solution(inst, sol.start_times)
        assert valid, f"Invalid solution: {errors}"

    def test_larger_random_valid(self):
        inst = RCPSPInstance.random(n=20, num_resources=3, seed=99)
        sol = simulated_annealing(inst, max_iterations=2000, seed=42)
        valid, errors = validate_solution(inst, sol.start_times)
        assert valid, f"Invalid solution: {errors}"


class TestRCPSPSAQuality:
    """Test solution quality."""

    def test_respects_critical_path(self):
        inst = _small_instance()
        sol = simulated_annealing(inst, seed=42)
        cp = inst.critical_path_length()
        assert sol.makespan >= cp

    def test_sa_not_worse_than_lft(self):
        inst = RCPSPInstance.random(n=10, num_resources=2, seed=42)
        lft_sol = serial_sgs(inst, priority_rule="lft")
        sa_sol = simulated_annealing(inst, max_iterations=3000, seed=42)
        assert sa_sol.makespan <= lft_sol.makespan

    def test_sa_improves_or_matches_heuristic(self):
        inst = RCPSPInstance.random(n=15, num_resources=2, seed=77)
        best_heuristic = min(
            serial_sgs(inst, priority_rule=rule).makespan
            for rule in ["lft", "est", "mts", "grpw"]
        )
        sa_sol = simulated_annealing(inst, max_iterations=5000, seed=42)
        # SA should match or beat the best heuristic
        assert sa_sol.makespan <= best_heuristic + 2


class TestRCPSPSADeterminism:
    """Test deterministic behavior with seed."""

    def test_same_seed_same_result(self):
        inst = RCPSPInstance.random(n=8, num_resources=2, seed=42)
        sol1 = simulated_annealing(inst, seed=42)
        sol2 = simulated_annealing(inst, seed=42)
        assert sol1.makespan == sol2.makespan

    def test_different_seed_both_valid(self):
        inst = RCPSPInstance.random(n=8, num_resources=2, seed=42)
        sol1 = simulated_annealing(inst, seed=1)
        sol2 = simulated_annealing(inst, seed=999)
        valid1, _ = validate_solution(inst, sol1.start_times)
        valid2, _ = validate_solution(inst, sol2.start_times)
        assert valid1 and valid2


class TestRCPSPSAEdgeCases:
    """Test edge cases."""

    def test_single_activity(self):
        inst = RCPSPInstance.from_arrays(
            durations=[0, 5, 0],
            resource_demands=[[0], [2], [0]],
            resource_capacities=[3],
            successors={0: [1], 1: [2]},
        )
        sol = simulated_annealing(inst, seed=42)
        valid, errors = validate_solution(inst, sol.start_times)
        assert valid, f"Invalid: {errors}"
        assert sol.makespan == 5

    def test_time_limit(self):
        inst = RCPSPInstance.random(n=15, num_resources=2, seed=42)
        sol = simulated_annealing(
            inst, max_iterations=1000000, time_limit=0.5, seed=42
        )
        valid, errors = validate_solution(inst, sol.start_times)
        assert valid, f"Invalid: {errors}"
