"""Tests for Tabu Search on Flexible Job Shop Scheduling."""

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


_inst = _load_mod("fjsp_instance_test_ts", os.path.join(_parent_dir, "instance.py"))
FlexibleJobShopInstance = _inst.FlexibleJobShopInstance
FlexibleJobShopSolution = _inst.FlexibleJobShopSolution
validate_solution = _inst.validate_solution
compute_makespan = _inst.compute_makespan

_ts = _load_mod(
    "fjsp_ts_test",
    os.path.join(_parent_dir, "metaheuristics", "tabu_search.py"),
)
tabu_search = _ts.tabu_search

_disp = _load_mod(
    "fjsp_disp_test_ts",
    os.path.join(_parent_dir, "heuristics", "dispatching_rules.py"),
)
dispatching_rule = _disp.dispatching_rule


def _small_fjsp():
    """Small 3-job, 2-machine total FJSP."""
    return FlexibleJobShopInstance(
        n=3, m=2,
        jobs=[
            [{0: 3, 1: 5}],
            [{0: 4, 1: 2}, {0: 3, 1: 6}],
            [{0: 5, 1: 4}],
        ],
    )


class TestFJSPTSValidity:
    """Test that TS produces valid solutions."""

    def test_small_valid(self):
        inst = _small_fjsp()
        sol = tabu_search(inst, seed=42)
        valid, errors = validate_solution(inst, sol.assignments, sol.start_times)
        assert valid, f"Invalid solution: {errors}"

    def test_random_partial_valid(self):
        inst = FlexibleJobShopInstance.random(
            n=4, m=3, flexibility=0.5, seed=42
        )
        sol = tabu_search(inst, seed=42)
        valid, errors = validate_solution(inst, sol.assignments, sol.start_times)
        assert valid, f"Invalid solution: {errors}"

    def test_random_total_valid(self):
        inst = FlexibleJobShopInstance.random_total(n=4, m=3, seed=42)
        sol = tabu_search(inst, seed=42)
        valid, errors = validate_solution(inst, sol.assignments, sol.start_times)
        assert valid, f"Invalid solution: {errors}"


class TestFJSPTSQuality:
    """Test solution quality."""

    def test_ts_competitive_with_dispatching(self):
        inst = FlexibleJobShopInstance.random_total(n=4, m=3, seed=42)
        disp_sol = dispatching_rule(inst, priority_rule="spt", machine_rule="ect")
        ts_sol = tabu_search(inst, max_iterations=1000, seed=42)
        assert ts_sol.makespan <= disp_sol.makespan + 5

    def test_makespan_matches_computed(self):
        inst = FlexibleJobShopInstance.random(n=4, m=3, flexibility=0.6, seed=42)
        sol = tabu_search(inst, seed=42)
        actual_ms = compute_makespan(inst, sol.assignments, sol.start_times)
        assert sol.makespan == actual_ms


class TestFJSPTSDeterminism:
    """Test deterministic behavior with seed."""

    def test_same_seed_same_result(self):
        inst = FlexibleJobShopInstance.random(n=4, m=3, flexibility=0.6, seed=42)
        sol1 = tabu_search(inst, seed=42)
        sol2 = tabu_search(inst, seed=42)
        assert sol1.makespan == sol2.makespan

    def test_different_seed_both_valid(self):
        inst = FlexibleJobShopInstance.random(n=4, m=3, flexibility=0.6, seed=42)
        sol1 = tabu_search(inst, seed=1)
        sol2 = tabu_search(inst, seed=999)
        valid1, _ = validate_solution(inst, sol1.assignments, sol1.start_times)
        valid2, _ = validate_solution(inst, sol2.assignments, sol2.start_times)
        assert valid1 and valid2


class TestFJSPTSEdgeCases:
    """Test edge cases."""

    def test_single_job_single_op(self):
        inst = FlexibleJobShopInstance(
            n=1, m=2,
            jobs=[[{0: 5, 1: 3}]],
        )
        sol = tabu_search(inst, seed=42)
        valid, errors = validate_solution(inst, sol.assignments, sol.start_times)
        assert valid, f"Invalid: {errors}"
        assert sol.makespan <= 5

    def test_time_limit(self):
        inst = FlexibleJobShopInstance.random(n=5, m=3, flexibility=0.6, seed=42)
        sol = tabu_search(
            inst, max_iterations=1000000, time_limit=0.5, seed=42
        )
        valid, errors = validate_solution(inst, sol.assignments, sol.start_times)
        assert valid, f"Invalid: {errors}"
