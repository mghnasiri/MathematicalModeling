"""Tests for Local Search on Single Machine Scheduling."""

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


_inst = _load_mod("sm_instance_test_ls", os.path.join(_parent_dir, "instance.py"))
SingleMachineInstance = _inst.SingleMachineInstance
SingleMachineSolution = _inst.SingleMachineSolution
compute_weighted_tardiness = _inst.compute_weighted_tardiness
compute_total_tardiness = _inst.compute_total_tardiness

_ls = _load_mod(
    "sm_ls_test",
    os.path.join(_parent_dir, "metaheuristics", "local_search.py"),
)
local_search = _ls.local_search

_atc_mod = _load_mod(
    "sm_atc_test_ls",
    os.path.join(_parent_dir, "heuristics", "apparent_tardiness_cost.py"),
)
atc = _atc_mod.atc


class TestSMLSValidity:
    """Test that LS produces valid solutions."""

    def test_random_instance_valid(self):
        inst = SingleMachineInstance.random(n=10, seed=42)
        sol = local_search(inst, seed=42)
        assert set(sol.sequence) == set(range(inst.n))

    def test_objective_matches(self):
        inst = SingleMachineInstance.random(n=10, seed=42)
        sol = local_search(inst, objective="weighted_tardiness", seed=42)
        actual = compute_weighted_tardiness(inst, sol.sequence)
        assert abs(sol.objective_value - actual) < 1e-6

    def test_total_tardiness_objective(self):
        inst = SingleMachineInstance.random(n=10, seed=42)
        sol = local_search(inst, objective="total_tardiness", seed=42)
        actual = compute_total_tardiness(inst, sol.sequence)
        assert abs(sol.objective_value - actual) < 1e-6


class TestSMLSQuality:
    """Test solution quality."""

    def test_ls_improves_or_matches_atc(self):
        inst = SingleMachineInstance.random(n=10, seed=42)
        atc_sol = atc(inst)
        ls_sol = local_search(inst, objective="weighted_tardiness")
        assert ls_sol.objective_value <= atc_sol.objective_value + 1e-6

    def test_both_neighborhoods(self):
        inst = SingleMachineInstance.random(n=10, seed=42)
        sol_swap = local_search(inst, neighborhood="swap")
        sol_both = local_search(inst, neighborhood="both")
        # "both" should be at least as good as "swap" alone
        assert sol_both.objective_value <= sol_swap.objective_value + 1e-6


class TestSMLSEdgeCases:
    """Test edge cases."""

    def test_single_job(self):
        inst = SingleMachineInstance.from_arrays(
            processing_times=[5],
            weights=[1],
            due_dates=[3],
        )
        sol = local_search(inst)
        assert sol.sequence == [0]

    def test_two_jobs(self):
        inst = SingleMachineInstance.from_arrays(
            processing_times=[3, 5],
            weights=[1, 1],
            due_dates=[4, 6],
        )
        sol = local_search(inst, neighborhood="swap")
        assert set(sol.sequence) == {0, 1}

    def test_time_limit(self):
        inst = SingleMachineInstance.random(n=15, seed=42)
        sol = local_search(
            inst, max_iterations=1000000, time_limit=0.5
        )
        assert set(sol.sequence) == set(range(inst.n))
