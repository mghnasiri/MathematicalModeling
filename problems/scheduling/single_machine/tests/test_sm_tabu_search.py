"""
Tests for Single Machine Tabu Search.

Run: python -m pytest problems/scheduling/single_machine/tests/test_sm_tabu_search.py -v
"""

import sys
import os
import importlib.util
import pytest
import numpy as np

_this_dir = os.path.dirname(os.path.abspath(__file__))
_sm_dir = os.path.dirname(_this_dir)


def _load_module(name: str, filepath: str):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_instance_mod = _load_module("sm_inst_ts_test", os.path.join(_sm_dir, "instance.py"))
_ts_mod = _load_module("sm_ts_test", os.path.join(_sm_dir, "metaheuristics", "tabu_search.py"))
_atc_mod = _load_module("sm_atc_ts_test", os.path.join(_sm_dir, "heuristics", "apparent_tardiness_cost.py"))
_rules_mod = _load_module("sm_rules_ts_test", os.path.join(_sm_dir, "heuristics", "dispatching_rules.py"))

SingleMachineInstance = _instance_mod.SingleMachineInstance
compute_weighted_tardiness = _instance_mod.compute_weighted_tardiness
compute_total_tardiness = _instance_mod.compute_total_tardiness
tabu_search = _ts_mod.tabu_search
atc = _atc_mod.atc
edd = _rules_mod.edd


class TestSingleMachineTabuSearch:
    """Test Tabu Search for single machine scheduling."""

    def test_returns_valid_sequence(self):
        inst = SingleMachineInstance.random(n=10, seed=42)
        sol = tabu_search(inst, max_iterations=100, seed=42)
        assert sorted(sol.sequence) == list(range(10))

    def test_weighted_tardiness_correct(self):
        inst = SingleMachineInstance.random(n=10, seed=42)
        sol = tabu_search(inst, objective="weighted_tardiness", max_iterations=100, seed=42)
        expected = compute_weighted_tardiness(inst, sol.sequence)
        assert sol.objective_value == expected

    def test_total_tardiness_correct(self):
        inst = SingleMachineInstance.random(n=10, seed=42)
        sol = tabu_search(inst, objective="total_tardiness", max_iterations=100, seed=42)
        expected = compute_total_tardiness(inst, sol.sequence)
        assert sol.objective_value == expected

    def test_no_worse_than_atc(self):
        inst = SingleMachineInstance.random(n=15, seed=42)
        atc_sol = atc(inst)
        ts_sol = tabu_search(
            inst, objective="weighted_tardiness",
            max_iterations=500, seed=42,
        )
        assert ts_sol.objective_value <= atc_sol.objective_value

    def test_no_worse_than_edd_for_total_tardiness(self):
        inst = SingleMachineInstance.random(n=15, seed=42)
        edd_sol = edd(inst)
        edd_tt = compute_total_tardiness(inst, edd_sol.sequence)
        ts_sol = tabu_search(
            inst, objective="total_tardiness",
            max_iterations=500, seed=42,
        )
        assert ts_sol.objective_value <= edd_tt

    def test_deterministic_with_seed(self):
        inst = SingleMachineInstance.random(n=10, seed=42)
        sol_a = tabu_search(inst, max_iterations=100, seed=123)
        sol_b = tabu_search(inst, max_iterations=100, seed=123)
        assert sol_a.objective_value == sol_b.objective_value
        assert sol_a.sequence == sol_b.sequence

    def test_single_job(self):
        inst = SingleMachineInstance.from_arrays(
            processing_times=[5], due_dates=[3], weights=[2],
        )
        sol = tabu_search(inst, max_iterations=10, seed=42)
        assert sol.sequence == [0]

    def test_two_jobs(self):
        inst = SingleMachineInstance.from_arrays(
            processing_times=[3, 5], due_dates=[4, 6], weights=[1, 1],
        )
        sol = tabu_search(inst, max_iterations=50, seed=42)
        assert sorted(sol.sequence) == [0, 1]

    def test_swap_neighborhood(self):
        inst = SingleMachineInstance.random(n=10, seed=42)
        sol = tabu_search(
            inst, neighborhood="swap", max_iterations=100, seed=42,
        )
        assert sorted(sol.sequence) == list(range(10))

    def test_insert_neighborhood(self):
        inst = SingleMachineInstance.random(n=10, seed=42)
        sol = tabu_search(
            inst, neighborhood="insert", max_iterations=100, seed=42,
        )
        assert sorted(sol.sequence) == list(range(10))

    def test_time_limit(self):
        inst = SingleMachineInstance.random(n=15, seed=42)
        sol = tabu_search(inst, time_limit=2.0, seed=42)
        assert sorted(sol.sequence) == list(range(15))

    def test_objective_name(self):
        inst = SingleMachineInstance.random(n=10, seed=42)
        sol_wt = tabu_search(inst, objective="weighted_tardiness", max_iterations=10, seed=42)
        assert sol_wt.objective_name == "ΣwjTj"
        sol_tt = tabu_search(inst, objective="total_tardiness", max_iterations=10, seed=42)
        assert sol_tt.objective_name == "ΣTj"
