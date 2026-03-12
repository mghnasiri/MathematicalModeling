"""Tests for Job Sequencing with Deadlines."""
from __future__ import annotations

import sys
import os
import importlib.util

import numpy as np
import pytest


def _load_mod(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_base = os.path.join(os.path.dirname(__file__), "..")
_inst_mod = _load_mod("js_instance_test", os.path.join(_base, "instance.py"))
_heur_mod = _load_mod(
    "js_greedy_test", os.path.join(_base, "heuristics", "greedy_js.py")
)

JobSequencingInstance = _inst_mod.JobSequencingInstance
JobSequencingSolution = _inst_mod.JobSequencingSolution
greedy_profit = _heur_mod.greedy_profit
greedy_edd = _heur_mod.greedy_edd


class TestInstance:

    def test_unit_processing(self):
        inst = JobSequencingInstance.unit_processing(5)
        assert inst.n == 5
        assert all(p == 1.0 for p in inst.processing_times)

    def test_is_feasible(self):
        inst = JobSequencingInstance.unit_processing(5)
        # Job 0 has deadline 1, job 2 has deadline 2
        assert inst.is_feasible([0])
        assert inst.is_feasible([0, 2])

    def test_total_profit(self):
        inst = JobSequencingInstance.unit_processing(5)
        assert inst.total_profit([0, 1]) == inst.profits[0] + inst.profits[1]

    def test_random(self):
        inst = JobSequencingInstance.random(n=8, seed=42)
        assert inst.n == 8


class TestGreedyProfit:

    def test_feasible_result(self):
        inst = JobSequencingInstance.unit_processing(5)
        sol = greedy_profit(inst)
        assert inst.is_feasible(sol.sequence)
        assert sol.total_profit > 0

    def test_selects_high_profit(self):
        inst = JobSequencingInstance.unit_processing(5)
        sol = greedy_profit(inst)
        # Should select profitable jobs
        assert sol.n_selected >= 1

    def test_random_instance(self):
        inst = JobSequencingInstance.random(n=10, seed=7)
        sol = greedy_profit(inst)
        assert inst.is_feasible(sol.sequence)
        assert sol.total_profit >= 0


class TestGreedyEDD:

    def test_feasible_result(self):
        inst = JobSequencingInstance.unit_processing(5)
        sol = greedy_edd(inst)
        assert inst.is_feasible(sol.sequence)

    def test_edd_ordered(self):
        inst = JobSequencingInstance.unit_processing(5)
        sol = greedy_edd(inst)
        # EDD greedy produces deadline-sorted sequence
        for i in range(len(sol.sequence) - 1):
            j1, j2 = sol.sequence[i], sol.sequence[i + 1]
            assert inst.deadlines[j1] <= inst.deadlines[j2] + 1e-10

    def test_profit_vs_edd(self):
        inst = JobSequencingInstance.unit_processing(5)
        sol_p = greedy_profit(inst)
        sol_e = greedy_edd(inst)
        # Both should be feasible
        assert inst.is_feasible(sol_p.sequence)
        assert inst.is_feasible(sol_e.sequence)

    def test_repr(self):
        inst = JobSequencingInstance.unit_processing(5)
        sol = greedy_profit(inst)
        assert "profit" in repr(sol)
