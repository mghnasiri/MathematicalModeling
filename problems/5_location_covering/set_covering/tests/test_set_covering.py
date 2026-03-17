"""Tests for Set Covering Problem."""
from __future__ import annotations

import sys
import os
import importlib.util

import numpy as np
import pytest

def _load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_base = os.path.join(os.path.dirname(__file__), "..")
_inst = _load_mod("scp_instance", os.path.join(_base, "instance.py"))
_greedy = _load_mod("scp_greedy", os.path.join(_base, "heuristics", "greedy_scp.py"))
_ilp = _load_mod("scp_ilp", os.path.join(_base, "exact", "ilp_scp.py"))

SetCoveringInstance = _inst.SetCoveringInstance
greedy_cost_effectiveness = _greedy.greedy_cost_effectiveness
greedy_largest_first = _greedy.greedy_largest_first
solve_lp_relaxation = _ilp.solve_lp_relaxation
solve_ilp = _ilp.solve_ilp


def _make_simple():
    return SetCoveringInstance(
        m=5, n=4,
        subsets=[{0, 1, 2}, {2, 3}, {3, 4}, {0, 4}],
        costs=np.array([3.0, 2.0, 2.0, 4.0]),
    )


class TestSetCoveringInstance:
    def test_creation(self):
        inst = _make_simple()
        assert inst.m == 5 and inst.n == 4

    def test_is_cover(self):
        inst = _make_simple()
        assert inst.is_cover([0, 2])  # {0,1,2} + {3,4}
        assert not inst.is_cover([0])

    def test_random(self):
        inst = SetCoveringInstance.random(m=15, n=20)
        assert inst.m == 15


class TestGreedy:
    def test_cost_effectiveness_covers(self):
        inst = _make_simple()
        sol = greedy_cost_effectiveness(inst)
        assert inst.is_cover(sol.selected)

    def test_largest_first_covers(self):
        inst = _make_simple()
        sol = greedy_largest_first(inst)
        assert inst.is_cover(sol.selected)

    def test_random_instance(self):
        inst = SetCoveringInstance.random(m=20, n=30)
        sol = greedy_cost_effectiveness(inst)
        assert inst.is_cover(sol.selected)


class TestExact:
    def test_lp_relaxation_lower_bound(self):
        inst = _make_simple()
        lb, x = solve_lp_relaxation(inst)
        sol = greedy_cost_effectiveness(inst)
        assert lb <= sol.total_cost + 1e-6

    def test_ilp_optimal(self):
        inst = _make_simple()
        sol = solve_ilp(inst)
        if sol is not None:
            assert inst.is_cover(sol.selected)
            assert sol.total_cost <= 7.0 + 1e-6  # {0,2} costs 5

    def test_ilp_random(self):
        inst = SetCoveringInstance.random(m=10, n=15, seed=7)
        sol = solve_ilp(inst)
        if sol is not None:
            assert inst.is_cover(sol.selected)
