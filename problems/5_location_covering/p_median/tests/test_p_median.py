"""
Test suite for p-Median Problem.

Tests cover:
- Instance creation and validation
- Greedy heuristic
- Teitz-Bart interchange
"""

from __future__ import annotations

import os
import sys
import pytest
import numpy as np
import importlib.util

_base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_module(name: str, filepath: str):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_inst_mod = _load_module("pm_inst_test", os.path.join(_base_dir, "instance.py"))
_gr_mod = _load_module("pm_gr_test", os.path.join(_base_dir, "heuristics", "greedy_pmedian.py"))

PMedianInstance = _inst_mod.PMedianInstance
PMedianSolution = _inst_mod.PMedianSolution
validate_solution = _inst_mod.validate_solution
small_pmedian_6_2 = _inst_mod.small_pmedian_6_2

greedy_pmedian = _gr_mod.greedy_pmedian
interchange = _gr_mod.interchange


@pytest.fixture
def inst6():
    return small_pmedian_6_2()


@pytest.fixture
def random_inst():
    return PMedianInstance.random(15, m=15, p=3, seed=42)


class TestPMedianInstance:
    def test_create_basic(self, inst6):
        assert inst6.n == 6
        assert inst6.m == 6
        assert inst6.p == 2

    def test_random_instance(self):
        inst = PMedianInstance.random(10, p=3, seed=123)
        assert inst.n == 10
        assert inst.p == 3

    def test_invalid_p(self):
        with pytest.raises(ValueError):
            PMedianInstance(
                n=3, m=3, p=0,
                weights=np.ones(3),
                distance_matrix=np.zeros((3, 3)))

    def test_p_exceeds_m(self):
        with pytest.raises(ValueError):
            PMedianInstance(
                n=3, m=3, p=4,
                weights=np.ones(3),
                distance_matrix=np.zeros((3, 3)))


class TestValidation:
    def test_valid_solution(self, inst6):
        sol = greedy_pmedian(inst6)
        valid, errors = validate_solution(inst6, sol)
        assert valid, errors

    def test_wrong_p(self, inst6):
        sol = PMedianSolution(
            open_facilities=[0],  # Only 1, need 2
            assignments=[0] * 6, cost=0.0)
        valid, errors = validate_solution(inst6, sol)
        assert not valid


class TestGreedyPMedian:
    def test_feasible(self, inst6):
        sol = greedy_pmedian(inst6)
        valid, errors = validate_solution(inst6, sol)
        assert valid, errors
        assert len(sol.open_facilities) == inst6.p

    def test_random_instance(self, random_inst):
        sol = greedy_pmedian(random_inst)
        valid, errors = validate_solution(random_inst, sol)
        assert valid, errors

    def test_cost_positive(self, inst6):
        sol = greedy_pmedian(inst6)
        assert sol.cost > 0


class TestInterchange:
    def test_feasible(self, inst6):
        sol = interchange(inst6)
        valid, errors = validate_solution(inst6, sol)
        assert valid, errors
        assert len(sol.open_facilities) == inst6.p

    def test_improves_greedy(self, random_inst):
        gr = greedy_pmedian(random_inst)
        tb = interchange(random_inst, initial=gr)
        assert tb.cost <= gr.cost + 1e-10

    def test_random_instance(self, random_inst):
        sol = interchange(random_inst)
        valid, errors = validate_solution(random_inst, sol)
        assert valid, errors


class TestCrossMethod:
    def test_all_valid(self, random_inst):
        methods = [greedy_pmedian, interchange]
        for method in methods:
            sol = method(random_inst)
            valid, errors = validate_solution(random_inst, sol)
            assert valid, errors
