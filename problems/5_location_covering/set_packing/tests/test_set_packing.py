"""
Test suite for Maximum Weight Set Packing Problem.

Tests cover:
- Instance creation and validation
- Greedy weight and density heuristics
- Disjointness checking
- Edge cases
"""

from __future__ import annotations

import os
import sys
import pytest
import numpy as np
import importlib.util

# -- Module loading ------------------------------------------------------------

_base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_mod(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_inst_mod = _load_mod("sp_instance_test", os.path.join(_base_dir, "instance.py"))
_greedy_mod = _load_mod(
    "sp_greedy_test", os.path.join(_base_dir, "heuristics", "greedy_sp.py")
)

SetPackingInstance = _inst_mod.SetPackingInstance
SetPackingSolution = _inst_mod.SetPackingSolution
small_sp_3 = _inst_mod.small_sp_3
disjoint_4 = _inst_mod.disjoint_4
conflict_5 = _inst_mod.conflict_5

greedy_weight = _greedy_mod.greedy_weight
greedy_density = _greedy_mod.greedy_density


# -- Fixtures ------------------------------------------------------------------


@pytest.fixture
def inst3():
    return small_sp_3()


@pytest.fixture
def inst_disj():
    return disjoint_4()


@pytest.fixture
def inst_conf():
    return conflict_5()


# -- Instance tests ------------------------------------------------------------


class TestSetPackingInstance:
    def test_create_basic(self, inst3):
        assert inst3.m == 3
        assert inst3.n_elements == 5

    def test_random_instance(self):
        inst = SetPackingInstance.random(10, m=6, seed=42)
        assert inst.m == 6
        assert inst.n_elements == 10

    def test_invalid_weights_shape(self):
        with pytest.raises(ValueError):
            SetPackingInstance(
                n_elements=3,
                sets=[frozenset({0, 1})],
                weights=np.array([1.0, 2.0]),
            )

    def test_invalid_element(self):
        with pytest.raises(ValueError):
            SetPackingInstance(
                n_elements=3,
                sets=[frozenset({0, 5})],
                weights=np.array([1.0]),
            )

    def test_are_disjoint(self, inst3):
        assert inst3.are_disjoint([0, 1])  # {0,1} and {2,3}
        assert not inst3.are_disjoint([0, 2])  # {0,1} and {1,4} share 1

    def test_total_weight(self, inst3):
        assert abs(inst3.total_weight([0, 1]) - 15.0) < 1e-10


# -- Greedy weight tests ------------------------------------------------------


class TestGreedyWeight:
    def test_small_valid(self, inst3):
        sol = greedy_weight(inst3)
        assert inst3.are_disjoint(sol.selected)

    def test_disjoint_all_selected(self, inst_disj):
        sol = greedy_weight(inst_disj)
        assert len(sol.selected) == 4
        assert abs(sol.total_weight - 23.0) < 1e-10

    def test_conflict_valid(self, inst_conf):
        sol = greedy_weight(inst_conf)
        assert inst_conf.are_disjoint(sol.selected)

    def test_random_valid(self):
        inst = SetPackingInstance.random(15, m=10, seed=99)
        sol = greedy_weight(inst)
        assert inst.are_disjoint(sol.selected)


# -- Greedy density tests -----------------------------------------------------


class TestGreedyDensity:
    def test_small_valid(self, inst3):
        sol = greedy_density(inst3)
        assert inst3.are_disjoint(sol.selected)

    def test_disjoint_all_selected(self, inst_disj):
        sol = greedy_density(inst_disj)
        assert len(sol.selected) == 4

    def test_repr(self, inst3):
        sol = greedy_weight(inst3)
        r = repr(sol)
        assert "SetPackingSolution" in r
