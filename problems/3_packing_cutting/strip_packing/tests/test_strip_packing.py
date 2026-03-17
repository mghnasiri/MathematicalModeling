"""
Test suite for 2D Strip Packing Problem.

Tests cover:
- Instance creation and validation
- Bottom-Left and NFDH level algorithms
- Solution validation (no overlaps, all items packed, correct height)
"""

from __future__ import annotations

import os
import sys
import pytest
import numpy as np
import importlib.util

# ── Module loading ───────────────────────────────────────────────────────────

_base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_module(name: str, filepath: str):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_inst_mod = _load_module(
    "spp_instance_test", os.path.join(_base_dir, "instance.py")
)
_level_mod = _load_module(
    "spp_level_test",
    os.path.join(_base_dir, "heuristics", "level_algorithms.py"),
)

StripPackingInstance = _inst_mod.StripPackingInstance
StripPackingSolution = _inst_mod.StripPackingSolution
StripPackingPlacement = _inst_mod.StripPackingPlacement
validate_solution = _inst_mod.validate_solution
small_spp_3 = _inst_mod.small_spp_3
uniform_spp_6 = _inst_mod.uniform_spp_6
wide_items_4 = _inst_mod.wide_items_4

bottom_left = _level_mod.bottom_left
nfdh_strip = _level_mod.nfdh_strip


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def inst3():
    return small_spp_3()


@pytest.fixture
def inst6():
    return uniform_spp_6()


@pytest.fixture
def inst4_wide():
    return wide_items_4()


@pytest.fixture
def random_inst():
    return StripPackingInstance.random(12, seed=42)


# ── Instance tests ───────────────────────────────────────────────────────────


class TestStripPackingInstance:
    def test_create_basic(self, inst3):
        assert inst3.n == 3
        assert inst3.strip_width == 10.0
        assert inst3.widths.shape == (3,)

    def test_random_instance(self):
        inst = StripPackingInstance.random(15, seed=123)
        assert inst.n == 15
        assert np.all(inst.widths <= inst.strip_width)

    def test_area_lower_bound(self, inst6):
        # 6 items of 3x3 = 54, W=10 => LB = 5.4
        assert abs(inst6.area_lower_bound() - 5.4) < 1e-10

    def test_max_height_lower_bound(self, inst3):
        # max height is 5.0
        assert inst3.max_height_lower_bound() == 5.0

    def test_invalid_width_exceeds_strip(self):
        with pytest.raises(ValueError):
            StripPackingInstance(
                n=1,
                widths=np.array([15.0]),
                heights=np.array([5.0]),
                strip_width=10.0,
            )

    def test_zero_strip_width(self):
        with pytest.raises(ValueError):
            StripPackingInstance(
                n=1,
                widths=np.array([5.0]),
                heights=np.array([5.0]),
                strip_width=0.0,
            )


# ── Validation tests ────────────────────────────────────────────────────────


class TestValidation:
    def test_valid_solution(self, inst3):
        sol = bottom_left(inst3)
        valid, errors = validate_solution(inst3, sol)
        assert valid, errors

    def test_missing_item(self, inst3):
        sol = StripPackingSolution(
            placements=[StripPackingPlacement(0, 0.0, 0.0)],
            height=4.0,
        )
        valid, errors = validate_solution(inst3, sol)
        assert not valid


# ── Bottom-Left tests ───────────────────────────────────────────────────────


class TestBottomLeft:
    def test_feasible_small(self, inst3):
        sol = bottom_left(inst3)
        valid, errors = validate_solution(inst3, sol)
        assert valid, errors

    def test_feasible_uniform(self, inst6):
        sol = bottom_left(inst6)
        valid, errors = validate_solution(inst6, sol)
        assert valid, errors

    def test_height_at_least_lb(self, inst3):
        sol = bottom_left(inst3)
        assert sol.height >= inst3.area_lower_bound() - 1e-10
        assert sol.height >= inst3.max_height_lower_bound() - 1e-10

    def test_wide_items_stacked(self, inst4_wide):
        sol = bottom_left(inst4_wide)
        valid, errors = validate_solution(inst4_wide, sol)
        assert valid, errors
        # Wide items mostly stack, height should be sum of heights
        assert sol.height >= 1.0  # at least min item height

    def test_random_feasible(self, random_inst):
        sol = bottom_left(random_inst)
        valid, errors = validate_solution(random_inst, sol)
        assert valid, errors

    def test_single_item(self):
        inst = StripPackingInstance(
            n=1,
            widths=np.array([5.0]),
            heights=np.array([3.0]),
            strip_width=10.0,
        )
        sol = bottom_left(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, errors
        assert abs(sol.height - 3.0) < 1e-10


# ── NFDH tests ───────────────────────────────────────────────────────────────


class TestNFDH:
    def test_feasible_small(self, inst3):
        sol = nfdh_strip(inst3)
        valid, errors = validate_solution(inst3, sol)
        assert valid, errors

    def test_feasible_uniform(self, inst6):
        sol = nfdh_strip(inst6)
        valid, errors = validate_solution(inst6, sol)
        assert valid, errors
        # 6 items 3x3, W=10: row1 has 3 items (9<=10), row2 has 3 items. h=6
        assert abs(sol.height - 6.0) < 1e-10

    def test_height_at_least_lb(self, random_inst):
        sol = nfdh_strip(random_inst)
        assert sol.height >= random_inst.area_lower_bound() - 1e-10
        assert sol.height >= random_inst.max_height_lower_bound() - 1e-10

    def test_random_feasible(self, random_inst):
        sol = nfdh_strip(random_inst)
        valid, errors = validate_solution(random_inst, sol)
        assert valid, errors
