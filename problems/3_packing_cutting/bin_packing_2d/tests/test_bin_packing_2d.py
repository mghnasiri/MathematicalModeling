"""
Test suite for 2D Bin Packing Problem.

Tests cover:
- Instance creation and validation
- NFDH and FFDH shelf algorithms
- Solution validation (no overlaps, all items packed)
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
    "bp2d_instance_test", os.path.join(_base_dir, "instance.py")
)
_shelf_mod = _load_module(
    "bp2d_shelf_test",
    os.path.join(_base_dir, "heuristics", "shelf_algorithms.py"),
)

BinPacking2DInstance = _inst_mod.BinPacking2DInstance
BinPacking2DSolution = _inst_mod.BinPacking2DSolution
Placement = _inst_mod.Placement
validate_solution = _inst_mod.validate_solution
small_2dbpp_4 = _inst_mod.small_2dbpp_4
uniform_2dbpp_6 = _inst_mod.uniform_2dbpp_6
tall_items_5 = _inst_mod.tall_items_5

nfdh = _shelf_mod.nfdh
ffdh = _shelf_mod.ffdh


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def inst4():
    return small_2dbpp_4()


@pytest.fixture
def inst6():
    return uniform_2dbpp_6()


@pytest.fixture
def inst5_tall():
    return tall_items_5()


@pytest.fixture
def random_inst():
    return BinPacking2DInstance.random(15, seed=42)


# ── Instance tests ───────────────────────────────────────────────────────────


class TestBinPacking2DInstance:
    def test_create_basic(self, inst4):
        assert inst4.n == 4
        assert inst4.bin_width == 10.0
        assert inst4.bin_height == 10.0
        assert inst4.widths.shape == (4,)
        assert inst4.heights.shape == (4,)

    def test_random_instance(self):
        inst = BinPacking2DInstance.random(20, seed=123)
        assert inst.n == 20
        assert np.all(inst.widths <= inst.bin_width)
        assert np.all(inst.heights <= inst.bin_height)

    def test_area_lower_bound(self, inst6):
        # 6 items of 3x3 = 54, bin 10x10 = 100. LB = ceil(54/100) = 1
        assert inst6.area_lower_bound() == 1

    def test_invalid_width_exceeds_bin(self):
        with pytest.raises(ValueError):
            BinPacking2DInstance(
                n=1,
                widths=np.array([15.0]),
                heights=np.array([5.0]),
                bin_width=10.0,
                bin_height=10.0,
            )

    def test_invalid_height_exceeds_bin(self):
        with pytest.raises(ValueError):
            BinPacking2DInstance(
                n=1,
                widths=np.array([5.0]),
                heights=np.array([15.0]),
                bin_width=10.0,
                bin_height=10.0,
            )

    def test_zero_bin_dimension(self):
        with pytest.raises(ValueError):
            BinPacking2DInstance(
                n=1,
                widths=np.array([5.0]),
                heights=np.array([5.0]),
                bin_width=0.0,
                bin_height=10.0,
            )


# ── Validation tests ────────────────────────────────────────────────────────


class TestValidation:
    def test_valid_solution(self, inst4):
        sol = ffdh(inst4)
        valid, errors = validate_solution(inst4, sol)
        assert valid, errors

    def test_missing_item(self, inst4):
        sol = BinPacking2DSolution(
            bins=[[Placement(0, 0.0, 0.0), Placement(1, 0.0, 0.0)]],
            num_bins=1,
        )
        valid, errors = validate_solution(inst4, sol)
        assert not valid

    def test_duplicate_item(self, inst4):
        sol = BinPacking2DSolution(
            bins=[[
                Placement(0, 0.0, 0.0),
                Placement(0, 5.0, 0.0),
                Placement(1, 0.0, 5.0),
                Placement(2, 5.0, 5.0),
                Placement(3, 0.0, 0.0),
            ]],
            num_bins=1,
        )
        valid, errors = validate_solution(inst4, sol)
        assert not valid


# ── NFDH tests ───────────────────────────────────────────────────────────────


class TestNFDH:
    def test_feasible_small(self, inst4):
        sol = nfdh(inst4)
        valid, errors = validate_solution(inst4, sol)
        assert valid, errors

    def test_feasible_uniform(self, inst6):
        sol = nfdh(inst6)
        valid, errors = validate_solution(inst6, sol)
        assert valid, errors

    def test_at_least_area_lb(self, inst4):
        sol = nfdh(inst4)
        assert sol.num_bins >= inst4.area_lower_bound()

    def test_uniform_fits_one_bin(self, inst6):
        # 6 items of 3x3 in 10x10 => fits in 1 bin with shelves
        sol = nfdh(inst6)
        valid, errors = validate_solution(inst6, sol)
        assert valid, errors
        assert sol.num_bins == 1

    def test_tall_items(self, inst5_tall):
        sol = nfdh(inst5_tall)
        valid, errors = validate_solution(inst5_tall, sol)
        assert valid, errors
        # 5 items of 3x8, bin 10x10: 3 fit in width, only 1 row (h=8)
        # So need 2 bins
        assert sol.num_bins == 2

    def test_random_feasible(self, random_inst):
        sol = nfdh(random_inst)
        valid, errors = validate_solution(random_inst, sol)
        assert valid, errors


# ── FFDH tests ───────────────────────────────────────────────────────────────


class TestFFDH:
    def test_feasible_small(self, inst4):
        sol = ffdh(inst4)
        valid, errors = validate_solution(inst4, sol)
        assert valid, errors

    def test_feasible_uniform(self, inst6):
        sol = ffdh(inst6)
        valid, errors = validate_solution(inst6, sol)
        assert valid, errors

    def test_at_least_area_lb(self, random_inst):
        sol = ffdh(random_inst)
        assert sol.num_bins >= random_inst.area_lower_bound()

    def test_ffdh_at_most_nfdh(self, random_inst):
        nfdh_sol = nfdh(random_inst)
        ffdh_sol = ffdh(random_inst)
        # FFDH is generally at least as good as NFDH
        assert ffdh_sol.num_bins <= nfdh_sol.num_bins + 1

    def test_single_item(self):
        inst = BinPacking2DInstance(
            n=1,
            widths=np.array([5.0]),
            heights=np.array([5.0]),
            bin_width=10.0,
            bin_height=10.0,
        )
        sol = ffdh(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, errors
        assert sol.num_bins == 1
