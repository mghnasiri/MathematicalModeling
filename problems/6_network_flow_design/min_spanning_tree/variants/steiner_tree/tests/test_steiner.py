"""Tests for Steiner Tree Problem."""

from __future__ import annotations

import sys
import os
import importlib.util

import numpy as np
import pytest

_variant_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_mod(name, filepath):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_inst = _load_mod("steiner_inst_test", os.path.join(_variant_dir, "instance.py"))
SteinerTreeInstance = _inst.SteinerTreeInstance
SteinerTreeSolution = _inst.SteinerTreeSolution
validate_solution = _inst.validate_solution
small_steiner_6 = _inst.small_steiner_6

_heur = _load_mod("steiner_heur_test", os.path.join(_variant_dir, "heuristics.py"))
kmb_heuristic = _heur.kmb_heuristic
shortest_path_heuristic = _heur.shortest_path_heuristic


class TestSteinerInstance:
    def test_random(self):
        inst = SteinerTreeInstance.random(n=8, n_terminals=3, seed=42)
        assert inst.n == 8
        assert len(inst.terminals) == 3

    def test_small(self):
        inst = small_steiner_6()
        assert inst.n == 6
        assert inst.terminals == {0, 4, 5}


class TestKMB:
    def test_valid(self):
        inst = SteinerTreeInstance.random(n=8, n_terminals=4, seed=42)
        sol = kmb_heuristic(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_small(self):
        inst = small_steiner_6()
        sol = kmb_heuristic(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_positive_weight(self):
        inst = SteinerTreeInstance.random(n=8, n_terminals=4, seed=42)
        sol = kmb_heuristic(inst)
        assert sol.total_weight > 0

    def test_single_terminal(self):
        inst = SteinerTreeInstance(n=4, edges=[(0,1,1),(1,2,2),(2,3,3)],
                                   terminals={1}, name="single")
        sol = kmb_heuristic(inst)
        assert sol.total_weight == 0.0


class TestShortestPathHeuristic:
    def test_valid(self):
        inst = SteinerTreeInstance.random(n=8, n_terminals=4, seed=42)
        sol = shortest_path_heuristic(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_small(self):
        inst = small_steiner_6()
        sol = shortest_path_heuristic(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_two_terminals(self):
        """Two terminals should give shortest path."""
        inst = SteinerTreeInstance(
            n=4,
            edges=[(0,1,1),(1,2,2),(0,2,10),(2,3,1)],
            terminals={0, 3},
            name="two_term",
        )
        sol = shortest_path_heuristic(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"
        assert sol.total_weight <= 4.0 + 1e-6  # 0->1->2->3 = 4
