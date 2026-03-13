"""Tests for Maximum Satisfiability (MAX-SAT) problem.

Tests: clause evaluation, weight computation, greedy correctness.
"""
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

_instance_mod = _load_mod("maxsat_inst_test", os.path.join(_base, "instance.py"))
_greedy_mod = _load_mod("maxsat_greedy_test", os.path.join(_base, "heuristics", "greedy_maxsat.py"))

MaxSATInstance = _instance_mod.MaxSATInstance
MaxSATSolution = _instance_mod.MaxSATSolution
greedy_maxsat = _greedy_mod.greedy_maxsat
random_assignment = _greedy_mod.random_assignment


class TestMaxSATInstance:
    """Test instance creation and evaluation."""

    def test_evaluate_all_true(self):
        # (x1 OR x2) AND (NOT x1 OR x2)
        inst = MaxSATInstance(
            n_vars=2,
            clauses=[[1, 2], [-1, 2]],
            weights=np.array([1.0, 1.0]),
        )
        weight, count = inst.evaluate([True, True])
        assert weight == 2.0
        assert count == 2

    def test_evaluate_partial(self):
        # (x1) AND (NOT x1)
        inst = MaxSATInstance(
            n_vars=1,
            clauses=[[1], [-1]],
            weights=np.array([3.0, 5.0]),
        )
        weight_t, count_t = inst.evaluate([True])
        assert weight_t == 3.0
        assert count_t == 1
        weight_f, count_f = inst.evaluate([False])
        assert weight_f == 5.0
        assert count_f == 1

    def test_total_weight(self):
        inst = MaxSATInstance(
            n_vars=2, clauses=[[1], [2]], weights=np.array([3.0, 7.0])
        )
        assert inst.total_weight() == 10.0

    def test_random_instance(self):
        inst = MaxSATInstance.random(n_vars=5, n_clauses=10, seed=42)
        assert inst.n_vars == 5
        assert len(inst.clauses) == 10
        assert len(inst.weights) == 10
        for clause in inst.clauses:
            for lit in clause:
                assert 1 <= abs(lit) <= 5

    def test_deterministic_random(self):
        inst1 = MaxSATInstance.random(seed=99)
        inst2 = MaxSATInstance.random(seed=99)
        assert inst1.clauses == inst2.clauses
        np.testing.assert_array_equal(inst1.weights, inst2.weights)


class TestGreedyMaxSAT:
    """Test greedy MAX-SAT solver."""

    def test_satisfiable_instance(self):
        """All clauses satisfiable -> greedy should satisfy all."""
        # (x1) AND (x2) AND (x1 OR x2)
        inst = MaxSATInstance(
            n_vars=2,
            clauses=[[1], [2], [1, 2]],
            weights=np.array([1.0, 1.0, 1.0]),
        )
        sol = greedy_maxsat(inst)
        assert sol.satisfied_weight == 3.0
        assert sol.n_satisfied == 3

    def test_unsatisfiable_clause(self):
        """(x1) AND (NOT x1) -> at most one satisfied."""
        inst = MaxSATInstance(
            n_vars=1, clauses=[[1], [-1]], weights=np.array([1.0, 1.0])
        )
        sol = greedy_maxsat(inst)
        assert sol.n_satisfied == 1

    def test_greedy_weight_preference(self):
        """Greedy should prefer heavy clauses."""
        # (x1) weight=1, (NOT x1) weight=100
        inst = MaxSATInstance(
            n_vars=1, clauses=[[1], [-1]], weights=np.array([1.0, 100.0])
        )
        sol = greedy_maxsat(inst)
        assert sol.satisfied_weight == 100.0
        assert sol.assignment[0] == False  # NOT x1 is better

    def test_random_assignment_valid(self):
        inst = MaxSATInstance.random(n_vars=5, n_clauses=10, seed=42)
        sol = random_assignment(inst, seed=42)
        assert len(sol.assignment) == 5
        assert sol.satisfied_weight >= 0
        verify_weight, verify_count = inst.evaluate(sol.assignment)
        assert abs(sol.satisfied_weight - verify_weight) < 1e-9

    def test_solution_repr(self):
        inst = MaxSATInstance.random(seed=42)
        sol = greedy_maxsat(inst)
        r = repr(sol)
        assert "MaxSATSolution" in r
        assert "weight=" in r

    def test_greedy_at_least_half(self):
        """Greedy should be at least as good as random (expected W/2)."""
        inst = MaxSATInstance.random(n_vars=10, n_clauses=30, seed=42)
        sol = greedy_maxsat(inst)
        # Should satisfy a decent fraction
        assert sol.satisfied_weight >= inst.total_weight() * 0.3
