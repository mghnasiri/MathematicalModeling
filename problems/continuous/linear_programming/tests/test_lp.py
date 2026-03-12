"""Tests for Linear Programming with Sensitivity Analysis."""
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
_instance_mod = _load_mod("lp_instance", os.path.join(_base, "instance.py"))
_solver_mod = _load_mod(
    "lp_solver", os.path.join(_base, "exact", "lp_solver.py")
)

LPInstance = _instance_mod.LPInstance
LPSolution = _instance_mod.LPSolution
solve_lp = _solver_mod.solve_lp
sensitivity_report = _solver_mod.sensitivity_report


class TestLPInstance:
    """Tests for LP instance construction."""

    def test_random_instance(self):
        inst = LPInstance.random(n=4, m_ub=3, seed=1)
        assert inst.n == 4
        assert inst.c.shape == (4,)
        assert inst.A_ub.shape == (3, 4)

    def test_objective_evaluation(self):
        inst = LPInstance(n=2, c=np.array([3.0, 5.0]))
        val = inst.objective(np.array([2.0, 1.0]))
        assert abs(val - 11.0) < 1e-10


class TestSolveLP:
    """Tests for the LP solver."""

    def test_simple_lp(self):
        """min -x1 - x2 s.t. x1 + x2 <= 4, x1 <= 3, x2 <= 3, x >= 0"""
        inst = LPInstance(
            n=2,
            c=np.array([-1.0, -1.0]),
            A_ub=np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]),
            b_ub=np.array([4.0, 3.0, 3.0]),
            bounds=[(0, None), (0, None)],
        )
        sol = solve_lp(inst)
        assert sol.success
        assert abs(sol.objective - (-4.0)) < 1e-4

    def test_production_problem(self):
        """max 5x1 + 4x2 => min -5x1 - 4x2"""
        inst = LPInstance(
            n=2,
            c=np.array([-5.0, -4.0]),
            A_ub=np.array([[6.0, 4.0], [1.0, 2.0]]),
            b_ub=np.array([24.0, 6.0]),
            bounds=[(0, None), (0, None)],
        )
        sol = solve_lp(inst)
        assert sol.success
        assert abs(sol.objective - (-21.0)) < 1e-4
        np.testing.assert_allclose(sol.x, [3.0, 1.5], atol=1e-4)

    def test_equality_constraint(self):
        """min x1 + x2 s.t. x1 + x2 = 5, x >= 0"""
        inst = LPInstance(
            n=2,
            c=np.array([1.0, 1.0]),
            A_eq=np.array([[1.0, 1.0]]),
            b_eq=np.array([5.0]),
            bounds=[(0, None), (0, None)],
        )
        sol = solve_lp(inst)
        assert sol.success
        assert abs(sol.objective - 5.0) < 1e-4

    def test_random_instance_solves(self):
        inst = LPInstance.random(n=4, m_ub=3, seed=10)
        sol = solve_lp(inst)
        assert sol.success

    def test_constraints_satisfied(self):
        inst = LPInstance.random(n=3, m_ub=2, seed=11)
        sol = solve_lp(inst)
        assert sol.success
        residuals = inst.A_ub @ sol.x - inst.b_ub
        assert np.all(residuals <= 1e-6)

    def test_bounds_satisfied(self):
        inst = LPInstance.random(n=3, m_ub=2, seed=12)
        sol = solve_lp(inst)
        assert sol.success
        assert np.all(sol.x >= -1e-6)


class TestSensitivity:
    """Tests for sensitivity analysis."""

    def test_shadow_prices_computed(self):
        inst = LPInstance(
            n=2,
            c=np.array([-5.0, -4.0]),
            A_ub=np.array([[6.0, 4.0], [1.0, 2.0]]),
            b_ub=np.array([24.0, 6.0]),
            bounds=[(0, None), (0, None)],
        )
        sol = solve_lp(inst)
        assert sol.success
        assert sol.shadow_prices is not None
        assert len(sol.shadow_prices) == 2

    def test_sensitivity_report_keys(self):
        inst = LPInstance.random(n=3, m_ub=2, seed=20)
        sol = solve_lp(inst)
        report = sensitivity_report(inst, sol)
        assert "optimal_objective" in report
        assert "optimal_x" in report
        assert "binding_constraints" in report
        assert "slack" in report

    def test_binding_constraint_zero_slack(self):
        inst = LPInstance(
            n=2,
            c=np.array([-1.0, -1.0]),
            A_ub=np.array([[1.0, 1.0]]),
            b_ub=np.array([4.0]),
            bounds=[(0, None), (0, None)],
        )
        sol = solve_lp(inst)
        report = sensitivity_report(inst, sol)
        # The constraint x1 + x2 <= 4 should be binding
        assert 0 in report["binding_constraints"]
