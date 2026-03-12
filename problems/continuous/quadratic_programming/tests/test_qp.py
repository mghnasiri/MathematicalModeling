"""Tests for the Quadratic Programming problem."""
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
_instance_mod = _load_mod("qp_instance", os.path.join(_base, "instance.py"))
_solver_mod = _load_mod(
    "qp_solver", os.path.join(_base, "exact", "qp_solver.py")
)

QPInstance = _instance_mod.QPInstance
QPSolution = _instance_mod.QPSolution
solve_qp_slsqp = _solver_mod.solve_qp_slsqp
solve_qp_trust = _solver_mod.solve_qp_trust


class TestQPInstance:
    """Tests for QP instance construction."""

    def test_random_instance(self):
        inst = QPInstance.random(n=5, m_ub=3, seed=1)
        assert inst.n == 5
        assert inst.Q.shape == (5, 5)
        assert len(inst.c) == 5

    def test_q_symmetric(self):
        inst = QPInstance.random(n=6, seed=2)
        np.testing.assert_allclose(inst.Q, inst.Q.T, atol=1e-10)

    def test_q_positive_semidefinite(self):
        inst = QPInstance.random(n=5, seed=3)
        eigenvalues = np.linalg.eigvalsh(inst.Q)
        assert np.all(eigenvalues >= -1e-10)

    def test_objective_at_zero(self):
        inst = QPInstance.random(n=3, seed=4)
        val = inst.objective(np.zeros(3))
        assert abs(val) < 1e-10


class TestSLSQP:
    """Tests for SLSQP QP solver."""

    def test_unconstrained_simple(self):
        """min (1/2) x^2 + x => x* = -1, obj* = -0.5"""
        inst = QPInstance(
            n=1, Q=np.array([[1.0]]), c=np.array([1.0]),
        )
        sol = solve_qp_slsqp(inst)
        assert sol.success
        assert abs(sol.x[0] - (-1.0)) < 1e-4
        assert abs(sol.objective - (-0.5)) < 1e-4

    def test_bounded_solution(self):
        """min (1/2) x^2 + x with x >= 0 => x* = 0"""
        inst = QPInstance(
            n=1, Q=np.array([[1.0]]), c=np.array([1.0]),
            bounds=[(0.0, None)],
        )
        sol = solve_qp_slsqp(inst)
        assert sol.success
        assert abs(sol.x[0]) < 1e-4
        assert abs(sol.objective) < 1e-4

    def test_two_variable(self):
        """min x1^2 + x2^2 with x1 + x2 = 1"""
        inst = QPInstance(
            n=2,
            Q=2 * np.eye(2),
            c=np.zeros(2),
            A_eq=np.array([[1.0, 1.0]]),
            b_eq=np.array([1.0]),
        )
        sol = solve_qp_slsqp(inst)
        assert sol.success
        np.testing.assert_allclose(sol.x, [0.5, 0.5], atol=1e-4)

    def test_random_instance_converges(self):
        inst = QPInstance.random(n=5, m_ub=3, seed=10)
        sol = solve_qp_slsqp(inst)
        assert sol.success

    def test_constraints_satisfied(self):
        inst = QPInstance.random(n=4, m_ub=2, seed=11)
        sol = solve_qp_slsqp(inst)
        assert sol.success
        if inst.A_ub is not None:
            residuals = inst.A_ub @ sol.x - inst.b_ub
            assert np.all(residuals <= 1e-6)


class TestTrustConstr:
    """Tests for trust-constr QP solver."""

    def test_matches_slsqp(self):
        inst = QPInstance.random(n=4, m_ub=2, seed=20)
        sol1 = solve_qp_slsqp(inst)
        sol2 = solve_qp_trust(inst)
        assert sol1.success and sol2.success
        assert abs(sol1.objective - sol2.objective) < 1e-3

    def test_simple_qp(self):
        inst = QPInstance(
            n=2, Q=np.eye(2), c=np.array([-1.0, -2.0]),
            bounds=[(0.0, None), (0.0, None)],
        )
        sol = solve_qp_trust(inst)
        assert sol.success
        np.testing.assert_allclose(sol.x, [1.0, 2.0], atol=1e-3)
