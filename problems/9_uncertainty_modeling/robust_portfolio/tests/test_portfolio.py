"""Tests for Robust Portfolio Optimization."""
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
_inst_mod = _load_mod("rp_instance", os.path.join(_base, "instance.py"))
_qp_mod = _load_mod("rp_qp", os.path.join(_base, "exact", "quadratic_solver.py"))
_heur_mod = _load_mod("rp_heur", os.path.join(_base, "heuristics", "equal_weight.py"))

RobustPortfolioInstance = _inst_mod.RobustPortfolioInstance
PortfolioSolution = _inst_mod.PortfolioSolution
solve_mean_variance = _qp_mod.solve_mean_variance
solve_robust = _qp_mod.solve_robust
equal_weight = _heur_mod.equal_weight
max_return = _heur_mod.max_return
min_variance = _heur_mod.min_variance


def _make_simple():
    return RobustPortfolioInstance(
        n_assets=3,
        expected_returns=np.array([0.10, 0.05, 0.08]),
        covariance=np.array([
            [0.04, 0.01, 0.02],
            [0.01, 0.02, 0.005],
            [0.02, 0.005, 0.03],
        ]),
        risk_aversion=1.0,
        uncertainty_radius=0.05,
    )


class TestPortfolioInstance:

    def test_creation(self):
        inst = _make_simple()
        assert inst.n_assets == 3

    def test_portfolio_return(self):
        inst = _make_simple()
        w = np.array([1.0, 0.0, 0.0])
        assert inst.portfolio_return(w) == pytest.approx(0.10)

    def test_portfolio_risk(self):
        inst = _make_simple()
        w = np.array([1.0, 0.0, 0.0])
        assert inst.portfolio_risk(w) == pytest.approx(0.04)

    def test_equal_weights_sum_to_one(self):
        inst = _make_simple()
        w = np.full(3, 1/3)
        assert abs(w.sum() - 1.0) < 1e-10

    def test_random_instance(self):
        inst = RobustPortfolioInstance.random(n_assets=5, seed=42)
        assert inst.n_assets == 5
        # Covariance should be positive semi-definite
        eigvals = np.linalg.eigvalsh(inst.covariance)
        assert all(v >= -1e-10 for v in eigvals)


class TestExactSolvers:

    def test_mean_variance_weights_sum_to_one(self):
        inst = _make_simple()
        sol = solve_mean_variance(inst)
        assert abs(sol.weights.sum() - 1.0) < 1e-6
        assert all(w >= -1e-6 for w in sol.weights)

    def test_mean_variance_positive_return(self):
        inst = _make_simple()
        sol = solve_mean_variance(inst)
        assert sol.expected_return > 0

    def test_robust_weights_sum_to_one(self):
        inst = _make_simple()
        sol = solve_robust(inst)
        assert abs(sol.weights.sum() - 1.0) < 1e-6

    def test_robust_more_conservative(self):
        """Robust solution should have lower return than MV."""
        inst = RobustPortfolioInstance.random(n_assets=5, seed=42,
                                             risk_aversion=0.5,
                                             uncertainty_radius=0.3)
        sol_mv = solve_mean_variance(inst)
        sol_rob = solve_robust(inst)
        # Robust should be more conservative (lower or equal return)
        assert sol_rob.expected_return <= sol_mv.expected_return + 1e-4

    def test_zero_uncertainty_matches_mv(self):
        """With delta=0, robust should match mean-variance."""
        inst = RobustPortfolioInstance(
            n_assets=3,
            expected_returns=np.array([0.10, 0.05, 0.08]),
            covariance=np.array([
                [0.04, 0.01, 0.02],
                [0.01, 0.02, 0.005],
                [0.02, 0.005, 0.03],
            ]),
            risk_aversion=1.0,
            uncertainty_radius=0.0,
        )
        sol_mv = solve_mean_variance(inst)
        sol_rob = solve_robust(inst)
        assert sol_mv.expected_return == pytest.approx(sol_rob.expected_return, abs=1e-3)


class TestHeuristics:

    def test_equal_weight(self):
        inst = _make_simple()
        sol = equal_weight(inst)
        assert abs(sol.weights.sum() - 1.0) < 1e-10
        assert all(abs(w - 1/3) < 1e-10 for w in sol.weights)

    def test_max_return_concentrated(self):
        inst = _make_simple()
        sol = max_return(inst)
        assert sol.weights[0] == pytest.approx(1.0)  # highest return asset

    def test_min_variance_lower_risk(self):
        inst = _make_simple()
        sol_eq = equal_weight(inst)
        sol_mv = min_variance(inst)
        assert sol_mv.portfolio_std <= sol_eq.portfolio_std + 1e-6

    def test_all_heuristics_valid_weights(self):
        inst = RobustPortfolioInstance.random(n_assets=5, seed=42)
        for method in [equal_weight, max_return, min_variance]:
            sol = method(inst)
            assert abs(sol.weights.sum() - 1.0) < 1e-6
            assert all(w >= -1e-6 for w in sol.weights)
