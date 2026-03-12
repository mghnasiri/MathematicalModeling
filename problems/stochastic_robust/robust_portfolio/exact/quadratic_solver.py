"""
Quadratic Programming Solver for Portfolio Optimization

Solves the Markowitz mean-variance problem:
    max  mu^T w - lambda * w^T Sigma w
    s.t. sum(w) = 1, w >= 0

Uses scipy.optimize.minimize with SLSQP method.

For the robust variant, adds a penalty term for return uncertainty.

Complexity: Polynomial (convex QP).

References:
    - Markowitz, H. (1952). Portfolio selection. J. Finance, 7(1), 77-91.
"""
from __future__ import annotations

import sys
import os
import importlib.util

import numpy as np
from scipy.optimize import minimize

def _load_parent(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_inst = _load_parent("rp_instance", os.path.join(os.path.dirname(__file__), "..", "instance.py"))
RobustPortfolioInstance = _inst.RobustPortfolioInstance
PortfolioSolution = _inst.PortfolioSolution


def solve_mean_variance(instance: RobustPortfolioInstance) -> PortfolioSolution:
    """Solve classical Markowitz mean-variance portfolio.

    Args:
        instance: RobustPortfolioInstance (uncertainty_radius ignored).

    Returns:
        Optimal PortfolioSolution.
    """
    n = instance.n_assets

    def neg_objective(w):
        return -(np.dot(instance.expected_returns, w)
                 - instance.risk_aversion * (w @ instance.covariance @ w))

    def neg_objective_jac(w):
        return -(instance.expected_returns
                 - 2 * instance.risk_aversion * (instance.covariance @ w))

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0, 1)] * n
    w0 = np.full(n, 1.0 / n)

    result = minimize(neg_objective, w0, jac=neg_objective_jac,
                      method="SLSQP", bounds=bounds, constraints=constraints)

    w = result.x
    w = np.maximum(w, 0)
    w /= w.sum()

    ret = instance.portfolio_return(w)
    std = instance.portfolio_std(w)
    obj = instance.mean_variance_objective(w)
    sharpe = ret / std if std > 1e-10 else 0.0

    return PortfolioSolution(
        weights=w, expected_return=ret, portfolio_std=std,
        objective=obj, sharpe_ratio=sharpe,
    )


def solve_robust(instance: RobustPortfolioInstance) -> PortfolioSolution:
    """Solve robust portfolio with ellipsoidal return uncertainty.

    max  mu^T w - delta * ||Sigma^{1/2} w||_2 - lambda * w^T Sigma w

    Args:
        instance: RobustPortfolioInstance.

    Returns:
        Robust-optimal PortfolioSolution.
    """
    n = instance.n_assets
    delta = instance.uncertainty_radius

    def neg_robust_obj(w):
        ret = np.dot(instance.expected_returns, w)
        risk = w @ instance.covariance @ w
        std = np.sqrt(max(0, risk))
        return -(ret - delta * std - instance.risk_aversion * risk)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0, 1)] * n
    w0 = np.full(n, 1.0 / n)

    result = minimize(neg_robust_obj, w0, method="SLSQP",
                      bounds=bounds, constraints=constraints)

    w = result.x
    w = np.maximum(w, 0)
    w /= w.sum()

    ret = instance.portfolio_return(w)
    std = instance.portfolio_std(w)
    obj = instance.robust_objective(w)
    sharpe = ret / std if std > 1e-10 else 0.0

    return PortfolioSolution(
        weights=w, expected_return=ret, portfolio_std=std,
        objective=obj, sharpe_ratio=sharpe,
    )
