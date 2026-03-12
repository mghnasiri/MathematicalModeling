"""
Simple Heuristics for Portfolio Optimization

Includes equal-weight (1/n) and maximum Sharpe ratio heuristics.

References:
    - DeMiguel, V., Garlappi, L. & Uppal, R. (2009). Optimal versus naive
      diversification. Rev. Financ. Stud., 22(5), 1915-1953.
      https://doi.org/10.1093/rfs/hhm075
"""
from __future__ import annotations

import sys
import os
import importlib.util

import numpy as np

def _load_parent(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_inst = _load_parent("rp_instance", os.path.join(os.path.dirname(__file__), "..", "instance.py"))
RobustPortfolioInstance = _inst.RobustPortfolioInstance
PortfolioSolution = _inst.PortfolioSolution


def equal_weight(instance: RobustPortfolioInstance) -> PortfolioSolution:
    """1/n equal-weight portfolio.

    Surprisingly competitive with optimized portfolios due to
    estimation error in mu and Sigma (DeMiguel et al., 2009).

    Args:
        instance: RobustPortfolioInstance.

    Returns:
        PortfolioSolution with equal weights.
    """
    n = instance.n_assets
    w = np.full(n, 1.0 / n)
    ret = instance.portfolio_return(w)
    std = instance.portfolio_std(w)
    obj = instance.robust_objective(w)
    sharpe = ret / std if std > 1e-10 else 0.0

    return PortfolioSolution(
        weights=w, expected_return=ret, portfolio_std=std,
        objective=obj, sharpe_ratio=sharpe,
    )


def max_return(instance: RobustPortfolioInstance) -> PortfolioSolution:
    """Invest 100% in the asset with highest expected return.

    Args:
        instance: RobustPortfolioInstance.

    Returns:
        Concentrated PortfolioSolution.
    """
    n = instance.n_assets
    best = int(np.argmax(instance.expected_returns))
    w = np.zeros(n)
    w[best] = 1.0
    ret = instance.portfolio_return(w)
    std = instance.portfolio_std(w)
    obj = instance.robust_objective(w)
    sharpe = ret / std if std > 1e-10 else 0.0

    return PortfolioSolution(
        weights=w, expected_return=ret, portfolio_std=std,
        objective=obj, sharpe_ratio=sharpe,
    )


def min_variance(instance: RobustPortfolioInstance) -> PortfolioSolution:
    """Minimum variance portfolio via closed-form.

    w* = Sigma^{-1} 1 / (1^T Sigma^{-1} 1)

    Args:
        instance: RobustPortfolioInstance.

    Returns:
        Minimum variance PortfolioSolution.
    """
    try:
        inv_cov = np.linalg.inv(instance.covariance)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(instance.covariance)

    ones = np.ones(instance.n_assets)
    w = inv_cov @ ones
    w = np.maximum(w, 0)  # enforce non-negativity
    if w.sum() > 1e-10:
        w /= w.sum()
    else:
        w = np.full(instance.n_assets, 1.0 / instance.n_assets)

    ret = instance.portfolio_return(w)
    std = instance.portfolio_std(w)
    obj = instance.robust_objective(w)
    sharpe = ret / std if std > 1e-10 else 0.0

    return PortfolioSolution(
        weights=w, expected_return=ret, portfolio_std=std,
        objective=obj, sharpe_ratio=sharpe,
    )
