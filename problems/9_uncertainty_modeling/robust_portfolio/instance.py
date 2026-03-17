"""
Robust Portfolio Optimization

Classical Markowitz mean-variance portfolio selection with robustness
against estimation error in expected returns.

    max  mu^T w - lambda * w^T Sigma w
    s.t. sum(w) = 1, w >= 0

The robust variant considers uncertainty in mu:
    max  min_{mu in U} mu^T w - lambda * w^T Sigma w

With ellipsoidal uncertainty U = {mu : ||Sigma^{-1/2}(mu-mu_hat)||_2 <= delta},
the robust problem becomes:
    max  mu_hat^T w - delta * ||Sigma^{1/2} w||_2 - lambda * w^T Sigma w

Complexity: Convex (SOCP), solvable in polynomial time.

References:
    - Markowitz, H. (1952). Portfolio selection. J. Finance, 7(1), 77-91.
      https://doi.org/10.2307/2975974
    - Goldfarb, D. & Iyengar, G. (2003). Robust portfolio selection problems.
      Math. Oper. Res., 28(1), 1-38. https://doi.org/10.1287/moor.28.1.1.14260
    - Bertsimas, D., Brown, D.B. & Caramanis, C. (2011). Theory and applications
      of robust optimization. SIAM Review, 53(3), 464-501.
      https://doi.org/10.1137/080734510
"""
from __future__ import annotations

import sys
import os
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


@dataclass
class RobustPortfolioInstance:
    """Robust portfolio optimization instance.

    Args:
        n_assets: Number of assets.
        expected_returns: Estimated expected return per asset (n,).
        covariance: Return covariance matrix (n, n).
        risk_aversion: Lambda — risk-aversion parameter.
        uncertainty_radius: Delta — size of uncertainty set for returns.
    """
    n_assets: int
    expected_returns: np.ndarray
    covariance: np.ndarray
    risk_aversion: float
    uncertainty_radius: float = 0.0

    def __post_init__(self):
        self.expected_returns = np.asarray(self.expected_returns, dtype=float)
        self.covariance = np.asarray(self.covariance, dtype=float)

    def portfolio_return(self, weights: np.ndarray) -> float:
        """Expected portfolio return: mu^T w."""
        return float(np.dot(self.expected_returns, weights))

    def portfolio_risk(self, weights: np.ndarray) -> float:
        """Portfolio variance: w^T Sigma w."""
        return float(weights @ self.covariance @ weights)

    def portfolio_std(self, weights: np.ndarray) -> float:
        """Portfolio standard deviation."""
        return float(np.sqrt(max(0, self.portfolio_risk(weights))))

    def mean_variance_objective(self, weights: np.ndarray) -> float:
        """Markowitz objective: mu^T w - lambda * w^T Sigma w."""
        return self.portfolio_return(weights) - self.risk_aversion * self.portfolio_risk(weights)

    def robust_objective(self, weights: np.ndarray) -> float:
        """Robust objective with worst-case return penalty.

        mu_hat^T w - delta * ||Sigma^{1/2} w||_2 - lambda * w^T Sigma w
        """
        ret = self.portfolio_return(weights)
        risk = self.portfolio_risk(weights)
        penalty = self.uncertainty_radius * self.portfolio_std(weights)
        return ret - penalty - self.risk_aversion * risk

    @classmethod
    def random(cls, n_assets: int = 5, seed: int = 42,
               risk_aversion: float = 1.0,
               uncertainty_radius: float = 0.1) -> RobustPortfolioInstance:
        """Generate a random portfolio instance with realistic correlations."""
        rng = np.random.default_rng(seed)
        expected_returns = rng.uniform(0.02, 0.15, n_assets)
        # Generate covariance from random factor model
        k = max(2, n_assets // 2)
        factors = rng.normal(0, 0.1, (k, n_assets))
        covariance = factors.T @ factors + np.diag(rng.uniform(0.01, 0.05, n_assets))
        # Ensure symmetric
        covariance = (covariance + covariance.T) / 2

        return cls(
            n_assets=n_assets,
            expected_returns=expected_returns,
            covariance=covariance,
            risk_aversion=risk_aversion,
            uncertainty_radius=uncertainty_radius,
        )


@dataclass
class PortfolioSolution:
    """Solution to the portfolio optimization problem.

    Args:
        weights: Asset allocation weights (sum to 1).
        expected_return: mu^T w.
        portfolio_std: sqrt(w^T Sigma w).
        objective: Objective function value (MV or robust).
        sharpe_ratio: Return / std (if std > 0).
    """
    weights: np.ndarray
    expected_return: float
    portfolio_std: float
    objective: float
    sharpe_ratio: float = 0.0

    def __repr__(self) -> str:
        return (f"PortfolioSolution(return={self.expected_return:.4f}, "
                f"std={self.portfolio_std:.4f}, "
                f"sharpe={self.sharpe_ratio:.3f})")
