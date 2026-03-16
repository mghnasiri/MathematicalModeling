"""
Real-World Application: Investment Portfolio Optimization Under Uncertainty.

Domain: Asset management / Pension fund allocation
Model: Robust Portfolio Optimization (Markowitz mean-variance + ellipsoidal uncertainty)

Scenario:
    A pension fund manager allocates capital across 8 asset classes:
    US Large-Cap Equities, International Equities, Emerging Markets,
    US Government Bonds, Corporate Bonds, Real Estate (REITs),
    Commodities, and Cash Equivalents.

    The manager faces two key challenges:
    1. Maximizing risk-adjusted returns (Markowitz mean-variance)
    2. Protecting against estimation error in expected returns
       (robust optimization with ellipsoidal uncertainty)

    The fund has a $100M allocation and must decide weights for each
    asset class, subject to full investment (weights sum to 1) and
    no short-selling (weights >= 0).

Real-world considerations modeled:
    - Historical return estimates with estimation uncertainty
    - Covariance structure reflecting asset class correlations
    - Risk aversion parameter reflecting fund's liability structure
    - Robustness radius capturing confidence in return forecasts

Industry context:
    Estimation error in expected returns is the dominant source of
    portfolio sub-optimality (Chopra & Ziemba, 1993). Robust portfolio
    models reduce turnover and improve out-of-sample Sharpe ratios
    by 20-40% compared to classical Markowitz (Goldfarb & Iyengar, 2003).

References:
    Markowitz, H. (1952). Portfolio Selection. The Journal of Finance,
    7(1), 77-91. https://doi.org/10.2307/2975974

    Goldfarb, D. & Iyengar, G. (2003). Robust Portfolio Selection
    Problems. Mathematics of Operations Research, 28(1), 1-38.
    https://doi.org/10.1287/moor.28.1.1.14260
"""

from __future__ import annotations

import sys
import os
import importlib.util

import numpy as np


def _load_mod(name: str, filepath: str):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ── Domain Data ──────────────────────────────────────────────────────────────

ASSET_CLASSES = [
    "US Large-Cap Equities",
    "International Equities",
    "Emerging Markets",
    "US Government Bonds",
    "Corporate Bonds",
    "Real Estate (REITs)",
    "Commodities",
    "Cash Equivalents",
]

# Annualized expected returns (%) — based on long-run historical averages
EXPECTED_RETURNS = np.array([
    10.0,   # US Large-Cap
    8.5,    # International
    11.0,   # Emerging Markets
    3.5,    # US Gov Bonds
    5.0,    # Corporate Bonds
    8.0,    # REITs
    5.5,    # Commodities
    2.0,    # Cash
]) / 100.0  # Convert to decimal

# Annualized volatilities (%)
VOLATILITIES = np.array([
    16.0,   # US Large-Cap
    18.0,   # International
    24.0,   # Emerging Markets
    5.0,    # US Gov Bonds
    7.0,    # Corporate Bonds
    20.0,   # REITs
    18.0,   # Commodities
    1.0,    # Cash
]) / 100.0

# Correlation matrix (symmetric, based on typical asset class correlations)
CORRELATIONS = np.array([
    [1.00, 0.75, 0.65, -0.10, 0.15, 0.55, 0.20, 0.00],
    [0.75, 1.00, 0.70, -0.05, 0.10, 0.45, 0.25, 0.00],
    [0.65, 0.70, 1.00,  0.00, 0.10, 0.40, 0.30, 0.00],
    [-0.10, -0.05, 0.00, 1.00, 0.65, 0.10, -0.05, 0.30],
    [0.15, 0.10, 0.10,  0.65, 1.00, 0.20, 0.05, 0.20],
    [0.55, 0.45, 0.40,  0.10, 0.20, 1.00, 0.15, 0.00],
    [0.20, 0.25, 0.30, -0.05, 0.05, 0.15, 1.00, 0.00],
    [0.00, 0.00, 0.00,  0.30, 0.20, 0.00, 0.00, 1.00],
])


def create_portfolio_instance(risk_aversion: float = 2.0,
                               uncertainty_radius: float = 0.0) -> dict:
    """Create a pension fund portfolio optimization instance.

    Args:
        risk_aversion: Lambda parameter for risk penalty (higher = more conservative).
        uncertainty_radius: Delta for robust model (0 = classical Markowitz).

    Returns:
        Dictionary with instance data and metadata.
    """
    n = len(ASSET_CLASSES)

    # Build covariance matrix from correlations and volatilities
    covariance = np.outer(VOLATILITIES, VOLATILITIES) * CORRELATIONS

    # Ensure positive semi-definite (fix numerical issues)
    eigvals, eigvecs = np.linalg.eigh(covariance)
    eigvals = np.maximum(eigvals, 1e-10)
    covariance = eigvecs @ np.diag(eigvals) @ eigvecs.T
    covariance = (covariance + covariance.T) / 2

    return {
        "n_assets": n,
        "asset_names": ASSET_CLASSES,
        "expected_returns": EXPECTED_RETURNS.copy(),
        "covariance": covariance,
        "volatilities": VOLATILITIES.copy(),
        "risk_aversion": risk_aversion,
        "uncertainty_radius": uncertainty_radius,
        "fund_size": 100_000_000,  # $100M
    }


def solve_portfolio(verbose: bool = True) -> dict:
    """Solve the pension fund portfolio optimization problem.

    Compares classical Markowitz, robust optimization, and simple heuristics.

    Returns:
        Dictionary with results from each method.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rp_dir = os.path.join(
        base_dir, "problems", "stochastic_robust", "robust_portfolio"
    )

    rp_inst_mod = _load_mod(
        "rp_inst_app", os.path.join(rp_dir, "instance.py")
    )
    rp_exact_mod = _load_mod(
        "rp_exact_app", os.path.join(rp_dir, "exact", "quadratic_solver.py")
    )
    rp_heur_mod = _load_mod(
        "rp_heur_app", os.path.join(rp_dir, "heuristics", "equal_weight.py")
    )

    results = {}

    # ── Classical Markowitz (no robustness) ──────────────────────────────
    data_mv = create_portfolio_instance(risk_aversion=2.0, uncertainty_radius=0.0)
    mv_instance = rp_inst_mod.RobustPortfolioInstance(
        n_assets=data_mv["n_assets"],
        expected_returns=data_mv["expected_returns"],
        covariance=data_mv["covariance"],
        risk_aversion=data_mv["risk_aversion"],
        uncertainty_radius=0.0,
    )

    mv_sol = rp_exact_mod.solve_mean_variance(mv_instance)
    results["Markowitz"] = {
        "weights": mv_sol.weights,
        "expected_return": mv_sol.expected_return,
        "portfolio_std": mv_sol.portfolio_std,
        "objective": mv_sol.objective,
    }

    # ── Robust Portfolio (uncertainty radius = 0.05) ─────────────────────
    data_rob = create_portfolio_instance(risk_aversion=2.0, uncertainty_radius=0.05)
    rob_instance = rp_inst_mod.RobustPortfolioInstance(
        n_assets=data_rob["n_assets"],
        expected_returns=data_rob["expected_returns"],
        covariance=data_rob["covariance"],
        risk_aversion=data_rob["risk_aversion"],
        uncertainty_radius=0.05,
    )

    rob_sol = rp_exact_mod.solve_robust(rob_instance)
    results["Robust"] = {
        "weights": rob_sol.weights,
        "expected_return": rob_sol.expected_return,
        "portfolio_std": rob_sol.portfolio_std,
        "objective": rob_sol.objective,
    }

    # ── Heuristic: Equal Weight (1/n) ────────────────────────────────────
    ew_sol = rp_heur_mod.equal_weight(mv_instance)
    results["Equal-Weight"] = {
        "weights": ew_sol.weights,
        "expected_return": ew_sol.expected_return,
        "portfolio_std": ew_sol.portfolio_std,
        "objective": ew_sol.objective,
    }

    # ── Heuristic: Min Variance ──────────────────────────────────────────
    minvar_sol = rp_heur_mod.min_variance(mv_instance)
    results["Min-Variance"] = {
        "weights": minvar_sol.weights,
        "expected_return": minvar_sol.expected_return,
        "portfolio_std": minvar_sol.portfolio_std,
        "objective": minvar_sol.objective,
    }

    # ── Heuristic: Max Return ────────────────────────────────────────────
    maxret_sol = rp_heur_mod.max_return(mv_instance)
    results["Max-Return"] = {
        "weights": maxret_sol.weights,
        "expected_return": maxret_sol.expected_return,
        "portfolio_std": maxret_sol.portfolio_std,
        "objective": maxret_sol.objective,
    }

    if verbose:
        fund_size = data_mv["fund_size"]
        print("=" * 70)
        print("PENSION FUND PORTFOLIO OPTIMIZATION")
        print(f"  Fund size: ${fund_size:,.0f}")
        print(f"  {data_mv['n_assets']} asset classes")
        print(f"  Risk aversion (lambda): {data_mv['risk_aversion']}")
        print("=" * 70)

        for method, res in results.items():
            print(f"\n--- {method} ---")
            print(f"  Expected return: {res['expected_return']:.2%}")
            print(f"  Portfolio std:   {res['portfolio_std']:.2%}")
            print(f"  Allocation:")
            for i, name in enumerate(ASSET_CLASSES):
                w = res["weights"][i]
                if w > 0.005:  # Show allocations > 0.5%
                    amount = w * fund_size
                    print(f"    {name:30s} {w:6.1%}  (${amount:>12,.0f})")

    return results


if __name__ == "__main__":
    solve_portfolio()
