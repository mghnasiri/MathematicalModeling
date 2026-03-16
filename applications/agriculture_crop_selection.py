"""
Real-World Application: Pre-Season Crop Portfolio Selection Under Weather Uncertainty.

Domain: Pre-season crop portfolio selection under weather uncertainty
Model: Robust Portfolio Optimization (Markowitz mean-variance with
       ellipsoidal uncertainty on expected returns) — select a crop mix
       to maximize expected profit while managing weather risk.

Scenario:
    A farmer with 500 acres must decide how to allocate land across
    6 candidate crops before the growing season. Each crop has:
    - An expected profit per acre (based on historical yields and prices)
    - Profit variance driven by weather uncertainty (drought, frost, etc.)
    - Correlation with other crops (e.g., drought affects all row crops)

    The farmer wants to maximize expected profit while controlling
    downside risk through diversification. The robust formulation
    accounts for uncertainty in the profit estimates themselves
    (e.g., price forecasts may be inaccurate).

    Three approaches are compared:
    1. Classical mean-variance (Markowitz) — optimal diversification
    2. Robust portfolio — hedges against forecast errors
    3. Simple heuristics — equal allocation, min-variance, max-return

Real-world considerations modeled:
    - Weather correlation between crops (drought affects all field crops)
    - Price uncertainty separate from yield uncertainty
    - Non-negative allocation (cannot short a crop)
    - Full land utilization (all 500 acres planted)
    - Different risk profiles (conservative vs aggressive farmer)

Industry context:
    Crop diversification reduces income volatility by 20-40% compared
    to monoculture. The USDA Risk Management Agency reports that farmers
    using diversified rotations file 30% fewer crop insurance claims.
    Modern precision agriculture and futures markets allow farmers to
    optimize portfolios similarly to financial assets (Hardaker et al., 2004).

References:
    Hardaker, J.B., Huirne, R.B.M., Anderson, J.R. & Lien, G. (2004).
    Coping with Risk in Agriculture. 2nd Edition, CABI Publishing.

    Markowitz, H. (1952). Portfolio Selection. The Journal of Finance,
    7(1), 77-91. https://doi.org/10.2307/2975974

    Goldfarb, D. & Iyengar, G. (2003). Robust Portfolio Selection
    Problems. Mathematics of Operations Research, 28(1), 1-38.
    https://doi.org/10.1287/moor.28.1.1.14260

    Hazell, P.B.R. & Norton, R.D. (1986). Mathematical Programming
    for Economic Analysis in Agriculture. Macmillan.
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

TOTAL_ACRES = 500

# 6 candidate crops
CROPS = [
    "Corn (Maize)",
    "Soybeans",
    "Winter Wheat",
    "Alfalfa (Hay)",
    "Sunflowers",
    "Oats",
]

# Expected profit per acre ($/acre) — based on 5-year county averages
EXPECTED_PROFIT_PER_ACRE = np.array([
    420.0,   # Corn — high yield, moderate price volatility
    380.0,   # Soybeans — good margins, trade-sensitive
    280.0,   # Winter Wheat — lower but stable
    220.0,   # Alfalfa — steady hay market
    350.0,   # Sunflowers — niche market, higher variance
    200.0,   # Oats — low margin but drought-tolerant
])

# Profit volatility per acre ($/acre standard deviation)
PROFIT_VOLATILITY = np.array([
    180.0,   # Corn — high weather sensitivity
    150.0,   # Soybeans — moderate
    80.0,    # Winter Wheat — planted in fall, lower risk
    60.0,    # Alfalfa — perennial, very stable
    200.0,   # Sunflowers — small market, price swings
    70.0,    # Oats — hardy, stable
])

# Correlation matrix between crop profits (weather-driven)
# Drought affects corn/soy/sunflower similarly; wheat/oats less so
CROP_CORRELATIONS = np.array([
    [1.00, 0.75, 0.30, 0.15, 0.60, 0.20],   # Corn
    [0.75, 1.00, 0.25, 0.10, 0.55, 0.15],   # Soybeans
    [0.30, 0.25, 1.00, 0.35, 0.20, 0.50],   # Winter Wheat
    [0.15, 0.10, 0.35, 1.00, 0.10, 0.40],   # Alfalfa
    [0.60, 0.55, 0.20, 0.10, 1.00, 0.15],   # Sunflowers
    [0.20, 0.15, 0.50, 0.40, 0.15, 1.00],   # Oats
])


def create_crop_portfolio_instance(risk_aversion: float = 1.5,
                                    uncertainty_radius: float = 0.0) -> dict:
    """Create a crop portfolio optimization instance.

    Args:
        risk_aversion: Lambda parameter for risk penalty.
            Higher = more conservative farmer.
        uncertainty_radius: Delta for robust model (0 = classical).
            Reflects confidence in profit forecasts.

    Returns:
        Dictionary with instance data and metadata.
    """
    n = len(CROPS)

    # Normalize to per-dollar returns (treat as portfolio fractions)
    # Expected returns as fraction of total potential
    total_max_profit = EXPECTED_PROFIT_PER_ACRE.max() * TOTAL_ACRES
    expected_returns = EXPECTED_PROFIT_PER_ACRE / total_max_profit

    # Build covariance matrix from volatilities and correlations
    vol_normalized = PROFIT_VOLATILITY / total_max_profit
    covariance = np.outer(vol_normalized, vol_normalized) * CROP_CORRELATIONS

    # Ensure positive semi-definite
    eigvals, eigvecs = np.linalg.eigh(covariance)
    eigvals = np.maximum(eigvals, 1e-12)
    covariance = eigvecs @ np.diag(eigvals) @ eigvecs.T
    covariance = (covariance + covariance.T) / 2

    return {
        "n_crops": n,
        "crop_names": CROPS,
        "expected_profit_per_acre": EXPECTED_PROFIT_PER_ACRE.copy(),
        "profit_volatility": PROFIT_VOLATILITY.copy(),
        "expected_returns": expected_returns,
        "covariance": covariance,
        "risk_aversion": risk_aversion,
        "uncertainty_radius": uncertainty_radius,
        "total_acres": TOTAL_ACRES,
    }


def solve_crop_selection(verbose: bool = True) -> dict:
    """Solve the crop portfolio selection problem.

    Compares classical Markowitz, robust optimization, and heuristics
    for selecting the optimal crop mix.

    Returns:
        Dictionary with results from each method.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rp_dir = os.path.join(
        base_dir, "problems", "stochastic_robust", "robust_portfolio"
    )

    rp_inst_mod = _load_mod(
        "rp_inst_crop", os.path.join(rp_dir, "instance.py")
    )
    rp_exact_mod = _load_mod(
        "rp_exact_crop", os.path.join(rp_dir, "exact", "quadratic_solver.py")
    )
    rp_heur_mod = _load_mod(
        "rp_heur_crop", os.path.join(rp_dir, "heuristics", "equal_weight.py")
    )

    results = {}

    # ── Classical Mean-Variance (moderate risk aversion) ─────────────────
    data_mv = create_crop_portfolio_instance(risk_aversion=1.5,
                                              uncertainty_radius=0.0)
    mv_instance = rp_inst_mod.RobustPortfolioInstance(
        n_assets=data_mv["n_crops"],
        expected_returns=data_mv["expected_returns"],
        covariance=data_mv["covariance"],
        risk_aversion=data_mv["risk_aversion"],
        uncertainty_radius=0.0,
    )

    mv_sol = rp_exact_mod.solve_mean_variance(mv_instance)
    results["Mean-Variance"] = {
        "weights": mv_sol.weights,
        "expected_return": mv_sol.expected_return,
        "portfolio_std": mv_sol.portfolio_std,
        "objective": mv_sol.objective,
    }

    # ── Robust Portfolio (uncertainty in profit forecasts) ────────────────
    data_rob = create_crop_portfolio_instance(risk_aversion=1.5,
                                               uncertainty_radius=0.03)
    rob_instance = rp_inst_mod.RobustPortfolioInstance(
        n_assets=data_rob["n_crops"],
        expected_returns=data_rob["expected_returns"],
        covariance=data_rob["covariance"],
        risk_aversion=data_rob["risk_aversion"],
        uncertainty_radius=0.03,
    )

    rob_sol = rp_exact_mod.solve_robust(rob_instance)
    results["Robust"] = {
        "weights": rob_sol.weights,
        "expected_return": rob_sol.expected_return,
        "portfolio_std": rob_sol.portfolio_std,
        "objective": rob_sol.objective,
    }

    # ── Conservative Farmer (high risk aversion) ─────────────────────────
    data_cons = create_crop_portfolio_instance(risk_aversion=5.0,
                                                uncertainty_radius=0.0)
    cons_instance = rp_inst_mod.RobustPortfolioInstance(
        n_assets=data_cons["n_crops"],
        expected_returns=data_cons["expected_returns"],
        covariance=data_cons["covariance"],
        risk_aversion=data_cons["risk_aversion"],
        uncertainty_radius=0.0,
    )

    cons_sol = rp_exact_mod.solve_mean_variance(cons_instance)
    results["Conservative"] = {
        "weights": cons_sol.weights,
        "expected_return": cons_sol.expected_return,
        "portfolio_std": cons_sol.portfolio_std,
        "objective": cons_sol.objective,
    }

    # ── Equal allocation (1/n baseline) ──────────────────────────────────
    ew_sol = rp_heur_mod.equal_weight(mv_instance)
    results["Equal-Allocation"] = {
        "weights": ew_sol.weights,
        "expected_return": ew_sol.expected_return,
        "portfolio_std": ew_sol.portfolio_std,
        "objective": ew_sol.objective,
    }

    # ── Min-Variance (most risk-averse) ──────────────────────────────────
    minvar_sol = rp_heur_mod.min_variance(mv_instance)
    results["Min-Variance"] = {
        "weights": minvar_sol.weights,
        "expected_return": minvar_sol.expected_return,
        "portfolio_std": minvar_sol.portfolio_std,
        "objective": minvar_sol.objective,
    }

    # ── Max-Return (all-in on best crop) ─────────────────────────────────
    maxret_sol = rp_heur_mod.max_return(mv_instance)
    results["Max-Return"] = {
        "weights": maxret_sol.weights,
        "expected_return": maxret_sol.expected_return,
        "portfolio_std": maxret_sol.portfolio_std,
        "objective": maxret_sol.objective,
    }

    if verbose:
        total_acres = data_mv["total_acres"]
        print("=" * 70)
        print("CROP PORTFOLIO SELECTION UNDER WEATHER UNCERTAINTY")
        print(f"  Total farmland: {total_acres} acres")
        print(f"  {data_mv['n_crops']} candidate crops")
        print("=" * 70)

        print("\n  Crop economics:")
        for i, crop in enumerate(CROPS):
            print(f"    {crop:20s}  E[profit]=${EXPECTED_PROFIT_PER_ACRE[i]:6.0f}/ac  "
                  f"Std=${PROFIT_VOLATILITY[i]:5.0f}/ac")

        for method, res in results.items():
            # Convert portfolio weights to actual farm allocation
            w = res["weights"]
            # Expected farm profit = sum of (weight_i * profit_per_acre_i * total_acres)
            exp_profit = float(np.dot(w, EXPECTED_PROFIT_PER_ACRE)) * total_acres
            # Portfolio std in dollar terms
            vol_per_acre = PROFIT_VOLATILITY
            cov_dollars = np.outer(vol_per_acre, vol_per_acre) * CROP_CORRELATIONS
            var_farm = float(w @ cov_dollars @ w) * (total_acres ** 2)
            std_profit = np.sqrt(max(0, var_farm))

            print(f"\n--- {method} ---")
            print(f"  Expected farm profit: ${exp_profit:>10,.0f}/year")
            print(f"  Profit std deviation: ${std_profit:>10,.0f}/year")
            print(f"  Allocation:")
            for i, crop in enumerate(CROPS):
                wi = w[i]
                if wi > 0.005:
                    acres = wi * total_acres
                    crop_profit = EXPECTED_PROFIT_PER_ACRE[i] * acres
                    print(f"    {crop:20s} {wi:6.1%}  ({acres:5.0f} acres)  "
                          f"E[profit]=${crop_profit:>10,.0f}")

        # Risk-return summary
        print("\n--- Risk-Return Summary ---")
        print(f"  {'Method':20s} {'E[Profit]':>12s} {'Std':>12s} {'Crops':>6s}")
        for method, res in results.items():
            w = res["weights"]
            exp_p = float(np.dot(w, EXPECTED_PROFIT_PER_ACRE)) * total_acres
            cov_d = np.outer(PROFIT_VOLATILITY, PROFIT_VOLATILITY) * CROP_CORRELATIONS
            std_p = np.sqrt(max(0, float(w @ cov_d @ w))) * total_acres
            n_crops = sum(1 for wi in w if wi > 0.01)
            print(f"  {method:20s} ${exp_p:>10,.0f} ${std_p:>10,.0f} {n_crops:>5d}")

    return results


if __name__ == "__main__":
    solve_crop_selection()
