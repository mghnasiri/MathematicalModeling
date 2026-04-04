# Robust Portfolio Optimization

## 1. Problem Definition

- **Input:** $n$ assets with estimated returns $\hat{\mu}$, covariance matrix $\Sigma$, risk-aversion parameter $\lambda$, uncertainty radius $\delta$
- **Decision:** Portfolio weights $w \in \mathbb{R}^n$ with $\sum w_i = 1, w \geq 0$
- **Objective:** Maximize risk-adjusted return under worst-case return uncertainty
- **Classification:** Convex (SOCP) — solvable in polynomial time

---

## 2. Mathematical Formulation

### Notation Table

| Symbol | Definition |
|--------|-----------|
| $n$ | Number of assets |
| $w \in \mathbb{R}^n$ | Portfolio weight vector |
| $\hat{\mu} \in \mathbb{R}^n$ | Estimated expected returns |
| $\Sigma \in \mathbb{R}^{n \times n}$ | Return covariance matrix |
| $\lambda$ | Risk-aversion parameter |
| $\delta$ | Uncertainty radius (size of ambiguity set) |

### Classical Markowitz (Mean-Variance)

$$\max_w \quad \hat{\mu}^T w - \lambda \, w^T \Sigma w \tag{1}$$

$$\sum_{i=1}^{n} w_i = 1, \quad w \geq 0 \tag{2}$$

### Robust Formulation (Ellipsoidal Uncertainty)

The true returns $\mu$ lie in an ellipsoidal set $\mathcal{U} = \{\mu : \|\Sigma^{-1/2}(\mu - \hat{\mu})\|_2 \leq \delta\}$. Optimizing against the worst-case $\mu \in \mathcal{U}$:

$$\max_w \quad \hat{\mu}^T w - \delta \|\Sigma^{1/2} w\|_2 - \lambda \, w^T \Sigma w \tag{3}$$

$$\sum_{i=1}^{n} w_i = 1, \quad w \geq 0 \tag{4}$$

The penalty term $\delta \|\Sigma^{1/2} w\|_2$ shrinks allocations to assets with high estimation uncertainty.

### Small Illustrative Instance

```
n = 3 assets
μ̂ = [0.10, 0.06, 0.04]   (expected returns)
Σ = [[0.04, 0.01, 0.00],  (covariance)
     [0.01, 0.02, 0.005],
     [0.00, 0.005, 0.01]]
λ = 1.0, δ = 0.0 (Markowitz)

Optimal (MV): w* ≈ [0.25, 0.35, 0.40] — diversified
With δ = 0.5 (robust): shifts weight toward lower-variance assets
```

---

## 3. Solution Methods

| Method | Type | Complexity | Description |
|--------|------|-----------|-------------|
| QP Solver (mean-variance) | Exact | $O(n^3)$ | SLSQP on Markowitz objective |
| QP Solver (robust) | Exact | $O(n^3)$ | SLSQP with uncertainty penalty (SOCP) |
| Equal Weight (1/n) | Heuristic | $O(1)$ | Naive diversification, surprisingly competitive |
| Min Variance | Heuristic | $O(n^3)$ | $\Sigma^{-1} \mathbf{1}$ closed-form, ignores returns |
| Max Return | Heuristic | $O(n)$ | Concentrate on highest expected return |

### QP Solver

Uses `scipy.optimize.minimize` with SLSQP method. For the robust variant, the $\|\Sigma^{1/2} w\|_2$ term makes the problem second-order cone (SOCP), but SLSQP handles it via gradient-based optimization.

### Equal Weight (1/n)

Allocates $w_i = 1/n$ for all assets. DeMiguel et al. (2009) showed this outperforms many optimized portfolios out-of-sample, especially when $n$ is large relative to the estimation sample.

---

## 4. Implementations in This Repository

```
robust_portfolio/
├── instance.py                    # RobustPortfolioInstance, PortfolioSolution
│                                  #   - portfolio_return(), portfolio_risk()
│                                  #   - mean_variance_objective(), robust_objective()
│                                  #   - random() factory
├── exact/
│   └── quadratic_solver.py        # Mean-variance QP + robust SOCP via SLSQP
├── heuristics/
│   └── equal_weight.py            # Equal-weight, min-variance, max-return
└── tests/
    └── test_portfolio.py          # 14 tests, 3 test classes
```

---

## 5. Key References

- Markowitz, H. (1952). Portfolio selection. *J. Finance*, 7(1), 77-91. https://doi.org/10.2307/2975974
- Goldfarb, D. & Iyengar, G. (2003). Robust portfolio selection problems. *Math. Oper. Res.*, 28(1), 1-38. https://doi.org/10.1287/moor.28.1.1.14260
- DeMiguel, V., Garlappi, L. & Uppal, R. (2009). Optimal versus naive diversification: How inefficient is the 1/N portfolio strategy? *Rev. Financ. Stud.*, 22(5), 1915-1953. https://doi.org/10.1093/rfs/hhm075
- Bertsimas, D., Brown, D.B. & Caramanis, C. (2011). Theory and applications of robust optimization. *SIAM Review*, 53(3), 464-501. https://doi.org/10.1137/080734510
