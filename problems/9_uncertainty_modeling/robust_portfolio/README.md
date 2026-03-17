# Robust Portfolio Optimization

## Problem Definition

Select asset weights $w$ to maximize risk-adjusted return under uncertainty in expected returns $\mu$.

### Classical Markowitz

$$\max_w \ \mu^T w - \lambda \, w^T \Sigma w \quad \text{s.t.} \ \sum w_i = 1, \ w \geq 0$$

### Robust (Ellipsoidal Uncertainty)

$$\max_w \ \hat{\mu}^T w - \delta \|\Sigma^{1/2} w\|_2 - \lambda \, w^T \Sigma w \quad \text{s.t.} \ \sum w_i = 1, \ w \geq 0$$

## Solution Approaches

| Method | Type | Description |
|--------|------|-------------|
| QP Solver (mean-variance) | Exact | SLSQP on Markowitz objective |
| QP Solver (robust) | Exact | SLSQP with uncertainty penalty (SOCP) |
| Equal Weight (1/n) | Heuristic | Naive diversification, surprisingly competitive |
| Min Variance | Heuristic | $\Sigma^{-1} \mathbf{1}$ closed-form |
| Max Return | Heuristic | Concentrate on best expected return |

## Key References

- Markowitz, H. (1952). Portfolio selection. *J. Finance*, 7(1), 77-91. https://doi.org/10.2307/2975974
- Goldfarb, D. & Iyengar, G. (2003). Robust portfolio selection problems. *Math. Oper. Res.*, 28(1), 1-38. https://doi.org/10.1287/moor.28.1.1.14260
- DeMiguel, V. et al. (2009). Optimal versus naive diversification. *Rev. Financ. Stud.*, 22(5), 1915-1953. https://doi.org/10.1093/rfs/hhm075
