# Quadratic Programming (QP)

## 1. Problem Definition

- **Input:** Symmetric matrix $Q \in \mathbb{R}^{n \times n}$, linear cost $c \in \mathbb{R}^n$, constraints $A_{\text{ub}} x \leq b_{\text{ub}}$, $A_{\text{eq}} x = b_{\text{eq}}$, variable bounds
- **Decision:** $x \in \mathbb{R}^n$
- **Objective:** Minimize $\frac{1}{2} x^T Q x + c^T x$
- **Classification:** Polynomial for convex QP ($Q \succeq 0$) via interior-point methods; NP-hard for non-convex QP

---

## 2. Mathematical Formulation

### Notation Table

| Symbol | Definition |
|--------|-----------|
| $n$ | Number of decision variables |
| $Q$ | Symmetric PSD quadratic cost matrix |
| $c$ | Linear cost vector |
| $A_{\text{ub}}, b_{\text{ub}}$ | Inequality constraints |
| $A_{\text{eq}}, b_{\text{eq}}$ | Equality constraints |

### Standard Form

$$\min \quad \frac{1}{2} x^T Q x + c^T x \tag{1}$$

$$A_{\text{ub}} x \leq b_{\text{ub}} \tag{2}$$

$$A_{\text{eq}} x = b_{\text{eq}} \tag{3}$$

$$\ell \leq x \leq u \tag{4}$$

### KKT Conditions

For convex QP, the KKT conditions are necessary and sufficient:

$$Qx^* + c + A_{\text{ub}}^T \lambda + A_{\text{eq}}^T \nu = 0$$

with complementary slackness $\lambda_i (a_i^T x^* - b_i) = 0, \lambda \geq 0$.

#### KKT Verification Pseudocode

```
VERIFY_KKT(x*, λ*, ν*, Q, c, A_ub, b_ub, A_eq, b_eq):
    # 1. Stationarity
    gradient = Q @ x* + c + A_ub.T @ λ* + A_eq.T @ ν*
    assert ||gradient|| ≈ 0

    # 2. Primal feasibility
    assert A_ub @ x* <= b_ub  (elementwise)
    assert A_eq @ x* = b_eq

    # 3. Dual feasibility
    assert λ* >= 0  (elementwise)

    # 4. Complementary slackness
    for each inequality i:
        assert λ*_i * (a_i^T x* - b_i) = 0

    return OPTIMAL
```

The KKT system forms the basis for active-set methods (identify binding constraints, solve reduced system) and interior-point methods (apply barrier functions to relax complementarity).

### Small Illustrative Instance

```
min  x₁² + x₂² - 2x₁  (Q = 2I, c = [-2, 0])
s.t. x₁ + x₂ ≤ 2, x₁, x₂ ≥ 0

Optimal: x* = (1, 0), z* = -1
```

---

## 3. Solution Methods

| Method | Type | Complexity | Description |
|--------|------|-----------|-------------|
| SLSQP (SciPy) | Exact | Polynomial (convex) | `scipy.optimize.minimize` with method='SLSQP' |
| Active-set | Exact | Polynomial (convex) | Identifies binding constraints iteratively |
| Interior-point | Exact | $O(n^{3.5} L)$ | Barrier method for convex QP |

### Special Cases

- **Unconstrained QP:** $x^* = -Q^{-1}c$ when $Q \succ 0$ (positive definite). Reduces to a linear system solve.
- **Equality-constrained QP:** Solved via the KKT linear system in $O(n^3)$.
- **Portfolio optimization:** QP with $Q = \Sigma$ (covariance), $c = -\mu$ (expected returns), and simplex/budget constraints.

### Applications

- **Markowitz portfolio optimization** (mean-variance tradeoff)
- **Model predictive control** (trajectory optimization in robotics)
- **Support vector machines** (SVM training as convex QP)
- **Structural engineering** (minimum weight design under stress constraints)

---

## 4. Implementations in This Repository

```
quadratic_programming/
├── instance.py                    # QPInstance, QPSolution
│                                  #   - Fields: n, Q, c, A_ub, b_ub, A_eq, b_eq, bounds
├── exact/
│   └── qp_solver.py               # QP solver via SciPy SLSQP
└── tests/
    └── test_qp.py                 # QP test suite
```

---

## 5. Key References

- Nocedal, J. & Wright, S.J. (2006). *Numerical Optimization*. 2nd ed. Springer.
- Boyd, S. & Vandenberghe, L. (2004). *Convex Optimization*. Cambridge University Press.
- Goldfarb, D. & Idnani, A. (1983). A numerically stable dual method for solving strictly convex quadratic programs. *Math. Programming*, 27(1), 1-33.
- Markowitz, H. (1952). Portfolio selection. *Journal of Finance*, 7(1), 77-91.
