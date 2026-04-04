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
