# Nonlinear Programming (NLP)

## 1. Problem Definition

- **Input:** Objective function $f: \mathbb{R}^n \to \mathbb{R}$, inequality constraints $g_i(x) \leq 0$, equality constraints $h_j(x) = 0$, variable bounds, initial point $x_0$
- **Decision:** $x \in \mathbb{R}^n$
- **Objective:** Minimize $f(x)$
- **Classification:** NP-hard in general. Polynomial for convex instances via interior-point methods.

---

## 2. Mathematical Formulation

### Notation Table

| Symbol | Definition |
|--------|-----------|
| $n$ | Number of decision variables |
| $f(x)$ | Objective function |
| $g_i(x)$ | Inequality constraint functions ($g_i(x) \leq 0$) |
| $h_j(x)$ | Equality constraint functions ($h_j(x) = 0$) |
| $x_0$ | Initial guess (starting point) |

### General Form

$$\min_{x \in \mathbb{R}^n} \quad f(x) \tag{1}$$

$$g_i(x) \leq 0 \quad i = 1, \ldots, m \tag{2}$$

$$h_j(x) = 0 \quad j = 1, \ldots, p \tag{3}$$

$$\ell \leq x \leq u \tag{4}$$

### KKT Conditions

$$\nabla f(x^*) + \sum_i \lambda_i \nabla g_i(x^*) + \sum_j \nu_j \nabla h_j(x^*) = 0$$

$$\lambda_i g_i(x^*) = 0, \quad \lambda_i \geq 0 \quad \forall i$$

### Small Illustrative Instance

```
min  (x₁ - 1)² + (x₂ - 2.5)²
s.t. x₁ - 2x₂ + 2 ≥ 0
     -x₁ - 2x₂ + 6 ≥ 0
     x₁, x₂ ≥ 0

Optimal: x* ≈ (1.4, 1.7), f* ≈ 0.80
```

---

## 3. Solution Methods

| Method | Type | Complexity | Description |
|--------|------|-----------|-------------|
| SLSQP (SciPy) | Exact (local) | Depends on problem | Sequential Least Squares Programming |
| trust-constr | Exact (local) | Depends on problem | Trust-region interior point |

Both methods find local optima; global optimality guaranteed only for convex problems.

---

## 4. Implementations in This Repository

```
nonlinear_programming/
├── instance.py                    # NLPInstance, NLPSolution
│                                  #   - Fields: objective, n_vars, x0, bounds,
│                                  #     ineq_constraints, eq_constraints, gradient
├── exact/
│   └── solve_nlp.py               # NLP solver via SciPy minimize
└── tests/
    └── test_nlp.py                # NLP test suite
```

---

## 5. Key References

- Nocedal, J. & Wright, S.J. (2006). *Numerical Optimization*. 2nd ed. Springer.
- Bazaraa, M.S., Sherali, H.D. & Shetty, C.M. (2006). *Nonlinear Programming: Theory and Algorithms*. 3rd ed. Wiley.
