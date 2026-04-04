# Linear Programming (LP) with Sensitivity Analysis

## 1. Problem Definition

- **Input:** Cost vector $c \in \mathbb{R}^n$, inequality constraints $A_{\text{ub}} x \leq b_{\text{ub}}$, equality constraints $A_{\text{eq}} x = b_{\text{eq}}$, variable bounds
- **Decision:** $x \in \mathbb{R}^n$
- **Objective:** Minimize $c^T x$
- **Classification:** Polynomial via interior-point methods; simplex is exponential worst-case but fast in practice

---

## 2. Mathematical Formulation

### Notation Table

| Symbol | Definition |
|--------|-----------|
| $n$ | Number of decision variables |
| $c$ | Objective coefficient vector |
| $A_{\text{ub}}, b_{\text{ub}}$ | Inequality constraint matrix and RHS |
| $A_{\text{eq}}, b_{\text{eq}}$ | Equality constraint matrix and RHS |

### Standard Form

$$\min \quad c^T x \tag{1}$$

$$A_{\text{ub}} x \leq b_{\text{ub}} \tag{2}$$

$$A_{\text{eq}} x = b_{\text{eq}} \tag{3}$$

$$\ell \leq x \leq u \tag{4}$$

### Sensitivity Analysis

- **Shadow prices** (dual variables): marginal value of relaxing each constraint by one unit
- **Reduced costs**: how much the objective must change for a non-basic variable to enter the basis
- **Allowable ranges**: intervals for $c_j$ and $b_i$ within which the optimal basis remains unchanged

#### Sensitivity Analysis Detail

Given an optimal basis $B$, the sensitivity analysis examines how changes to the input parameters affect optimality:

1. **Right-hand side ranging:** For each constraint $i$, determine the range $[b_i^-, b_i^+]$ such that the current basis remains optimal. The shadow price $y_i$ gives $\Delta z^* = y_i \cdot \Delta b_i$ within this range.

2. **Objective coefficient ranging:** For each variable $j$, find the interval $[c_j^-, c_j^+]$ over which the current basis stays optimal. Outside this range, a different basis becomes optimal.

3. **New variable analysis:** A new variable with cost $c_{n+1}$ and column $a_{n+1}$ enters the basis if its reduced cost $\bar{c}_{n+1} = c_{n+1} - c_B^T B^{-1} a_{n+1} < 0$.

### Duality

The dual of the standard LP is:

$$\max \quad b_{\text{ub}}^T y$$

$$A_{\text{ub}}^T y \leq c, \quad y \geq 0$$

Strong duality holds: at optimality, $c^T x^* = b^T y^*$. The dual variables $y^*$ are the shadow prices.

### Small Illustrative Instance

```
min  -x₁ - 2x₂
s.t.  x₁ + x₂ ≤ 4
      x₁ - x₂ ≤ 2
      x₁, x₂ ≥ 0

Optimal: x* = (1, 3), z* = -7
Shadow prices: [1, 0.5] for the two inequalities
```

---

## 3. Solution Methods

| Method | Type | Complexity | Description |
|--------|------|-----------|-------------|
| HiGHS (via SciPy) | Exact | Polynomial (IPM) | `scipy.optimize.linprog` with sensitivity |
| Simplex method | Exact | $O(2^n)$ worst, fast in practice | Pivot-based, visits BFS vertices |
| Interior-point | Exact | $O(n^{3.5} L)$ | Barrier method, polynomial guarantee |

### Applications

- **Resource allocation** (production planning, workforce scheduling)
- **Supply chain optimization** (transportation and transshipment problems)
- **Diet and blending problems** (nutrient optimization at minimum cost)
- **Financial planning** (portfolio optimization under linear constraints)

---

## 4. Implementations in This Repository

```
linear_programming/
├── instance.py                    # LPInstance, LPSolution
│                                  #   - Fields: n, c, A_ub, b_ub, A_eq, b_eq, bounds
├── exact/
│   └── lp_solver.py               # LP solver with sensitivity analysis (HiGHS)
└── tests/
    └── test_lp.py                 # LP test suite
```

---

## 5. Key References

- Dantzig, G.B. (1963). *Linear Programming and Extensions*. Princeton University Press.
- Bertsimas, D. & Tsitsiklis, J.N. (1997). *Introduction to Linear Optimization*. Athena Scientific.
- Karmarkar, N. (1984). A new polynomial-time algorithm for linear programming. *Combinatorica*, 4(4), 373-395.
- Vanderbei, R.J. (2014). *Linear Programming: Foundations and Extensions*. 4th ed. Springer.
