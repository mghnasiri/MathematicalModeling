# Distributionally Robust Optimization (DRO)

## 1. Problem Definition

- **Input:** Decision dimension $n$, nominal cost vector $c$, discrete support points $\xi_k$ ($K$ points), nominal distribution $\hat{P}$, ambiguity set parameters ($\epsilon$ for Wasserstein, mean tolerance for moment-based)
- **Decision:** Decision vector $x$ (subject to $Ax \leq b$)
- **Objective:** Minimize worst-case expected cost over all distributions in the ambiguity set: $\min_x \max_{P \in \mathcal{A}} E_P[f(x, \xi)]$
- **Classification:** Convex (finite-dimensional reformulation). LP for linear cost with Wasserstein/moment ambiguity.

---

## 2. Mathematical Formulation

### Notation Table

| Symbol | Definition |
|--------|-----------|
| $x \in \mathbb{R}^n$ | Decision vector |
| $\xi \in \mathbb{R}^n$ | Random perturbation vector |
| $c \in \mathbb{R}^n$ | Nominal cost vector |
| $f(x, \xi) = (c + \xi)^T x$ | Cost function |
| $\mathcal{A}$ | Ambiguity set of distributions |
| $K$ | Number of support points |
| $\hat{P}$ | Nominal (empirical) distribution |
| $\epsilon$ | Wasserstein ball radius |

### General DRO Formulation

$$\min_x \max_{P \in \mathcal{A}} \mathbb{E}_P\bigl[(c + \xi)^T x\bigr] \tag{1}$$

$$Ax \leq b \tag{2}$$

### Ambiguity Sets

| Type | Description | Reformulation |
|------|-------------|---------------|
| **Wasserstein ball** | $\{P : W_1(P, \hat{P}) \leq \epsilon\}$ | LP with L1-norm regularization |
| **Moment-based** | $\{P : \|E_P[\xi] - \hat{\mu}\|_\infty \leq \tau\}$ | LP over probability simplex |

### Wasserstein DRO Dual

For the 1-Wasserstein ball with discrete support, the inner maximization over $P$ becomes:

$$\max_{P \in \mathcal{A}} E_P[f(x,\xi)] = \hat{E}[f(x,\xi)] + \epsilon \cdot \|x\|_* $$

where $\|x\|_*$ is the dual norm. This adds a regularization term proportional to $\epsilon$.

### Small Illustrative Instance

```
n = 2, K = 3 support points
c = [3, 5], x ∈ [0, 1]²
Support: ξ₁ = [-1, 0], ξ₂ = [0, 1], ξ₃ = [1, -1]
Nominal: P̂ = [1/3, 1/3, 1/3]
ε = 0.5 (Wasserstein radius)

Nominal cost at x = [0.5, 0.3]:
  E[(c+ξ)ᵀx] = 1/3[(2)(0.5)+(5)(0.3)] + 1/3[(3)(0.5)+(6)(0.3)] + 1/3[(4)(0.5)+(4)(0.3)]
             = 1/3[2.5] + 1/3[3.3] + 1/3[3.2] = 3.0

Worst-case distribution shifts mass toward ξ₂ → higher cost
```

---

## 3. Solution Methods

| Method | Ambiguity | Type | Description |
|--------|-----------|------|-------------|
| Wasserstein LP | Wasserstein | Exact | Tractable LP reformulation via duality |
| Nominal LP | None | Exact | Baseline — optimize under $\hat{P}$ only |
| Moment DRO | Moment | Heuristic | Grid search over $x$ + inner LP for worst-case distribution |

### Wasserstein LP

Reformulates the DRO as a finite LP by taking the dual of the inner maximization. The Wasserstein penalty acts as regularization, producing solutions that are robust to distribution shift.

### Moment-Based DRO

For each candidate $x$ (from a grid), solve an inner LP over the probability simplex to find the worst-case distribution matching the moment constraints. Select $x$ with minimum worst-case cost.

---

## 4. Implementations in This Repository

```
dro/
├── instance.py                    # DROInstance, DROSolution
│                                  #   - cost(), nominal_expected_cost(), worst_case_cost()
│                                  #   - random() factory
├── exact/
│   └── wasserstein_dro.py         # Wasserstein LP reformulation + nominal LP baseline
├── heuristics/
│   └── moment_dro.py              # Moment-based DRO with inner LP, grid search
└── tests/
    └── test_dro.py                # 12 tests, 4 test classes
```

---

## 5. Key References

- Delage, E. & Ye, Y. (2010). Distributionally robust optimization under moment uncertainty with application to data-driven problems. *Oper. Res.*, 58(3), 595-612. https://doi.org/10.1287/opre.1090.0741
- Esfahani, P.M. & Kuhn, D. (2018). Data-driven distributionally robust optimization using the Wasserstein metric: performance guarantees and tractable reformulations. *Math. Program.*, 171(1-2), 115-166. https://doi.org/10.1007/s10107-017-1172-1
- Rahimian, H. & Mehrotra, S. (2022). Frameworks and results in distributionally robust optimization. *Open J. Math. Optim.*, 3(4), 1-85. https://doi.org/10.5802/ojmo.15
- Wiesemann, W., Kuhn, D. & Sim, M. (2014). Distributionally robust convex optimization. *Oper. Res.*, 62(6), 1358-1376. https://doi.org/10.1287/opre.2014.1314
