# Newsvendor Problem (Single-Period Stochastic Inventory)

## 1. Problem Definition

- **Input:** Unit cost $c$, selling price $p > c$, salvage value $v < c$, random demand $D$
- **Decision:** Order quantity $Q$ before demand is realized
- **Objective:** Minimize expected cost $E[c_o \cdot (Q-D)^+ + c_u \cdot (D-Q)^+]$
  - Overage cost: $c_o = c - v$ (unsold units)
  - Underage cost: $c_u = p - c$ (unmet demand)
- **Classification:** Stochastic optimization. $O(1)$ for continuous distributions; $O(S \log S)$ for discrete scenarios.

### Optimal Solution (Critical Fractile)

$$Q^* = F^{-1}\!\left(\frac{c_u}{c_u + c_o}\right) = F^{-1}\!\left(\frac{p - c}{p - v}\right)$$

The critical ratio $\frac{c_u}{c_u + c_o}$ is the optimal service level.

---

## 2. Mathematical Formulation

### Notation Table

| Symbol | Definition |
|--------|-----------|
| $c$ | Unit purchase cost |
| $p$ | Selling price ($p > c$) |
| $v$ | Salvage value ($v < c$) |
| $D$ | Random demand (CDF $F$) |
| $c_u = p - c$ | Underage cost |
| $c_o = c - v$ | Overage cost |

### Expected Cost Formulation

$$\min_Q \quad E\bigl[c_o \cdot \max(0, Q - D) + c_u \cdot \max(0, D - Q)\bigr]$$

This is a convex function of $Q$; the FOC gives the critical fractile.

### Small Illustrative Instance

```
c = 10, p = 25, v = 5
c_u = 15, c_o = 5
Critical ratio = 15/20 = 0.75
If D ~ U[0, 100]: Q* = 75
```

---

## 3. Solution Methods

| Method | Type | Complexity | Description |
|--------|------|-----------|-------------|
| Critical Fractile | Exact | $O(S \log S)$ | Sort scenarios, scan CDF to critical ratio |
| Grid Search | Exact | $O(S \cdot G)$ | Brute-force over demand range |
| Marginal Allocation | Heuristic | $O(n \cdot B)$ | Multi-product with budget constraint |
| Independent + Scale | Heuristic | $O(n)$ | Solve independently, scale to fit budget |

---

## 4. Implementations in This Repository

```
newsvendor/
├── instance.py                    # NewsvendorInstance, critical fractile
├── exact/
│   └── critical_fractile.py       # Critical fractile + grid search
├── heuristics/
│   └── multi_product.py           # Marginal allocation, independent+scale
└── tests/
    └── test_newsvendor.py         # 13 tests
```

---

## 5. Key References

- Arrow, K.J., Harris, T. & Marschak, J. (1951). Optimal inventory policy. *Econometrica*, 19(3), 250-272.
- Silver, E.A., Pyke, D.F. & Thomas, D.J. (2017). *Inventory and Production Management in Supply Chains* (4th ed.). CRC Press.
