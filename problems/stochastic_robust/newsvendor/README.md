# Newsvendor Problem (Single-Period Stochastic Inventory)

## Problem Definition

A retailer must decide how many units $Q$ of a perishable product to order **before** observing uncertain demand $D$. Each unsold unit incurs overage cost $c_o = c - v$; each unit of unmet demand incurs underage cost $c_u = p - c$.

## Mathematical Formulation

$$\min_Q \ E\left[c_o \cdot \max(0, Q - D) + c_u \cdot \max(0, D - Q)\right]$$

**Optimal solution** (critical fractile): order $Q^*$ such that

$$P(D \leq Q^*) = \frac{c_u}{c_u + c_o} = \frac{p - c}{p - v}$$

## Parameters

| Symbol | Description |
|--------|-------------|
| $c$ | Unit purchase cost |
| $p$ | Selling price per unit ($p > c$) |
| $v$ | Salvage value per unsold unit ($v < c$) |
| $D$ | Random demand |

## Complexity

$O(1)$ for continuous distributions (closed-form quantile), $O(S \log S)$ for $S$ discrete scenarios.

## Solution Approaches

| Method | Type | Description |
|--------|------|-------------|
| Critical Fractile | Exact | Sort scenarios, scan CDF to critical ratio |
| Grid Search | Exact | Brute-force evaluation over demand range |
| Marginal Allocation | Heuristic | Multi-product with budget constraint |
| Independent + Scale | Heuristic | Solve independently, scale to fit budget |

## Key References

- Arrow, K.J., Harris, T. & Marschak, J. (1951). Optimal inventory policy. *Econometrica*, 19(3), 250-272. https://doi.org/10.2307/1906813
- Silver, E.A., Pyke, D.F. & Thomas, D.J. (2017). *Inventory and Production Management in Supply Chains*, 4th ed. CRC Press.
- Hadley, G. & Whitin, T.M. (1963). *Analysis of Inventory Systems*. Prentice-Hall.
