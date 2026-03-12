# Two-Stage Stochastic Programming (2SSP)

## Problem Definition

Make first-stage decisions $x$ **before** uncertainty is revealed, then take recourse actions $y(s)$ for each realized scenario $s$ to compensate.

## Mathematical Formulation

$$\min \ c^T x + \sum_{s \in S} p_s \, q(s)^T y(s)$$

$$\text{s.t.} \quad Ax = b \quad \text{(first-stage)}$$

$$T(s)x + W(s)y(s) \leq h(s), \quad y(s) \geq 0 \quad \forall s \in S \quad \text{(second-stage)}$$

$$x \geq 0$$

## Key Concepts

| Term | Definition |
|------|------------|
| **Recourse** | Second-stage corrective action after uncertainty is observed |
| **Deterministic Equivalent** | Large LP expanding all scenarios explicitly |
| **Expected Value (EV)** | Solution using mean scenario — lower bound |
| **Value of Stochastic Solution (VSS)** | Benefit of stochastic over deterministic approach |
| **SAA** | Sample Average Approximation — solve on random subset of scenarios |

## Complexity

The deterministic equivalent LP has $n_1 + S \cdot n_2$ variables and $m_1 + S \cdot m_2$ constraints. Polynomial in the LP size, but grows linearly with $S$.

## Solution Approaches

| Method | Type | Description |
|--------|------|-------------|
| Deterministic Equivalent | Exact | Expand all scenarios into one large LP (HiGHS) |
| Expected Value Solution | Heuristic | Solve with mean scenario as proxy |
| Sample Average Approximation | Metaheuristic | Solve smaller sampled problems, replicate for CI |

## Key References

- Birge, J.R. & Louveaux, F. (2011). *Introduction to Stochastic Programming*, 2nd ed. Springer. https://doi.org/10.1007/978-1-4614-0237-4
- Dantzig, G.B. (1955). Linear programming under uncertainty. *Management Science*, 1(3-4), 197-206. https://doi.org/10.1287/mnsc.1.3-4.197
- Kleywegt, A.J., Shapiro, A. & Homem-de-Mello, T. (2002). The sample average approximation method. *SIAM J. Optim.*, 12(2), 479-502. https://doi.org/10.1137/S1052623499363220
