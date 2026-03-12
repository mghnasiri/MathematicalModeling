# Distributionally Robust Optimization (DRO)

## Problem Definition

Optimize decisions under the worst-case distribution from an ambiguity set $\mathcal{A}$:

$$\min_x \max_{P \in \mathcal{A}} \mathbb{E}_P[f(x, \xi)]$$

## Ambiguity Sets

| Type | Description | Reformulation |
|------|-------------|---------------|
| Wasserstein ball | $W_1(P, \hat{P}) \leq \epsilon$ | LP with L1-norm regularization |
| Moment-based | Matching mean $\pm$ tolerance | LP over probability simplex |

## Solution Approaches

| Method | Ambiguity | Description |
|--------|-----------|-------------|
| Wasserstein LP | Wasserstein | Tractable LP reformulation |
| Nominal LP | None | Baseline without robustness |
| Moment DRO | Moment | Grid search + inner LP for worst-case distribution |

## Key References

- Esfahani, P.M. & Kuhn, D. (2018). Data-driven DRO using Wasserstein metric. *Math. Program.*, 171, 115-166.
- Delage, E. & Ye, Y. (2010). DRO under moment uncertainty. *Oper. Res.*, 58(3), 595-612.
