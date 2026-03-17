# Stochastic Knapsack Problem

## Problem Definition

Select items with deterministic values $v_i$ but **random weights** $W_i(s)$ to maximize total value subject to a capacity constraint that holds with high probability.

## Formulations

| Variant | Constraint | Description |
|---------|-----------|-------------|
| Expected capacity | $E[\sum w_i x_i] \leq W$ | Average-case feasibility |
| Chance-constrained | $P(\sum w_i x_i \leq W) \geq 1-\alpha$ | High-probability feasibility |

## Solution Approaches

| Method | Type | Description |
|--------|------|-------------|
| Greedy (mean weight) | Heuristic | Value-density ranking on expected weights |
| Greedy (chance-constrained) | Heuristic | Add items maintaining P(feasible) >= 1-alpha |
| Simulated Annealing | Metaheuristic | Flip-bit neighborhood with infeasibility penalty |

## Key References

- Kleinberg, J., Rabani, Y. & Tardos, E. (1997). Allocating bandwidth for bursty connections. *STOC*, 664-673. https://doi.org/10.1145/258533.258661
- Dean, B.C., Goemans, M.X. & Vondrák, J. (2008). Approximating the stochastic knapsack problem. *Math. Oper. Res.*, 33(1), 1-14. https://doi.org/10.1287/moor.1070.0285
