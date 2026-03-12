# Robust Shortest Path Problem

## Problem Definition

Given a directed graph $G = (V, E)$ with **uncertain** edge weights described by $S$ scenarios, find the path from source $s$ to target $t$ that is best under the worst-case realization.

## Robustness Criteria

| Criterion | Formulation | Complexity |
|-----------|-------------|------------|
| **Min-Max Cost** | $\min_P \max_{s \in S} \text{cost}_s(P)$ | Polynomial (discrete scenarios) |
| **Min-Max Regret** | $\min_P \max_{s \in S} [\text{cost}_s(P) - \text{cost}_s(P^*_s)]$ | NP-hard (general intervals) |
| **Expected Cost** | $\min_P \sum_s p_s \cdot \text{cost}_s(P)$ | Polynomial (weighted Dijkstra) |

## Solution Approaches

| Method | Criterion | Description |
|--------|-----------|-------------|
| Label-Setting | Min-Max Cost | Multi-objective Dijkstra with dominance pruning |
| Scenario Enumeration | Min-Max Cost | Solve Dijkstra per scenario, cross-evaluate |
| Regret Enumeration | Min-Max Regret | Evaluate candidate paths against per-scenario optima |
| Midpoint Heuristic | Min-Max Regret | Shortest path on mean-weight graph |

## Key References

- Kouvelis, P. & Yu, G. (1997). *Robust Discrete Optimization and Its Applications*. Springer. https://doi.org/10.1007/978-1-4757-2620-6
- Bertsimas, D. & Sim, M. (2003). Robust discrete optimization and network flows. *Math. Program.*, 98(1-3), 49-71. https://doi.org/10.1007/s10107-003-0396-4
- Averbakh, I. & Lebedev, V. (2004). Interval data minmax regret network optimization problems. *DAM*, 138(3), 289-301. https://doi.org/10.1016/S0166-218X(03)00462-1
