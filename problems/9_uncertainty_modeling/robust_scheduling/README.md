# Robust Single Machine Scheduling

## Problem Definition

Schedule $n$ jobs on a single machine with **uncertain processing times** to minimize the worst-case regret of total weighted completion time $\sum w_j C_j$.

## Robustness Criteria

| Criterion | Formulation |
|-----------|-------------|
| Min-Max Regret | $\min_\pi \max_s [\sum w_j C_j(\pi,s) - \sum w_j C_j(\pi^*_s, s)]$ |
| Expected Cost | $\min_\pi E[\sum w_j C_j]$ |

## Solution Approaches

| Method | Type | Description |
|--------|------|-------------|
| Midpoint WSPT | Heuristic | WSPT on mean processing times |
| Scenario Enumeration | Heuristic | WSPT per scenario, cross-evaluate regret |
| Worst-Case WSPT | Heuristic | WSPT on maximum processing times |
| Simulated Annealing | Metaheuristic | Swap/insertion to minimize max regret |

## Key References

- Kouvelis, P. & Yu, G. (1997). *Robust Discrete Optimization*. Springer.
- Kasperski, A. & Zielinski, P. (2008). A 2-approximation for interval minmax regret. *ORL*, 36(5), 561-564.
