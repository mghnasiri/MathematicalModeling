# Multi-dimensional Knapsack Problem (MKP)

## Problem Definition

Given **n** items, each with value v_i and a **d-dimensional weight vector** (w_i1, ..., w_id), and a knapsack with **d capacity constraints** (W_1, ..., W_d), select a subset to maximize total value:

```
max  Σ v_i * x_i
s.t. Σ w_ij * x_i ≤ W_j    ∀ j ∈ {1,...,d}
     x_i ∈ {0, 1}           ∀ i ∈ {1,...,n}
```

## Complexity

NP-hard (strongly for d ≥ 2). The pseudo-polynomial DP for 1D knapsack does not directly extend.

## Applications

- **Capital budgeting**: projects consume multiple resources (budget, labor, time)
- **Cargo loading**: items have weight, volume, and fragility constraints
- **Resource allocation**: tasks require CPU, memory, and bandwidth
- **Project selection**: proposals scored by value, constrained by budget across departments

## Algorithms

| Algorithm | Type | Reference |
|-----------|------|-----------|
| Pseudo-utility greedy | Heuristic | Pirkul (1987) |
| Max-value greedy | Heuristic | — |
| Genetic Algorithm | Metaheuristic | Chu & Beasley (1998) |
