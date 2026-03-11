# Multiple Knapsack Problem (mKP)

## Problem Definition

Given n items with weights w_j and values v_j, and k knapsacks with capacities C_1, ..., C_k, assign items to knapsacks to maximize total value:

```
max  Σ_j v_j * x_j
s.t. Σ_{j: a_j=i} w_j ≤ C_i    ∀ i ∈ {1,...,k}  (capacity)
     a_j ∈ {-1, 0, ..., k-1}    ∀ j               (assignment or skip)
```

## Complexity

NP-hard (generalizes 0-1 Knapsack when k=1).

## Applications

- **Cargo loading**: distribute goods across multiple vehicles
- **Budget allocation**: allocate projects to multiple funding sources
- **Cloud computing**: assign tasks to servers with resource limits
- **Portfolio management**: distribute investments across accounts

## Algorithms

| Algorithm | Type | Reference |
|-----------|------|-----------|
| Greedy Value-Density | Heuristic | Martello & Toth (1990) |
| Greedy Best-Fit | Heuristic | Martello & Toth (1990) |
| Genetic Algorithm | Metaheuristic | Chu & Beasley (1998) |
