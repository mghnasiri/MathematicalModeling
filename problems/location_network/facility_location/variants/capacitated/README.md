# Capacitated Facility Location Problem (CFLP)

## Problem Definition

Extends UFLP: each facility i has **maximum capacity u_i** limiting total demand it can serve.

```
min  Σ f_i * y_i + Σ c_ij * x_ij
s.t. Σ x_ij = 1               ∀ j (each customer assigned)
     Σ d_j * x_ij ≤ u_i * y_i  ∀ i (capacity constraints)
     y_i ∈ {0,1}, x_ij ≥ 0
```

## Complexity

NP-hard (harder than UFLP due to capacity constraints).

## Applications

- **Warehouse location**: distribution centers with limited throughput
- **Server placement**: data centers with bandwidth/CPU limits
- **Hospital siting**: emergency facilities with bed capacity
- **School districting**: schools with enrollment limits

## Algorithms

| Algorithm | Type | Reference |
|-----------|------|-----------|
| Greedy-add (capacity-aware) | Heuristic | Cornuéjols et al. (1991) |
| Simulated Annealing | Metaheuristic | Kirkpatrick et al. (1983) |
