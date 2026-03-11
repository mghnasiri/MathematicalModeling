# No-Wait Job Shop Scheduling (Jm | no-wait | Cmax)

## Problem Definition

Extends JSP: once a job starts processing, it must proceed through all its operations without any idle time between consecutive operations. The start time of a job determines all its operation times.

```
min  Cmax
s.t. start(j, k+1) = end(j, k)        ∀ j, k  (no-wait)
     no two operations on same machine overlap
```

## Complexity

NP-hard (generalizes JSP).

## Applications

- **Steel manufacturing**: continuous casting requires uninterrupted processing
- **Chemical processing**: reactions cannot be paused
- **Food processing**: perishable goods require continuous processing
- **Pharmaceutical manufacturing**: drug production with timing constraints

## Algorithms

| Algorithm | Type | Reference |
|-----------|------|-----------|
| Greedy Insertion | Heuristic | Mascis & Pacciarelli (2002) |
| SPT Schedule | Heuristic | — |
| Simulated Annealing | Metaheuristic | Mascis & Pacciarelli (2002) |
