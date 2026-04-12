# Multiple Knapsack Problem (MKP)

## 1. Problem Definition

- **Input:** $n$ items with weights $w_i$ and values $v_i$; $m$ knapsacks with capacities $C_j$
- **Decision:** Assign items to knapsacks (each item to at most one)
- **Objective:** Maximize total value of assigned items
- **Constraints:** Total weight per knapsack $\leq C_j$
- **Classification:** NP-hard (generalizes 0-1 knapsack)

The MKP generalizes both the 0-1 knapsack (single bin) and the bin packing problem
(uniform items). Applications include assigning jobs to machines with capacity
limits, distributing cargo across vehicles, and memory allocation in computing.

### Mathematical Formulation

$$\max \sum_{i=1}^{n} \sum_{j=1}^{m} v_i \cdot x_{ij}$$

$$\sum_{j=1}^{m} x_{ij} \leq 1 \quad \forall i \quad \text{(each item assigned at most once)}$$

$$\sum_{i=1}^{n} w_i \cdot x_{ij} \leq C_j \quad \forall j \quad \text{(capacity per knapsack)}$$

$$x_{ij} \in \{0, 1\}$$

---

## 2. Solution Methods

| Method | Type | Complexity | Description |
|--------|------|-----------|-------------|
| ILP (HiGHS) | Exact | Exponential worst | MILP via `scipy.optimize.milp` |
| Greedy MKP | Heuristic | $O(n \cdot m \log n)$ | Assign by value-density to largest-capacity knapsack |
| DP per knapsack | Exact | $O(m \cdot n \cdot W_{\max})$ | Solve each knapsack independently (ignores cross-knapsack interactions) |
| Mulknap B&B | Exact | Exponential worst | Pisinger (1999) specialized branch-and-bound |

### Greedy Assignment Pseudocode

The greedy heuristic sorts items by value-density, then assigns each item to the
knapsack with the largest remaining capacity that can fit it.

```
GREEDY-MKP(items, knapsacks):
    remaining[1..m] <- capacities[1..m]
    assignment <- {i: None for all items i}
    total_value <- 0

    Sort items by v[i] / w[i] descending

    for each item i in sorted order:
        // Find best knapsack: largest remaining capacity that fits item
        best_knapsack <- None
        best_remaining <- -1
        for j = 1 to m:
            if w[i] <= remaining[j] and remaining[j] > best_remaining:
                best_knapsack <- j
                best_remaining <- remaining[j]
        if best_knapsack is not None:
            assignment[i] <- best_knapsack
            remaining[best_knapsack] -= w[i]
            total_value += v[i]

    return assignment, total_value
```

---

## 3. Illustrative Instance

Consider $n = 5$ items and $m = 2$ knapsacks with capacities $C_1 = 10$, $C_2 = 8$:

| Item | Weight | Value | Density |
|------|--------|-------|---------|
| A | 3 | 9 | 3.00 |
| B | 4 | 10 | 2.50 |
| C | 5 | 11 | 2.20 |
| D | 6 | 12 | 2.00 |
| E | 2 | 3 | 1.50 |

Greedy assigns: A to KS1 (rem: 7), B to KS1 (rem: 3), C to KS2 (rem: 3),
E to KS1 (rem: 1). D does not fit anywhere. Total value = 33.

Optimal (by ILP): A + B + E in KS1 (weight 9), C in KS2 (weight 5) = 33,
or D + E in KS1, C in KS2 = 26. Greedy matches optimal here.

---

## 4. Implementations in This Repository

```
multiple_knapsack/
├── instance.py                        # MultipleKnapsackInstance, MultipleKnapsackSolution
├── exact/
│   └── ilp_mk.py                      # ILP formulation
├── heuristics/
│   └── greedy_mk.py                   # Greedy assignment
└── tests/
    └── test_multiple_knapsack.py      # MKP test suite
```

---

## 5. Key References

- Martello, S. & Toth, P. (1990). *Knapsack Problems: Algorithms and Computer Implementations*. Wiley.
- Kellerer, H., Pferschy, U. & Pisinger, D. (2004). *Knapsack Problems*. Springer.
- Pisinger, D. (1999). An exact algorithm for large multiple knapsack problems. *European J. Oper. Res.*, 114(3), 528-541.
- Chekuri, C. & Khanna, S. (2005). A polynomial time approximation scheme for the multiple knapsack problem. *SIAM J. Comput.*, 35(3), 713-728.
