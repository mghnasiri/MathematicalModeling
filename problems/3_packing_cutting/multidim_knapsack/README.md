# Multi-Dimensional Knapsack Problem (MdKP)

## 1. Problem Definition

- **Input:** $n$ items with values $v_i$ consuming $d$ resources (weight $w_{id}$ per dimension), $d$ capacity constraints $C_1, \ldots, C_d$
- **Decision:** Binary selection $x_i \in \{0, 1\}$
- **Objective:** Maximize $\sum v_i x_i$
- **Constraints:** $\sum_i w_{id} x_i \leq C_d$ for all dimensions $d$
- **Classification:** NP-hard (harder than 0-1 KP due to multiple constraints)

---

## 2. Solution Methods

| Method | Type | Complexity | Description |
|--------|------|-----------|-------------|
| Greedy MdKP | Heuristic | $O(n \log n \cdot d)$ | Value-density with multi-dimensional feasibility checks |
| LP Relaxation + Rounding | Heuristic | $O(\text{LP})$ | Solve LP relaxation, round fractional variables |
| Genetic Algorithm | Metaheuristic | Problem-dependent | Binary encoding with repair for multi-constraint feasibility |

### Pseudo-Utility Greedy

The core greedy heuristic computes a pseudo-utility score for each item that accounts
for consumption across all $d$ resource dimensions, then selects items in decreasing
order of this score while maintaining feasibility across every constraint.

```
GREEDY-MDKP(items, capacities[1..d]):
    remaining[1..d] <- capacities[1..d]
    selected <- {}

    // Compute pseudo-utility: value per aggregate resource consumption
    for each item i:
        score[i] <- v[i] / sum_{j=1}^{d} (w[i][j] / capacities[j])

    Sort items by score descending

    for each item i in sorted order:
        feasible <- true
        for j = 1 to d:
            if w[i][j] > remaining[j]:
                feasible <- false
                break
        if feasible:
            selected <- selected + {i}
            for j = 1 to d:
                remaining[j] <- remaining[j] - w[i][j]

    return selected
```

The pseudo-utility normalization `w[i][j] / capacities[j]` ensures items consuming
a large fraction of a tight resource are penalized appropriately.

---

## 3. Illustrative Instance

Consider $n = 4$ items with $d = 2$ resource dimensions and capacities $C_1 = 10$, $C_2 = 8$:

| Item | Value | Weight (dim 1) | Weight (dim 2) | Pseudo-Utility |
|------|-------|----------------|----------------|----------------|
| A | 12 | 4 | 3 | 12 / (4/10 + 3/8) = 15.48 |
| B | 10 | 5 | 2 | 10 / (5/10 + 2/8) = 13.33 |
| C | 8 | 3 | 4 | 8 / (3/10 + 4/8) = 10.32 |
| D | 6 | 2 | 5 | 6 / (2/10 + 5/8) = 7.06 |

Greedy selects A (remaining: 6, 5), then B (remaining: 1, 3), then stops
(C needs 3 in dim 1 but only 1 remains). Total value = 22.

---

## 4. Implementations in This Repository

```
multidim_knapsack/
├── instance.py                        # MultiDimKnapsackInstance, MultiDimKnapsackSolution
├── heuristics/
│   └── greedy_mdk.py                  # Greedy multi-dimensional knapsack
└── tests/
    └── test_multidim_knapsack.py      # MdKP test suite
```

---

## 5. Key References

- Fréville, A. (2004). The multidimensional 0-1 knapsack problem: An overview. *European J. Oper. Res.*, 155(1), 1-21.
- Chu, P.C. & Beasley, J.E. (1998). A genetic algorithm for the multidimensional knapsack problem. *J. Heuristics*, 4(1), 63-86.
- Pirkul, H. (1987). A heuristic solution procedure for the multiconstraint zero-one knapsack problem. *Naval Research Logistics*, 34(2), 161-172.
- Loulou, R. & Michaelides, E. (1979). New greedy-like heuristics for the multidimensional 0-1 knapsack problem. *Oper. Res.*, 27(6), 1101-1114.
