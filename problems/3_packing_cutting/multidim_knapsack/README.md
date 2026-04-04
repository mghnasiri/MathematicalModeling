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

---

## 3. Implementations in This Repository

```
multidim_knapsack/
├── instance.py                        # MultiDimKnapsackInstance, MultiDimKnapsackSolution
├── heuristics/
│   └── greedy_mdk.py                  # Greedy multi-dimensional knapsack
└── tests/
    └── test_multidim_knapsack.py      # MdKP test suite
```

---

## 4. Key References

- Fréville, A. (2004). The multidimensional 0-1 knapsack problem: An overview. *European J. Oper. Res.*, 155(1), 1-21.
- Chu, P.C. & Beasley, J.E. (1998). A genetic algorithm for the multidimensional knapsack problem. *J. Heuristics*, 4(1), 63-86.
