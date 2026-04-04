# Multiple Knapsack Problem (MKP)

## 1. Problem Definition

- **Input:** $n$ items with weights $w_i$ and values $v_i$; $m$ knapsacks with capacities $C_j$
- **Decision:** Assign items to knapsacks (each item to at most one)
- **Objective:** Maximize total value of assigned items
- **Constraints:** Total weight per knapsack $\leq C_j$
- **Classification:** NP-hard (generalizes 0-1 knapsack)

---

## 2. Solution Methods

| Method | Type | Complexity | Description |
|--------|------|-----------|-------------|
| ILP (HiGHS) | Exact | Exponential worst | MILP via `scipy.optimize.milp` |
| Greedy MKP | Heuristic | $O(n \cdot m \log n)$ | Assign by value-density to largest-capacity knapsack |

---

## 3. Implementations in This Repository

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

## 4. Key References

- Martello, S. & Toth, P. (1990). *Knapsack Problems: Algorithms and Computer Implementations*. Wiley.
- Kellerer, H., Pferschy, U. & Pisinger, D. (2004). *Knapsack Problems*. Springer.
