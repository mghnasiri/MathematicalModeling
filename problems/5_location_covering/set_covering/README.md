# Set Covering Problem (SCP)

## 1. Problem Definition

- **Input:** Universe $U = \{1, \ldots, m\}$, collection of $n$ subsets $S_j$ with costs $c_j$
- **Decision:** Select a sub-collection of subsets
- **Objective:** Minimize total cost $\sum c_j x_j$
- **Constraints:** Every element in $U$ is covered by at least one selected subset
- **Classification:** NP-hard. Greedy achieves $\ln(m) + 1$ approximation (best possible unless P = NP).

---

## 2. Mathematical Formulation

$$\min \sum_{j=1}^{n} c_j x_j \tag{1}$$

$$\sum_{j: i \in S_j} x_j \geq 1 \quad \forall i \in U \tag{2}$$

$$x_j \in \{0, 1\} \tag{3}$$

---

## 3. Solution Methods

| Method | Type | Complexity | Description |
|--------|------|-----------|-------------|
| ILP (HiGHS) | Exact | Exponential worst | MILP via `scipy.optimize.milp` |
| Greedy SCP | Heuristic | $O(m \cdot n)$ | Select cheapest cost-per-new-element subset |

---

## 4. Implementations in This Repository

```
set_covering/
├── instance.py                    # SetCoveringInstance, SetCoveringSolution
├── exact/
│   └── ilp_scp.py                 # ILP formulation
├── heuristics/
│   └── greedy_scp.py              # Greedy set covering
└── tests/
    └── test_set_covering.py       # SCP test suite
```

---

## 5. Key References

- Chvatal, V. (1979). A greedy heuristic for the set-covering problem. *Math. Oper. Res.*, 4(3), 233-235.
- Caprara, A., Fischetti, M. & Toth, P. (1999). A heuristic method for the set covering problem. *Oper. Res.*, 47(5), 730-743.
