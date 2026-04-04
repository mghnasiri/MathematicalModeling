# Simple Assembly Line Balancing Problem (SALBP-1)

## 1. Problem Definition

- **Input:** $n$ tasks with processing times $t_j$, precedence constraints (DAG), cycle time $C$
- **Decision:** Assign each task to a workstation
- **Objective:** Minimize the number of workstations
- **Constraints:** Precedence respected; total task time per station $\leq C$
- **Classification:** NP-hard (Wee & Magazine, 1982)

---

## 2. Mathematical Formulation

$$\min \sum_{k=1}^{K} y_k \tag{1}$$

$$\sum_{k=1}^{K} x_{jk} = 1 \quad \forall j \tag{2}$$

$$\sum_{j=1}^{n} t_j x_{jk} \leq C \cdot y_k \quad \forall k \tag{3}$$

$$\sum_{k} k \cdot x_{ik} \leq \sum_{k} k \cdot x_{jk} \quad \forall (i,j) \in \text{precedence} \tag{4}$$

---

## 3. Solution Methods

| Method | Type | Complexity | Description |
|--------|------|-----------|-------------|
| Ranked Positional Weight | Heuristic | $O(n \log n)$ | Priority = task time + sum of successor times |

---

## 4. Implementations in This Repository

```
assembly_line_balancing/
├── instance.py                    # SALBPInstance, SALBPSolution
├── heuristics/
│   └── rpw.py                     # Ranked Positional Weight heuristic
└── tests/
    └── test_salbp.py              # SALBP test suite
```

---

## 5. Key References

- Scholl, A. & Becker, C. (2006). State-of-the-art exact and heuristic solution procedures for SALBP. *European J. Oper. Res.*, 168(3), 666-693.
- Wee, T.S. & Magazine, M.J. (1982). Assembly line balancing as generalized bin packing. *Oper. Res. Lett.*, 1(2), 56-58.
