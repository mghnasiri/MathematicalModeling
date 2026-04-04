# Multi-Commodity Flow Problem (MCFP)

## 1. Problem Definition

- **Input:** Directed graph with edge capacities, $K$ commodities each with source $s_k$, sink $t_k$, and demand $d_k$
- **Decision:** Flow $f^k_e$ for each commodity $k$ on each edge $e$
- **Objective:** Minimize total flow cost (or find feasible flow)
- **Constraints:** Flow conservation per commodity; shared edge capacity $\sum_k f^k_e \leq u_e$
- **Classification:** **Polynomial** (LP formulation)

---

## 2. Solution Methods

| Method | Type | Complexity | Description |
|--------|------|-----------|-------------|
| LP formulation | Exact | Polynomial | Arc-commodity LP via `scipy.optimize.linprog` |

---

## 3. Implementations in This Repository

```
multi_commodity_flow/
├── instance.py                    # MCFInstance, MCFSolution
├── exact/
│   └── lp_formulation.py          # LP formulation for MCFP
└── tests/
    └── test_mcf.py                # MCFP test suite
```

---

## 4. Key References

- Ahuja, R.K., Magnanti, T.L. & Orlin, J.B. (1993). *Network Flows: Theory, Algorithms, and Applications*. Prentice Hall.
