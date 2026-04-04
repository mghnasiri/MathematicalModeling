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
| Arc-commodity LP | Exact | $O(K \cdot |E| \cdot |V|)$ variables | Full LP via `scipy.optimize.linprog` |
| Path-based LP | Exact | Exponential columns | Column generation with shortest-path pricing |
| Decomposition | Heuristic | Problem-dependent | Route commodities independently, then resolve conflicts |

### LP Formulation Structure

The arc-commodity formulation has one flow variable $f^k_e$ per commodity $k$ per edge $e$:

**Variables:** $f^k_e \geq 0$ for each commodity $k \in \{1, \ldots, K\}$ and edge $e \in E$

**Objective:** $\min \sum_{k=1}^{K} \sum_{e \in E} c^k_e \cdot f^k_e$

**Flow conservation** (for each commodity $k$ and node $v$):
$$\sum_{e \in \delta^+(v)} f^k_e - \sum_{e \in \delta^-(v)} f^k_e = \begin{cases} d_k & \text{if } v = s_k \\ -d_k & \text{if } v = t_k \\ 0 & \text{otherwise} \end{cases}$$

**Bundle (shared capacity) constraints** (for each edge $e$):
$$\sum_{k=1}^{K} f^k_e \leq u_e$$

```
BUILD-MCF-LP(G, commodities, capacities):
    // Variables: f[k][e] for each commodity k, edge e
    // Total variables: K * |E|
    // Total constraints: K * |V| (flow conservation) + |E| (capacity)

    A_eq rows <- []    // flow conservation (equality)
    A_ub rows <- []    // capacity (inequality)

    for each commodity k = 1 to K:
        for each node v in V:
            row <- zeros(K * |E|)
            for each outgoing edge e of v:
                row[k * |E| + index(e)] = +1
            for each incoming edge e to v:
                row[k * |E| + index(e)] = -1
            b_eq <- supply/demand of v for commodity k
            A_eq.append(row, b_eq)

    for each edge e in E:
        row <- zeros(K * |E|)
        for each commodity k:
            row[k * |E| + index(e)] = 1
        A_ub.append(row, u_e)

    return linprog(c, A_ub, b_ub, A_eq, b_eq)
```

The LP has $K \cdot |E|$ variables and $K \cdot |V| + |E|$ constraints.
For large instances, decomposition methods (Dantzig-Wolfe, Lagrangian relaxation)
are preferred to avoid building the full constraint matrix.

---

## 3. Illustrative Instance

Graph with 4 nodes, 5 edges, 2 commodities:

```
Edges: (1->2, cap=10), (1->3, cap=8), (2->4, cap=7), (3->4, cap=9), (2->3, cap=5)
Commodity 1: source=1, sink=4, demand=8
Commodity 2: source=1, sink=4, demand=5
```

All edge costs are 1. Total demand = 13 must not exceed any shared edge capacity.
One feasible routing: Commodity 1 via 1->2->4 (7 units) and 1->3->4 (1 unit);
Commodity 2 via 1->3->4 (5 units). Edge (1->2) carries 7, edge (2->4) carries 7,
edge (1->3) carries 6, edge (3->4) carries 6 -- all within capacity.

---

## 4. Implementations in This Repository

```
multi_commodity_flow/
├── instance.py                    # MCFInstance, MCFSolution
├── exact/
│   └── lp_formulation.py          # LP formulation for MCFP
└── tests/
    └── test_mcf.py                # MCFP test suite
```

---

## 5. Key References

- Ahuja, R.K., Magnanti, T.L. & Orlin, J.B. (1993). *Network Flows: Theory, Algorithms, and Applications*. Prentice Hall.
- Barnhart, C., Hane, C.A. & Vance, P.H. (2000). Using branch-and-price-and-cut to solve origin-destination integer multicommodity flow problems. *Oper. Res.*, 48(2), 318-326.
- Kennington, J.L. & Helgason, R.V. (1980). *Algorithms for Network Programming*. Wiley.
