# Multi-Objective Shortest Path Problem (MOSP)

## 1. Problem Definition

- **Input:** Directed graph $G = (V, E)$ with $k$ cost vectors per edge, source $s$, target $t$
- **Decision:** Path from $s$ to $t$
- **Objective:** Find all Pareto-optimal paths minimizing the $k$ cost objectives simultaneously
- **Classification:** NP-hard in general; polynomial for fixed $k$ with bounded edge costs

---

## 2. Mathematical Formulation

### Notation Table

| Symbol | Definition |
|--------|-----------|
| $n$ | Number of nodes |
| $k$ | Number of objectives |
| $c^\ell(e)$ | Cost of edge $e$ under objective $\ell$ |
| $s, t$ | Source and target nodes |

### Multi-Objective Formulation

$$\min \left( \sum_{e \in P} c^1(e), \ldots, \sum_{e \in P} c^k(e) \right) \tag{1}$$

over all $s$-$t$ paths $P$. A path $P_1$ **dominates** $P_2$ if $c^\ell(P_1) \leq c^\ell(P_2)$ for all $\ell$ with at least one strict inequality.

---

## 3. Solution Methods

| Method | Type | Complexity | Description |
|--------|------|-----------|-------------|
| Label-setting | Exact | $O(n \cdot L_{\max})$ | Multi-objective Dijkstra with dominance pruning |

### Label-Setting Algorithm

Extends Dijkstra to maintain a set of non-dominated labels (cost vectors) per node. At each step, expand the label with the smallest lexicographic cost. Prune labels dominated by existing labels at the same node. Returns all Pareto-optimal paths.

The number of Pareto-optimal labels $L_{\max}$ can be exponential in the worst case but is typically manageable in practice.

#### Label-Setting Pseudocode

```
LABEL_SETTING_MOSP(G, source s, target t, k objectives):
    labels[s] = {(0, 0, ..., 0)}       # zero-cost label at source
    labels[v] = {} for all v != s
    Q = priority queue with (0-vector, s)

    while Q is not empty:
        (cost_vector, u) = extract_min(Q)  # min by first objective (lex)
        if cost_vector is dominated at u:
            continue
        if u == t:
            record cost_vector as Pareto-optimal
            continue
        for each edge (u, v) with cost c(u,v):
            new_cost = cost_vector + c(u,v)
            if new_cost is not dominated by any label in labels[v]:
                remove labels in labels[v] dominated by new_cost
                labels[v] = labels[v] ∪ {new_cost}
                insert (new_cost, v) into Q
    return all non-dominated labels at t
```

**Complexity:** $O(n \cdot L_{\max} \cdot \log(n \cdot L_{\max}))$ where $L_{\max}$ is the maximum number of Pareto-optimal labels per node. In the worst case $L_{\max}$ is exponential, but for two objectives with bounded integer costs, $L_{\max} \leq C_{\max}$ (the maximum single-objective cost).

### Small Illustrative Instance

```
4 nodes, 2 objectives, edges with (cost1, cost2):
0→1: (1, 4),  0→2: (3, 1),  1→3: (2, 1),  2→3: (1, 3)
Paths s=0, t=3:
  0→1→3: (3, 5)
  0→2→3: (4, 4)
Both are Pareto-optimal (neither dominates the other).
```

### Applications

- **Transportation planning** (minimizing both travel time and fuel cost)
- **Network routing** (balancing latency and bandwidth usage)
- **Emergency evacuation** (minimizing both distance and risk exposure)

---

## 4. Implementations in This Repository

```
multi_objective_shortest_path/
├── instance.py                    # MultiObjectiveSPInstance, MultiObjectiveSPSolution
│                                  #   - Fields: n, n_objectives, edges (with cost tuples),
│                                  #     source, target
├── exact/
│   └── label_setting.py           # Multi-objective label-setting (Pareto Dijkstra)
└── tests/
    └── test_mosp.py               # MOSP test suite
```

---

## 5. Key References

- Hansen, P. (1980). Bicriterion path problems. In *Multiple Criteria Decision Making Theory and Application* (pp. 109-127). Springer.
- Martins, E.Q.V. (1984). On a multicriteria shortest path problem. *European J. Oper. Res.*, 16(2), 236-245.
- Ehrgott, M. (2005). *Multicriteria Optimization*. 2nd ed. Springer.
- Raith, A. & Ehrgott, M. (2009). A comparison of solution strategies for biobjective shortest path problems. *Computers & Oper. Res.*, 36(4), 1299-1331.
- Skriver, A.J.V. & Andersen, K.A. (2000). A label correcting approach for solving bicriterion shortest-path problems. *Computers & Oper. Res.*, 27(6), 507-524.
