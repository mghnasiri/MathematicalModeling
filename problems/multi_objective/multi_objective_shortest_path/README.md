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
