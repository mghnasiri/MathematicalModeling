# Robust Shortest Path Problem

## 1. Problem Definition

- **Input:** Directed graph $G = (V, E)$ with $S$ edge-weight scenarios $w^s(e)$, source $s$, target $t$, scenario probabilities $p_s$
- **Decision:** Select an $s$-$t$ path $P$
- **Objective:** Minimize worst-case cost $\max_{s} \text{cost}_s(P)$ or worst-case regret $\max_{s} [\text{cost}_s(P) - \text{cost}_s(P^*_s)]$
- **Constraints:** $P$ must be a simple path from source to target
- **Classification:** Min-max cost: $O(S \cdot (V+E) \log V)$; Min-max regret: NP-hard (general intervals)

---

## 2. Mathematical Formulation

### Notation Table

| Symbol | Definition |
|--------|-----------|
| $V, E$ | Vertex and edge sets |
| $w^s(e)$ | Weight of edge $e$ under scenario $s$ |
| $S$ | Number of scenarios |
| $P$ | An $s$-$t$ path |
| $P^*_s$ | Optimal path under scenario $s$ |
| $\text{cost}_s(P)$ | Total weight of path $P$ under scenario $s$ |

### Robustness Criteria

| Criterion | Formulation | Complexity |
|-----------|-------------|------------|
| **Min-Max Cost** | $\min_P \max_{s \in S} \text{cost}_s(P)$ | Polynomial (discrete scenarios) |
| **Min-Max Regret** | $\min_P \max_{s \in S} [\text{cost}_s(P) - \text{cost}_s(P^*_s)]$ | NP-hard (general intervals) |
| **Expected Cost** | $\min_P \sum_s p_s \cdot \text{cost}_s(P)$ | Polynomial (weighted Dijkstra) |

### Small Illustrative Instance

```
Graph: 0 → 1 → 3, 0 → 2 → 3 (4 nodes, 4 edges)
Scenario 1 weights: [2, 5, 4, 3]  → Path 0-1-3: 7, Path 0-2-3: 7
Scenario 2 weights: [6, 1, 2, 8]  → Path 0-1-3: 7, Path 0-2-3: 10

Min-max cost: Path 0-1-3 (max=7), Path 0-2-3 (max=10) → choose 0-1-3
Min-max regret: regret(0-1-3) = max(7-7, 7-7) = 0 → optimal
```

---

## 3. Solution Methods

| Method | Criterion | Complexity | Description |
|--------|-----------|-----------|-------------|
| Label-Setting | Min-Max Cost | $O(S \cdot (V+E) \log V)$ | Multi-objective Dijkstra with dominance pruning |
| Scenario Enumeration | Min-Max Cost | $O(S \cdot (V+E) \log V)$ | Dijkstra per scenario, cross-evaluate |
| Regret Enumeration | Min-Max Regret | $O(S^2 \cdot (V+E) \log V)$ | Evaluate candidate paths against per-scenario optima |
| Midpoint Heuristic | Min-Max Regret | $O((V+E) \log V)$ | Shortest path on mean-weight graph |

### Label-Setting (Min-Max Cost)

Extends Dijkstra with vector labels $(c_1, c_2, \ldots, c_S)$ per node. A label dominates another if it is componentwise $\leq$. Prune dominated labels to limit explosion.

### Midpoint Heuristic

Compute mean weights $\bar{w}(e) = \sum_s p_s w^s(e)$, then run standard Dijkstra. Fast but ignores worst-case structure.

---

## 4. Implementations in This Repository

```
robust_shortest_path/
├── instance.py                    # RobustSPInstance, RobustSPSolution
│                                  #   - adjacency_list(), path_cost(), max_cost()
│                                  #   - random() factory
├── exact/
│   └── minmax_cost.py             # Label-setting + scenario enumeration
├── heuristics/
│   └── minmax_regret.py           # Regret enumeration, midpoint heuristic
└── tests/
    └── test_robust_sp.py          # 13 tests, 4 test classes
```

---

## 5. Key References

- Kouvelis, P. & Yu, G. (1997). *Robust Discrete Optimization and Its Applications*. Springer. https://doi.org/10.1007/978-1-4757-2620-6
- Bertsimas, D. & Sim, M. (2003). Robust discrete optimization and network flows. *Math. Program.*, 98(1-3), 49-71. https://doi.org/10.1007/s10107-003-0396-4
- Averbakh, I. & Lebedev, V. (2004). Interval data minmax regret network optimization problems. *Discrete Appl. Math.*, 138(3), 289-301. https://doi.org/10.1016/S0166-218X(03)00462-1
- Montemanni, R. & Gambardella, L.M. (2005). The robust shortest path problem with interval data via Benders decomposition. *4OR*, 3(4), 315-328.
