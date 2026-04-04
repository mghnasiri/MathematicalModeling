# Maximum Flow Problem (Max-Flow)

## 1. Problem Definition

- **Input:** Directed graph $G = (V, E)$ with edge capacities $c(u,v) \geq 0$, source $s$, sink $t$
- **Decision:** Assign flow $f(u,v)$ to each edge
- **Objective:** Maximize total flow from $s$ to $t$
- **Constraints:** (1) Capacity: $0 \leq f(u,v) \leq c(u,v)$. (2) Conservation: $\sum_u f(u,v) = \sum_w f(v,w)$ for all $v \neq s, t$.
- **Classification:** Polynomial — solvable in $O(VE^2)$ by Edmonds-Karp

### Key Theorems

- **Max-Flow Min-Cut Theorem** (Ford & Fulkerson, 1956): The maximum flow equals the minimum capacity of an $s$-$t$ cut.
- **LP Duality:** The max-flow LP dual is the min-cut LP. Strong duality holds, and both LPs have integral optimal solutions.

### Complexity

| Algorithm | Complexity | Notes |
|----------|-----------|-------|
| Edmonds-Karp | $O(VE^2)$ | BFS augmenting paths |
| Dinic's | $O(V^2 E)$ | Layered graph + blocking flows |
| Push-Relabel | $O(V^2 E)$ or $O(V^3)$ | FIFO variant |

---

## 2. Mathematical Formulation

### LP Formulation

$$\max \sum_{v: (s,v) \in E} f(s,v) \tag{1}$$

$$0 \leq f(u,v) \leq c(u,v) \quad \forall (u,v) \in E \tag{2}$$

$$\sum_{u} f(u,v) = \sum_{w} f(v,w) \quad \forall v \neq s, t \tag{3}$$

The constraint matrix is totally unimodular, so the LP always has an integral optimal solution.

---

## 3. Variants

| Variant | Directory |
|---------|-----------|
| Minimum Cost Flow | `variants/min_cost_flow/` |

### Minimum Cost Flow

Add cost $a(u,v)$ per unit flow. Minimize $\sum a(u,v) f(u,v)$ subject to flow conservation, capacity, and a required flow value. Generalizes both shortest path and max-flow.

---

## 4. Solution Methods

### Edmonds-Karp (1972)

Ford-Fulkerson with BFS to find shortest augmenting paths. Guarantees $O(VE)$ augmentations, each $O(E)$.

### Dinic's Algorithm (1970)

Build layered graph via BFS. Find blocking flows via DFS. Repeat until no augmenting path exists. $O(V^2 E)$.

---

## 5. Implementations in This Repository

```
max_flow/
├── instance.py                    # MaxFlowInstance, capacity matrix
├── exact/
│   ├── edmonds_karp.py            # Edmonds-Karp O(VE²), min-cut extraction
│   └── dinics.py                  # Dinic's algorithm O(V²E)
├── variants/
│   └── min_cost_flow/             # MCF variant
└── tests/
    ├── test_max_flow.py           # Edmonds-Karp tests
    └── test_dinics.py             # Dinic's tests
```

---

## 6. Key References

- Ford, L.R. & Fulkerson, D.R. (1956). Maximal flow through a network. *Canadian Journal of Mathematics*, 8, 399-404.
- Edmonds, J. & Karp, R.M. (1972). Theoretical improvements in algorithmic efficiency for network flow problems. *JACM*, 19(2), 248-264.
- Dinic, E.A. (1970). Algorithm for solution of a problem of maximum flow in networks with power estimation. *Soviet Mathematics Doklady*, 11, 1277-1280.
- Goldberg, A.V. & Tarjan, R.E. (1988). A new approach to the maximum-flow problem. *JACM*, 35(4), 921-940.
