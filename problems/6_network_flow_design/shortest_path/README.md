# Shortest Path Problem (SPP)

## 1. Problem Definition

- **Input:** Directed graph $G = (V, E)$ with edge weights $w(u,v)$, source $s$, target $t$
- **Decision:** Find a path from $s$ to $t$
- **Objective:** Minimize total path weight $\sum_{(u,v) \in P} w(u,v)$
- **Constraints:** Path must be simple (no cycles if negative weights exist)
- **Classification:** Polynomial — solvable in $O((V+E)\log V)$ for non-negative weights

### Complexity

| Variant | Algorithm | Complexity |
|---------|----------|-----------|
| Non-negative weights | Dijkstra | $O((V+E) \log V)$ |
| General weights | Bellman-Ford | $O(VE)$ |
| DAG | Topological sort + relax | $O(V+E)$ |
| All-pairs | Floyd-Warshall | $O(V^3)$ |
| Negative cycle detection | Bellman-Ford | $O(VE)$ |

---

## 2. Mathematical Formulation

### LP Formulation (shortest $s$-$t$ path)

$$\min \sum_{(u,v) \in E} w(u,v) \cdot x_{uv} \tag{1}$$

$$\sum_{v: (s,v) \in E} x_{sv} - \sum_{u: (u,s) \in E} x_{us} = 1 \quad \text{(flow out of source)} \tag{2}$$

$$\sum_{v: (t,v) \in E} x_{tv} - \sum_{u: (u,t) \in E} x_{ut} = -1 \quad \text{(flow into sink)} \tag{3}$$

$$\sum_{v} x_{uv} - \sum_{v} x_{vu} = 0 \quad \forall u \neq s, t \quad \text{(conservation)} \tag{4}$$

$$x_{uv} \geq 0 \tag{5}$$

The LP relaxation is always integral (totally unimodular constraint matrix).

---

## 3. Variants

| Variant | Directory |
|---------|-----------|
| All-Pairs Shortest Path | `variants/all_pairs/` |

---

## 4. Solution Methods

### Dijkstra's Algorithm (non-negative weights)

```
ALGORITHM Dijkstra(G, w, s)
  dist[v] ← ∞ for all v; dist[s] ← 0
  Q ← min-priority-queue with all vertices
  WHILE Q not empty:
    u ← extract-min(Q)
    FOR each neighbor v of u:
      IF dist[u] + w(u,v) < dist[v]:
        dist[v] ← dist[u] + w(u,v)
        decrease-key(Q, v, dist[v])
  RETURN dist
```

### Bellman-Ford (general weights, negative cycle detection)

```
ALGORITHM BellmanFord(G, w, s)
  dist[v] ← ∞ for all v; dist[s] ← 0
  FOR i = 1 TO |V| - 1:
    FOR each edge (u,v):
      IF dist[u] + w(u,v) < dist[v]:
        dist[v] ← dist[u] + w(u,v)
  // Negative cycle check:
  FOR each edge (u,v):
    IF dist[u] + w(u,v) < dist[v]:
      RETURN "Negative cycle detected"
  RETURN dist
```

---

## 5. Implementations in This Repository

```
shortest_path/
├── instance.py                    # ShortestPathInstance, edge/matrix creation
├── exact/
│   ├── dijkstra.py                # Dijkstra O((V+E)log V)
│   └── bellman_ford.py            # Bellman-Ford O(VE), negative cycles
├── variants/
│   └── all_pairs/                 # Floyd-Warshall / Johnson's
└── tests/
    └── test_shortest_path.py      # 21 tests
```

---

## 6. Key References

- Dijkstra, E.W. (1959). A note on two problems in connexion with graphs. *Numerische Mathematik*, 1(1), 269-271.
- Bellman, R. (1958). On a routing problem. *Quarterly of Applied Mathematics*, 16, 87-90.
- Ford, L.R. (1956). Network flow theory. *RAND Corporation Report P-923*.
- Fredman, M.L. & Tarjan, R.E. (1987). Fibonacci heaps and their uses in improved network optimization algorithms. *JACM*, 34(3), 596-615.
