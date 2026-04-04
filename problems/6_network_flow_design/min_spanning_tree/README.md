# Minimum Spanning Tree (MST)

## 1. Problem Definition

- **Input:** Undirected, weighted, connected graph $G = (V, E)$ with edge weights $w(e) \geq 0$
- **Decision:** Select $|V| - 1$ edges forming a spanning tree
- **Objective:** Minimize total edge weight $\sum_{e \in T} w(e)$
- **Constraints:** Selected edges must form a connected acyclic subgraph spanning all vertices
- **Classification:** Polynomial — solvable in $O(E \log E)$ or $O(E \log V)$

### Complexity

| Algorithm | Complexity | Key Data Structure |
|----------|-----------|-------------------|
| Kruskal's | $O(E \log E)$ | Union-Find |
| Prim's | $O(E \log V)$ | Binary heap |
| Prim's (Fibonacci) | $O(E + V \log V)$ | Fibonacci heap |

### Key Properties

- **Cut property:** The minimum-weight edge crossing any cut must be in the MST.
- **Cycle property:** The maximum-weight edge in any cycle cannot be in the MST.
- **Uniqueness:** If all edge weights are distinct, the MST is unique.

---

## 2. Mathematical Formulation

### LP Formulation

$$\min \sum_{e \in E} w(e) \cdot x_e \tag{1}$$

$$\sum_{e \in E} x_e = |V| - 1 \quad \text{(tree has } |V|-1 \text{ edges)} \tag{2}$$

$$\sum_{e \in E(S)} x_e \leq |S| - 1 \quad \forall S \subset V,\; |S| \geq 2 \quad \text{(subtour elimination)} \tag{3}$$

$$x_e \in \{0, 1\} \tag{4}$$

The LP relaxation is integral (matroid intersection).

---

## 3. Variants

| Variant | Directory |
|---------|-----------|
| Steiner Tree | `variants/steiner_tree/` |

### Steiner Tree Problem

Find the minimum-weight tree spanning a required subset $R \subseteq V$ (may use non-required Steiner vertices). NP-hard for general graphs; 1.39-approximable.

---

## 4. Solution Methods

### Kruskal's Algorithm (1956)

Sort edges by weight. Add each edge if it doesn't create a cycle (check via union-find).

```
ALGORITHM Kruskal(G)
  Sort edges by ascending weight
  T ← {}
  UF ← UnionFind(V)
  FOR each edge (u,v) in sorted order:
    IF UF.find(u) ≠ UF.find(v):
      T ← T ∪ {(u,v)}
      UF.union(u, v)
  RETURN T
```

### Prim's Algorithm (1957)

Grow tree from an arbitrary vertex. At each step, add the minimum-weight edge connecting the tree to a non-tree vertex.

```
ALGORITHM Prim(G, s)
  key[v] ← ∞ for all v; key[s] ← 0
  Q ← min-priority-queue with all vertices
  WHILE Q not empty:
    u ← extract-min(Q)
    FOR each neighbor v of u with edge weight w:
      IF v ∈ Q AND w < key[v]:
        key[v] ← w; parent[v] ← u
  RETURN parent tree
```

---

## 5. Implementations in This Repository

```
min_spanning_tree/
├── instance.py                    # MSTInstance, undirected graph
├── exact/
│   └── mst_algorithms.py         # Kruskal O(E log E), Prim O(E log V)
├── variants/
│   └── steiner_tree/              # Steiner Tree Problem
└── tests/
    └── test_mst.py                # 16 tests
```

---

## 6. Key References

- Kruskal, J.B. (1956). On the shortest spanning subtree of a graph and the traveling salesman problem. *Proceedings of the AMS*, 7(1), 48-50.
- Prim, R.C. (1957). Shortest connection networks and some generalizations. *Bell System Technical Journal*, 36(6), 1389-1401.
- Tarjan, R.E. (1983). *Data Structures and Network Algorithms*. SIAM.
- Chazelle, B. (2000). A minimum spanning tree algorithm with inverse-Ackermann type complexity. *JACM*, 47(6), 1028-1047.
