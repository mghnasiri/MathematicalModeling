# Graph Coloring Problem (GCP)

## 1. Problem Definition

- **Input:** Undirected graph $G = (V, E)$
- **Decision:** Assign a color $c(v) \in \{1, \ldots, k\}$ to each vertex $v$
- **Objective:** Minimize the number of colors $k$ used (the chromatic number $\chi(G)$)
- **Constraints:** Adjacent vertices must receive different colors: $c(u) \neq c(v)$ for all $(u, v) \in E$
- **Classification:** NP-hard. Even deciding if $\chi(G) \leq 3$ is NP-complete.

---

## 2. Mathematical Formulation

### Notation Table

| Symbol | Definition |
|--------|-----------|
| $V, E$ | Vertex and edge sets |
| $n = \|V\|$ | Number of vertices |
| $k$ | Number of colors |
| $c(v)$ | Color assigned to vertex $v$ |
| $\chi(G)$ | Chromatic number of $G$ |

### ILP Formulation

$$\min \sum_{j=1}^{n} w_j \tag{1}$$

$$\sum_{j=1}^{n} x_{vj} = 1 \quad \forall v \in V \tag{2}$$

$$x_{uj} + x_{vj} \leq w_j \quad \forall (u,v) \in E, \; \forall j \tag{3}$$

$$x_{vj} \in \{0,1\}, \quad w_j \in \{0,1\} \tag{4}$$

where $x_{vj} = 1$ if vertex $v$ gets color $j$, and $w_j = 1$ if color $j$ is used.

### Small Illustrative Instance

```
Graph: 4 vertices, edges = {(0,1), (1,2), (2,3), (0,3), (0,2)}
This is K₄ minus edge (1,3).
χ(G) = 3: c = [1, 2, 3, 2]
```

---

## 3. Solution Methods

| Method | Type | Complexity | Description |
|--------|------|-----------|-------------|
| Greedy coloring | Heuristic | $O(V + E)$ | Assign smallest available color per vertex |
| DSatur (Brélaz) | Heuristic | $O(V^2)$ | Greedy by saturation degree (most constrained first) |

### DSatur (Degree of Saturation)

Select the uncolored vertex with the highest saturation degree (number of distinct colors among its neighbors). Break ties by graph degree. Assign the smallest feasible color. Optimal for bipartite graphs and interval graphs.

---

## 4. Implementations in This Repository

```
graph_coloring/
├── instance.py                    # GraphColoringInstance, GraphColoringSolution
├── heuristics/
│   └── greedy_coloring.py         # Greedy and DSatur coloring
└── tests/
    └── test_graph_coloring.py     # Graph coloring test suite
```

---

## 5. Key References

- Brélaz, D. (1979). New methods to color the vertices of a graph. *Comm. ACM*, 22(4), 251-256. https://doi.org/10.1145/359094.359101
- Jensen, T.R. & Toft, B. (2011). *Graph Coloring Problems*. Wiley.
- Garey, M.R. & Johnson, D.S. (1979). *Computers and Intractability*. W.H. Freeman.
