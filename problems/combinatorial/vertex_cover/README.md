# Minimum Vertex Cover Problem (MVC)

## 1. Problem Definition

- **Input:** Undirected graph $G = (V, E)$
- **Decision:** Find a subset $S \subseteq V$ such that every edge has at least one endpoint in $S$
- **Objective:** Minimize $|S|$ (the vertex cover number $\tau(G)$)
- **Classification:** NP-hard (Karp, 1972). 2-approximable via maximal matching. No $(2 - \epsilon)$-approximation known (assuming UGC).

---

## 2. Mathematical Formulation

### Notation Table

| Symbol | Definition |
|--------|-----------|
| $V, E$ | Vertex and edge sets |
| $n = \|V\|$ | Number of vertices |
| $S$ | Vertex cover |
| $\tau(G)$ | Vertex cover number |

### ILP Formulation

$$\min \sum_{v \in V} x_v \tag{1}$$

$$x_u + x_v \geq 1 \quad \forall (u,v) \in E \tag{2}$$

$$x_v \in \{0,1\} \tag{3}$$

### Relationship to Other Problems

- **Maximum Independent Set**: $\tau(G) + \alpha(G) = n$ (Gallai's theorem)
- LP relaxation gives half-integral solution ($x_v \in \{0, 1/2, 1\}$)

### Small Illustrative Instance

```
Graph: 4 vertices, edges = {(0,1), (0,2), (1,3), (2,3)}
Minimum vertex cover: S = {0, 3} → τ(G) = 2
Check: (0,1)→0✓, (0,2)→0✓, (1,3)→3✓, (2,3)→3✓
```

---

## 3. Solution Methods

| Method | Type | Complexity | Description |
|--------|------|-----------|-------------|
| Greedy (degree) | Heuristic | $O(V + E)$ | Repeatedly add highest-degree vertex |
| 2-approximation | Heuristic | $O(V + E)$ | Take both endpoints of a maximal matching |

### 2-Approximation (Bar-Yehuda & Even, 1981)

Find any maximal matching $M$ (greedily). Include both endpoints of every matched edge. Since any vertex cover must include at least one endpoint of each matched edge, $|S| \leq 2 \cdot |M| \leq 2 \cdot \tau(G)$.

---

## 4. Implementations in This Repository

```
vertex_cover/
├── instance.py                    # VertexCoverInstance, VertexCoverSolution
├── heuristics/
│   └── greedy_vc.py               # Greedy degree-based and matching-based VC
└── tests/
    └── test_vertex_cover.py       # Vertex cover test suite
```

---

## 5. Key References

- Karp, R.M. (1972). Reducibility among combinatorial problems. In *Complexity of Computer Computations* (pp. 85-103). Plenum.
- Bar-Yehuda, R. & Even, S. (1981). A linear-time approximation algorithm for the weighted vertex cover problem. *J. Algorithms*, 2(2), 198-203. https://doi.org/10.1016/0196-6774(81)90020-1
- Dinur, I. & Safra, S. (2005). On the hardness of approximating minimum vertex cover. *Annals of Mathematics*, 162(1), 439-485.
