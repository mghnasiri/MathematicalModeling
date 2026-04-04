# Maximum Clique Problem

## 1. Problem Definition

- **Input:** Undirected graph $G = (V, E)$
- **Decision:** Find a subset $C \subseteq V$ such that every pair of vertices in $C$ is adjacent
- **Objective:** Maximize $|C|$ (the clique number $\omega(G)$)
- **Classification:** NP-hard (Karp, 1972). No polynomial-time constant-factor approximation unless P = NP.

---

## 2. Mathematical Formulation

### Notation Table

| Symbol | Definition |
|--------|-----------|
| $V, E$ | Vertex and edge sets |
| $n = \|V\|$ | Number of vertices |
| $C$ | Clique (complete subgraph) |
| $\omega(G)$ | Clique number (size of maximum clique) |

### ILP Formulation

$$\max \sum_{v \in V} x_v \tag{1}$$

$$x_u + x_v \leq 1 \quad \forall (u,v) \notin E \tag{2}$$

$$x_v \in \{0,1\} \tag{3}$$

### Relationship to Other Problems

- **Maximum Independent Set** on complement graph $\bar{G}$
- **Minimum Vertex Cover**: $\alpha(G) + \tau(G) = n$ (Gallai's theorem)

### Small Illustrative Instance

```
Graph: 5 vertices
Edges: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3), (3,4)
Maximum clique: {0, 1, 2, 3} → ω(G) = 4
```

---

## 3. Solution Methods

| Method | Type | Complexity | Description |
|--------|------|-----------|-------------|
| Bron-Kerbosch | Exact | $O(3^{n/3})$ | Backtracking with pivot selection |

### Bron-Kerbosch Algorithm (1973)

Recursive backtracking that maintains three sets: $R$ (current clique), $P$ (candidates), $X$ (already explored). At each step, branch on each vertex in $P$, adding it to $R$ and restricting $P$ and $X$ to its neighbors. Pivot selection reduces branching factor.

---

## 4. Implementations in This Repository

```
max_clique/
├── instance.py                    # MaxCliqueInstance, MaxCliqueSolution
│                                  #   - adjacency set per vertex
├── exact/
│   └── bron_kerbosch.py           # Bron-Kerbosch with pivoting
└── tests/
    └── test_max_clique.py         # Max clique test suite
```

---

## 5. Key References

- Bron, C. & Kerbosch, J. (1973). Algorithm 457: Finding all cliques of an undirected graph. *Comm. ACM*, 16(9), 575-577. https://doi.org/10.1145/362342.362367
- Tomita, E., Tanaka, A. & Takahashi, H. (2006). The worst-case time complexity for generating all maximal cliques and computational experiments. *Theor. Comp. Sci.*, 363(1), 28-42.
- Karp, R.M. (1972). Reducibility among combinatorial problems. In *Complexity of Computer Computations* (pp. 85-103). Plenum.
