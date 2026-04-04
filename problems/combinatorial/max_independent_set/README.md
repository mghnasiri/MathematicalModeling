# Maximum Independent Set Problem (MIS)

## 1. Problem Definition

- **Input:** Undirected graph $G = (V, E)$
- **Decision:** Find a subset $S \subseteq V$ such that no two vertices in $S$ are adjacent
- **Objective:** Maximize $|S|$ (the independence number $\alpha(G)$)
- **Classification:** NP-hard (Karp, 1972). Inapproximable within $n^{1-\epsilon}$ for any $\epsilon > 0$ unless P = NP.

---

## 2. Mathematical Formulation

### Notation Table

| Symbol | Definition |
|--------|-----------|
| $V, E$ | Vertex and edge sets |
| $n = \|V\|$ | Number of vertices |
| $S$ | Independent set |
| $\alpha(G)$ | Independence number |

### ILP Formulation

$$\max \sum_{v \in V} x_v \tag{1}$$

$$x_u + x_v \leq 1 \quad \forall (u,v) \in E \tag{2}$$

$$x_v \in \{0,1\} \tag{3}$$

### Relationship to Other Problems

- **Maximum Clique** on complement graph $\bar{G}$: $\alpha(G) = \omega(\bar{G})$
- **Minimum Vertex Cover**: $\alpha(G) + \tau(G) = n$ (Gallai's theorem)

### Small Illustrative Instance

```
Graph: 5 vertices
Edges: (0,1), (1,2), (2,3), (3,4), (0,4)  (cycle C₅)
α(C₅) = 2: e.g., S = {0, 2} or {1, 3}
```

---

## 3. Solution Methods

| Method | Type | Complexity | Description |
|--------|------|-----------|-------------|
| Branch and Bound | Exact | Exponential | Backtracking with greedy upper bounds |
| Greedy MIS | Heuristic | $O(V + E)$ | Add minimum-degree vertex, remove neighbors |

### Greedy MIS

Repeatedly select the vertex with the fewest neighbors, add it to $S$, and remove it and all its neighbors from the graph. Runs in $O(V + E)$ but provides no approximation guarantee for general graphs.

```
GREEDY-MIS(G = (V, E)):
  S ← ∅
  G' ← copy of G
  while G' has vertices:
    v ← vertex in G' with minimum degree
    S ← S ∪ {v}
    remove v and all neighbors N(v) from G'
  return S
```

- Halldorsson, M.M. & Radhakrishnan, J. (1997). Greed is good: Approximating independent sets in sparse and bounded-degree graphs. *Algorithmica*, 18(1), 145-163.

### Branch and Bound

Recursively decide whether to include or exclude each vertex. Upper bound from fractional relaxation or graph coloring. Prune branches where bound $\leq$ incumbent.

---

## 4. Implementations in This Repository

```
max_independent_set/
├── instance.py                    # MISInstance, MISSolution
├── exact/
│   └── branch_and_bound.py        # B&B with greedy upper bounds
├── heuristics/
│   └── greedy_mis.py              # Greedy minimum-degree selection
└── tests/
    └── test_mis.py                # MIS test suite
```

---

## 5. Key References

- Garey, M.R. & Johnson, D.S. (1979). *Computers and Intractability: A Guide to the Theory of NP-Completeness*. W.H. Freeman.
- Boppana, R. & Halldorsson, M.M. (1992). Approximating maximum independent sets by excluding subgraphs. *BIT Numerical Mathematics*, 32(2), 180-196.
- Tarjan, R.E. & Trojanowski, A.E. (1977). Finding a maximum independent set. *SIAM J. Comput.*, 6(3), 537-546.

---

## 6. Notes

- For bounded-degree graphs ($\Delta \leq d$), the greedy MIS achieves an $O(n/d)$ approximation.
- The complement relationship $\alpha(G) = \omega(\bar{G})$ allows reuse of maximum clique solvers (e.g., Bron-Kerbosch) for MIS.
- Xiao, M. & Nagamochi, H. (2017). Exact algorithms for maximum independent set. *Information and Computation*, 255, 126-146.
