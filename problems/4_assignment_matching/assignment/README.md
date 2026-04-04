# Linear Assignment Problem (LAP)

## 1. Problem Definition

- **Input:** An $n \times n$ cost matrix $C$ where $c_{ij}$ = cost of assigning agent $i$ to task $j$
- **Decision:** Find a one-to-one assignment (permutation $\sigma$)
- **Objective:** Minimize total cost $\sum_{i=1}^{n} c_{i,\sigma(i)}$
- **Constraints:** Each agent assigned to exactly one task; each task assigned to exactly one agent
- **Classification:** Polynomial — $O(n^3)$ via Hungarian method

### Complexity

| Variant | Complexity | Reference |
|---------|-----------|-----------|
| Square LAP ($n \times n$) | $O(n^3)$ | Kuhn (1955), Munkres (1957) |
| Rectangular LAP | $O(n^2 m)$ | Pad with dummy rows/columns |
| Bottleneck Assignment | $O(n^{2.5})$ | Threshold-based |
| Quadratic Assignment (QAP) | NP-hard | Koopmans & Beckmann (1957) |

---

## 2. Mathematical Formulation

$$\min \sum_{i=1}^{n} \sum_{j=1}^{n} c_{ij} x_{ij} \tag{1}$$

$$\sum_{j=1}^{n} x_{ij} = 1 \quad \forall i \quad \text{(each agent to one task)} \tag{2}$$

$$\sum_{i=1}^{n} x_{ij} = 1 \quad \forall j \quad \text{(each task to one agent)} \tag{3}$$

$$x_{ij} \in \{0, 1\} \tag{4}$$

The constraint matrix is totally unimodular — the LP relaxation always gives an integral solution. This is why the assignment problem is polynomial despite being an integer program.

---

## 3. Variants

| Variant | Directory | Key Difference |
|---------|-----------|---------------|
| Generalized AP | `variants/generalized/` | Agents can handle multiple tasks; capacity constraints |
| Max Weight Matching | `variants/max_weight_matching/` | Maximize weight in a bipartite graph |
| Quadratic AP (QAP) | `variants/quadratic/` | Cost depends on pairs of assignments — NP-hard |

---

## 4. Solution Methods

### Hungarian Algorithm (Kuhn-Munkres, 1955/1957)

**Idea:** Iteratively reduce the cost matrix using dual variables (row/column reductions) and find augmenting paths in the equality subgraph.

**Complexity:** $O(n^3)$ using shortest-augmenting-path variant.

```
HUNGARIAN(C, n):
  u[1..n] ← 0; v[1..n] ← 0          // dual variables (row/col potentials)
  match[1..n] ← -1                     // column matched to each row
  for i ← 1 to n:                      // augment one row at a time
    find shortest augmenting path from row i
      in the equality subgraph {(i,j): C[i][j] - u[i] - v[j] = 0}
      using Dijkstra on reduced costs
    update potentials u, v along the path
    augment matching along the path
  return match, Σ C[i][match[i]]
```

### Greedy Assignment

Assign the cheapest available (agent, task) pair iteratively. $O(n^2)$. No optimality guarantee.

---

## 5. Implementations in This Repository

```
assignment/
├── instance.py                    # AssignmentInstance, cost matrix
├── exact/
│   └── hungarian.py               # Hungarian (Kuhn-Munkres) O(n³)
���── heuristics/
│   └── greedy_assignment.py       # Greedy min-cost O(n²)
├─�� variants/
│   ├── generalized/               # GAP
│   ├── max_weight_matching/       # Maximum weight bipartite matching
│   └── quadratic/                 # QAP
└─��� tests/
    └── test_assignment.py         # 17 tests
```

---

## 6. Key References

- Kuhn, H.W. (1955). The Hungarian method for the assignment problem. *Naval Research Logistics Quarterly*, 2(1-2), 83-97.
- Munkres, J. (1957). Algorithms for the assignment and transportation problems. *SIAM Journal*, 5(1), 32-38.
- Burkard, R.E., Dell'Amico, M. & Martello, S. (2009). *Assignment Problems*. SIAM.
- Koopmans, T.C. & Beckmann, M. (1957). Assignment problems and the location of economic activities. *Econometrica*, 25(1), 53-76.

---

## 7. Notes

- The shortest-augmenting-path variant of the Hungarian algorithm (Jonker & Volgenant, 1987) is the fastest practical implementation at $O(n^3)$.
- Jonker, R. & Volgenant, A. (1987). A shortest augmenting path algorithm for dense and sparse linear assignment problems. *Computing*, 38(4), 325-340.
